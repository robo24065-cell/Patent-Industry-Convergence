# 3단계 독립 재계산: holdout_pred.parquet 로 지표 재계산, 기준선 독립 재적합, 권리자 군집 부트스트랩(청크), 산업 단위 순위 타당성
# 사용: python h02_recompute.py base | boot <seed> <B> | summary
import sys, os, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import roc_auc_score, average_precision_score
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(H, 'holdout_check')
WD = os.path.join(H, '_work_holdout', 'cache')
def p_at(y, p, f=0.01):
    k = max(1, int(len(y) * f)); return float(np.asarray(y)[np.argsort(-np.asarray(p), kind='stable')[:k]].mean())
mode = sys.argv[1]
if mode == 'base':
    # 기준선 독립 재구현(statsmodels 가 아닌 sklearn 이지만 코드 독립 작성): 학습 L2006-2016, 적용 L2019-2021
    from sklearn.linear_model import LogisticRegression
    cols = ['rgst_no', 'L', 'age', 'rgst_year', 'orig_type', 'ksic', 'claims', 'y_MAIN']
    tr = pd.concat([pd.read_parquet(f'{WD}/panel_L{y}.parquet', columns=cols) for y in range(2006, 2017)])
    ap = pd.concat([pd.read_parquet(f'{WD}/panel_L{y}.parquet', columns=cols) for y in (2019, 2020, 2021)])
    assert (tr.L + 3 <= 2019).all()
    med = tr.claims.median()
    def X(d, mu=None, sd=None, cats=None):
        n = pd.DataFrame({'age': d.age, 'rgst_year': d.rgst_year, 'lc': np.log1p(d.claims.fillna(med))})
        if mu is None: mu, sd = n.mean(), n.std()
        n = (n - mu) / sd
        dm = pd.get_dummies(d[['orig_type', 'ksic']].astype(str), dtype=float)
        if cats is None: cats = dm.columns
        dm = dm.reindex(columns=cats, fill_value=0.0)
        return pd.concat([n.reset_index(drop=True), dm.reset_index(drop=True)], axis=1).values, mu, sd, cats
    Xt, mu, sd, cats = X(tr); Xa, *_ = X(ap, mu, sd, cats)
    m = LogisticRegression(max_iter=3000, C=1.0).fit(Xt, tr.y_MAIN.values)
    ap['p_base_indep'] = m.predict_proba(Xa)[:, 1]
    pr = pd.read_parquet(os.path.join(H, 'holdout_pred.parquet'))
    mm = pr.merge(ap[['rgst_no', 'L', 'p_base_indep']], on=['rgst_no', 'L'], how='left')
    print('rows pred', len(pr), 'panel apply', len(ap), 'merged NA', int(mm.p_base_indep.isna().sum()))
    y = mm.y_MAIN.values
    r = dict(n=len(mm), pos=int(y.sum()), base_rate=float(y.mean()), n_owner_clusters=int(mm.cur.nunique()),
             n_pos_owner_clusters=int(mm[mm.y_MAIN == 1].cur.nunique()),
             auc_model=roc_auc_score(y, mm.p), auc_base_script=roc_auc_score(y, mm.p_base), auc_base_indep=roc_auc_score(y, mm.p_base_indep),
             prauc_model=average_precision_score(y, mm.p), prauc_base_script=average_precision_score(y, mm.p_base),
             prauc_base_indep=average_precision_score(y, mm.p_base_indep), p1_model=p_at(y, mm.p), p1_base_script=p_at(y, mm.p_base),
             p1_base_indep=p_at(y, mm.p_base_indep), corr_base=float(np.corrcoef(mm.p_base, mm.p_base_indep)[0, 1]),
             auc_owner_hat_only=roc_auc_score(y, mm.owner_hat), auc_p_pat_only=roc_auc_score(y, mm.p_pat))
    # 상위 10% 적중
    k = int(len(mm) * 0.10); top = np.argsort(-mm.p.values, kind='stable')[:k]
    r['top10_precision'] = float(y[top].mean()); r['top10_recall'] = float(y[top].sum() / y.sum())
    kb = np.argsort(-mm.p_base.values, kind='stable')[:k]; r['top10_precision_base'] = float(y[kb].mean()); r['top10_recall_base'] = float(y[kb].sum() / y.sum())
    k1 = int(len(mm) * 0.01); t1 = np.argsort(-mm.p.values, kind='stable')[:k1]; r['top1_recall'] = float(y[t1].sum() / y.sum())
    # L별
    r['per_L'] = {int(L): dict(n=int(len(d)), pos=int(d.y_MAIN.sum()), auc=roc_auc_score(d.y_MAIN, d.p), auc_base=roc_auc_score(d.y_MAIN, d.p_base),
                               prauc=average_precision_score(d.y_MAIN, d.p)) for L, d in mm.groupby('L')}
    # 양도인 유형별
    r['by_cur_type'] = {int(t): dict(n=int(len(d)), pos=int(d.y_MAIN.sum()), auc=(roc_auc_score(d.y_MAIN, d.p) if d.y_MAIN.nunique() > 1 else None))
                        for t, d in mm.groupby(mm.cur_type.fillna(0)) if len(d) >= 1000}
    # 부차 결과
    r['auc_y_ANY'] = roc_auc_score(mm.y_ANY, mm.p); r['auc_y_CORP'] = roc_auc_score(mm.y_CORP, mm.p)
    # 보정: 예측평균 vs 관측
    r['mean_pred'] = float(mm.p.mean()); r['obs_over_pred'] = float(y.mean() / mm.p.mean())
    # 산업(KSIC) 단위: 예측 평균 vs 관측 양도율 스피어만 (L별)
    from scipy.stats import spearmanr
    ks = {}
    for L, d in mm.groupby('L'):
        g = d.groupby('ksic').agg(pred=('p', 'mean'), obs=('y_MAIN', 'mean'), base=('p_base', 'mean'), n=('p', 'size'))
        rho, pv = spearmanr(g.pred, g.obs); rb, pb = spearmanr(g.base, g.obs)
        ks[int(L)] = dict(n_ksic=len(g), rho_model=rho, p_model=pv, rho_base=rb, p_base=pb)
    r['ksic_spearman'] = ks
    # 번들 구조: 양성의 (L, 양도인, 날짜) 번들 크기
    b = mm[mm.y_MAIN == 1].groupby(['L', 'cur', 'y_date']).size()
    r['bundles'] = dict(n_bundles=int(len(b)), max_bundle=int(b.max()), share_pos_in_bundles_ge50=float(b[b >= 50].sum() / b.sum()))
    top_sellers = mm[mm.y_MAIN == 1].groupby('cur').size().sort_values(ascending=False).head(10)
    r['top_sellers'] = top_sellers.to_dict()
    json.dump(r, open(f'{O}/recompute_base.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
elif mode == 'boot':
    seed, B = int(sys.argv[2]), int(sys.argv[3])
    mm = pd.read_parquet(os.path.join(H, 'holdout_pred.parquet'))
    y = mm.y_MAIN.values; p = mm.p.values; pb = mm.p_base.values
    rng = np.random.default_rng(1000 + seed)
    codes, uniq = pd.factorize(mm.cur.fillna('NA')); order = np.argsort(codes, kind='stable'); cs = codes[order]
    st = np.searchsorted(cs, np.arange(len(uniq))); en = np.searchsorted(cs, np.arange(len(uniq)), side='right')
    R = []
    for _ in range(B):
        idx = np.concatenate([order[st[c]:en[c]] for c in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        am, ab = roc_auc_score(yy, p[idx]), roc_auc_score(yy, pb[idx])
        R.append((am, ab, am - ab, (am - ab) / (1 - ab), average_precision_score(yy, p[idx]), average_precision_score(yy, pb[idx]), p_at(yy, p[idx])))
    pd.DataFrame(R, columns=['auc', 'auc_b', 'dauc', 'skill', 'pr', 'pr_b', 'p1']).to_parquet(f'{O}/boot_{seed}.parquet', index=False)
    print('saved', seed, len(R))
elif mode == 'summary':
    import glob
    R = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{O}/boot_*.parquet')])
    s = {c: [float(np.percentile(R[c], 2.5)), float(np.percentile(R[c], 97.5))] for c in R.columns}
    s['B'] = len(R); s['P_dauc_gt0'] = float((R.dauc > 0).mean())
    json.dump(s, open(f'{O}/boot_summary.json', 'w'), indent=1); print(json.dumps(s, indent=1))
