"""T3 기술셀 부상 예측 — 동결 모형 실행기 (2단계 동결, 2026-09-27)

모형: final_spec.json (v30 = 로지스틱 + XGB 분류 + 음이항 꼬리확률 구조모형의 t0 내 동일가중 순위평균, 중립 동률깨기 레이블)

사용 (트랙 폴더에서):
  python final_run.py --mode dryrun    # 검증 t0=2011/2013/2015 롤링 원점 재현 → dryrun_result.json (캐시: cache/, 출원연도 ≤2018)
  python final_run.py --mode holdout   # 🔒 3단계 평가자 전용. 캐시 cache_holdout/ (출원연도 ≤2021) 빌드 → 학습 t0 2005..2015
                                       #    (결과창 ≤2018 = 홀드아웃 기준시점 이전 완전 관측) → t0=2018 예측·평가 → holdout_result.json
  python final_run.py --mode selftest  # 코드 점검: 검증 캐시(≤2018)로 holdout·future 코드 경로를 가짜 원점(t0=2015, 학습 ≤2012)으로 실행
                                       #    → selftest_holdout.json, selftest_future.csv (홀드아웃 자료 미접근)
  python final_run.py --mode future    # 홀드아웃 평가 뒤에만. 캐시 cache_future/ (출원연도 ≤2022) → 학습 t0 2005..2019 → t0=2022 예측
                                       #    (2023–2025 KR 부상 예상 셀) → future_forecast.csv, future_forecast_ksic.csv
빌드 단계는 산출물이 이미 있으면 건너뛴다(9분 제한에 걸리면 같은 명령을 다시 실행하면 이어서 진행).
"""
import os, sys, json, argparse, subprocess, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['dryrun', 'holdout', 'future', 'selftest'])
ap.add_argument('--n_boot', type=int, default=1000)
args = ap.parse_args()
SPEC = json.load(open(os.path.join(HERE, 'final_spec.json'), encoding='utf-8'))
MODE_CFG = SPEC['modes']['dryrun' if args.mode == 'selftest' else args.mode]

# ---- 환경 설정은 t3lib import 전에 (t3lib 가 import 시점에 캐시·연도 상한을 읽음) ----
env = dict(T3_CACHE=MODE_CFG['cache'], T3_MAXYEAR=str(MODE_CFG['max_year']), T3_LABEL_MAX=str(MODE_CFG['label_max']),
           T3_T0MAX=str(MODE_CFG['t0_max_feat2']))
if args.mode in ('holdout', 'future'):
    env['T3_HOLDOUT_UNLOCK'] = '1'   # 🔒 해제는 이 두 모드에서만 (3단계 평가자가 실행)
else:
    os.environ.pop('T3_HOLDOUT_UNLOCK', None)
    assert MODE_CFG['max_year'] <= 2018 and MODE_CFG['label_max'] <= 2018
os.environ.update(env)
os.chdir(HERE)
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, MODE_CFG['cache'])
os.makedirs(CACHE, exist_ok=True)

def build(step, out):
    if os.path.exists(os.path.join(CACHE, out)):
        print(f'[build] {step}: {out} 있음 → 건너뜀', flush=True); return
    t = time.time(); print(f'[build] {step} 실행 …', flush=True)
    r = subprocess.run([sys.executable, step], cwd=HERE, env={**os.environ, **env})
    if r.returncode != 0: raise SystemExit(f'{step} 실패')
    print(f'[build] {step} 완료 {time.time()-t:.0f}s', flush=True)
build('build_base.py', 'base_rows.parquet')
build('build_panel.py', 'panel_ccy.parquet')
build('build_feat2.py', 'feat2.parquet')

import numpy as np, pandas as pd
import t3lib as L
assert L.Y1 == MODE_CFG['max_year'] and L.LABEL_MAX_YEAR == MODE_CFG['label_max']
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import nbinom, rankdata, spearmanr
import xgboost as xgb

M = SPEC['model']; LAB = SPEC['label']
LCOLS, XCOLS, NBCOLS = M['logit']['features'], M['xgb_clf']['features'], M['nb_struct']['features']

# ---------------- 레이블 (중립 동률깨기) ----------------
def hkey(cells):
    return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cells], dtype=np.int64)
def breakout_neutral(y, tie, key, top):
    k = max(1, int(round(top * len(y)))); o = np.lexsort((key, -tie, -y))
    yb = np.zeros(len(y), dtype=int); yb[o[:k]] = 1; return yb
def add_labels(df):
    d = df[df.tot_W >= LAB['min_total']].copy(); d['yb'] = 0
    for t, g in d.groupby('t0'):
        d.loc[g.index, 'yb'] = breakout_neutral(g.y.values, (g.kr_fut - g.kr_W).values, hkey(g.cell.values), LAB['top'])
    return d.reset_index(drop=True)

# ---------------- 파생 피처 (models.enrich 동결 사본) ----------------
def enrich(D):
    P = L.load(); kr = P['A']['n'][:, 0, :]; tot = P['A']['n'].sum(1)
    f2 = pd.read_parquet(os.path.join(L.CACHE, 'feat2.parquet')).drop(columns=['sub'])
    n0 = len(D); D = D.merge(f2, on=['ksic', 'mg', 't0'], how='left'); assert len(D) == n0
    idx = np.searchsorted(P['cells'], D.cell.values)
    long = np.full(len(D), np.nan); dd = np.full(len(D), np.nan); tl = np.full(len(D), np.nan)
    for t0, g in D.groupby('t0'):
        a = max(L.Y0, t0 - 8); ii = idx[g.index]
        long[g.index] = np.log1p(kr[ii, a - L.Y0:t0 - 3 - L.Y0 + 1].mean(1))
        dd[g.index] = np.log1p(kr[ii, a - L.Y0:t0 - L.Y0 + 1].max(1)) - np.log1p(kr[ii, t0 - 2 - L.Y0:t0 - L.Y0 + 1].mean(1))
        tl[g.index] = np.log1p(tot[ii, a - L.Y0:t0 - 3 - L.Y0 + 1].mean(1))
    D['kr_long'] = long; D['kr_drawdown'] = dd; D['kr_vs_long'] = D.kr_lvl - np.log(3) - D.kr_long
    D['tot_vs_long'] = D.tot_lvl - np.log(3) - tl
    D['kr_share_rel_ksic'] = D.kr_share - D.groupby(['t0', 'ksic']).kr_share.transform('median')
    D['kr_share_rel_sub'] = D.kr_share - D.sub_krsh
    D['kr_share_rel_mg'] = D.kr_share - D.groupby(['t0', 'mg']).kr_share.transform('median')
    return D

def get_data(t0s, need_label=True):
    df = pd.concat([L.get_feats(t) for t in t0s], ignore_index=True)
    if need_label: d = add_labels(df)
    else: d = df[df.tot_W >= LAB['min_total']].reset_index(drop=True)
    return enrich(d).reset_index(drop=True)

# ---------------- 모형 ----------------
def fit_logit(tr, te):
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=M['logit']['C'], max_iter=3000))
    m.fit(tr[LCOLS].values, tr.yb.values); return m.predict_proba(te[LCOLS].values)[:, 1]
def fit_xgbc(tr, te):
    p = dict(M['xgb_clf']['params']); m = xgb.train(p, xgb.DMatrix(tr[XCOLS].values, label=tr.yb.values), M['xgb_clf']['rounds'])
    return m.predict(xgb.DMatrix(te[XCOLS].values))
def fit_alpha(F, mu):
    best = None
    for a in np.exp(np.linspace(np.log(0.01), np.log(5), 60)):
        r = 1 / a; ll = nbinom.logpmf(F, r, r / (r + mu)).sum()
        if best is None or ll > best[0]: best = (ll, a)
    return best[1]
def fit_nb(tr, te, info):
    p = dict(M['nb_struct']['params']); d = xgb.DMatrix(tr[NBCOLS].values, label=tr.kr_fut.values)
    m = xgb.train(p, d, M['nb_struct']['rounds'])
    a = fit_alpha(tr.kr_fut.values, m.predict(d)); mu = m.predict(xgb.DMatrix(te[NBCOLS].values))
    tau = float(np.median([np.quantile(g.y.values, 1 - LAB['top']) for _, g in tr.groupby('t0')]))
    need = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9); r = 1 / a
    info.update(alpha=round(float(a), 4), tau=round(tau, 4))
    return nbinom.sf(need - 1, r, r / (r + mu))
def model_fp(tr, te, info=None):
    info = {} if info is None else info
    ps = [fit_logit(tr, te), fit_xgbc(tr, te), fit_nb(tr, te, info)]
    return np.mean([rankdata(p) / len(p) for p in ps], axis=0)
def baseline_fp(tr, te):
    """고정 기준선: 과거 3년 KR 성장률(kr_mom) 단독 로지스틱"""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values)
    return m.predict_proba(te[['kr_mom']].values)[:, 1]

def metrics(y, yb, p):
    k = max(1, int(round(0.10 * len(y)))); top = np.argsort(-p)[:k]; pr = average_precision_score(yb, p)
    return dict(auc=float(roc_auc_score(yb, p)), prauc=float(pr), prauc_x_base=float(pr / yb.mean()), p_at10=float(yb[top].mean()),
                rho=float(spearmanr(y, p).statistic), n=int(len(y)), n_evt=int(yb.sum()), base_rate=float(yb.mean()))

def cluster_boot(blocks, n_boot, seed=0):
    """blocks: [(mg, yb, p_model, p_base)] 원점별. 군집 = IPC 메인그룹(같은 메인그룹의 다른 KSIC 셀·다른 원점을 함께 재표집)."""
    rng = np.random.default_rng(seed)
    ums = np.unique(np.concatenate([b[0] for b in blocks]))
    gi = [pd.Series(np.arange(len(b[0]))).groupby(b[0]).apply(lambda s: s.values).to_dict() for b in blocks]
    out = []
    for _ in range(n_boot):
        s = rng.choice(ums, len(ums)); am, ab = [], []
        for b, g in zip(blocks, gi):
            ii = np.concatenate([g[u] for u in s if u in g]); y = b[1][ii]
            if y.min() == y.max(): continue
            am.append(roc_auc_score(y, b[2][ii])); ab.append(roc_auc_score(y, b[3][ii]))
        a, a0 = np.mean(am), np.mean(ab); out.append((a, a0, (a - a0) / (1 - a0), a - a0))
    o = np.array(out); q = lambda j: [float(v) for v in np.percentile(o[:, j], [2.5, 97.5])]
    return dict(auc=q(0), auc_baseline=q(1), skill=q(2), auc_diff=q(3), n_boot=n_boot, cluster='IPC 메인그룹', n_clusters=int(len(ums)))

def run_eval(D, test_t0s, out_json, pred_file):
    res, blocks, preds = [], [], []
    for v in test_t0s:
        tr = D[D.t0 <= v - 3]; te = D[D.t0 == v]
        assert len(tr) and tr.t0.max() + 3 <= v, '학습 레이블은 기준시점 이전 완전 관측분만'
        info = {}; p = model_fp(tr, te, info); p0 = baseline_fp(tr, te)
        m = metrics(te.y.values, te.yb.values, p); m0 = metrics(te.y.values, te.yb.values, p0)
        res.append(dict(t0=v, train_t0=[int(tr.t0.min()), int(tr.t0.max())], n_train=int(len(tr)), nb=info, model=m, baseline=m0,
                        skill=(m['auc'] - m0['auc']) / (1 - m0['auc'])))
        print(f"t0={v}: 모형 AUC {m['auc']:.4f} PR-AUC {m['prauc']:.3f} P@10 {m['p_at10']:.3f} rho {m['rho']:.3f} | 기준선 AUC {m0['auc']:.4f} | skill {res[-1]['skill']:.4f} | NB {info}", flush=True)
        blocks.append((te.mg.values, te.yb.values, p, p0))
        preds.append(pd.DataFrame(dict(cell=te.cell.values, ksic=te.ksic.values, mg=te.mg.values, t0=v, p=p, p_base=p0,
                                       kr_W=te.kr_W.values, kr_fut=te.kr_fut.values, y=te.y.values, yb=te.yb.values)))
    auc = float(np.mean([r['model']['auc'] for r in res])); auc0 = float(np.mean([r['baseline']['auc'] for r in res]))
    summ = dict(auc=auc, auc_baseline=auc0, skill=(auc - auc0) / (1 - auc0),
                prauc=float(np.mean([r['model']['prauc'] for r in res])), prauc_baseline=float(np.mean([r['baseline']['prauc'] for r in res])),
                prauc_x_base=float(np.mean([r['model']['prauc_x_base'] for r in res])),
                prauc_x_base_baseline=float(np.mean([r['baseline']['prauc_x_base'] for r in res])),
                p_at10=float(np.mean([r['model']['p_at10'] for r in res])), p_at10_baseline=float(np.mean([r['baseline']['p_at10'] for r in res])),
                rho=float(np.mean([r['model']['rho'] for r in res])), rho_baseline=float(np.mean([r['baseline']['rho'] for r in res])))
    ci = cluster_boot(blocks, args.n_boot)
    out = dict(mode=args.mode, model=SPEC['model_name'], test_t0=test_t0s, summary=summ, ci95=ci, by_t0=res,
               label=LAB, created=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(out, open(os.path.join(HERE, out_json), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    pd.concat(preds, ignore_index=True).to_parquet(os.path.join(HERE, 'preds', pred_file))
    print(json.dumps(dict(summary=summ, ci95=ci), ensure_ascii=False, indent=1))
    print('저장:', out_json, 'preds/' + pred_file)

def run_future(FT, LAST, OUT):
    ft, last_train = FT, LAST; assert last_train + 3 <= MODE_CFG['label_max']
    tr = get_data(range(SPEC['train']['first_t0'], last_train + 1))
    te = get_data([ft], need_label=False)
    info = {}; p = model_fp(tr, te, info); te['score'] = p
    te['rank_pct'] = te.score.rank(pct=True, ascending=False); te['pred_breakout_top10'] = (te.rank_pct <= 0.10).astype(int)
    te['rank_in_ksic'] = te.groupby('ksic').score.rank(ascending=False, method='first').astype(int)
    cols = ['ksic', 'mg', 'kr_W', 'tot_W', 'kr_mom', 'US_mom', 'CN_mom', 'JP_mom', 'EP_mom', 'kr_share_rel_ksic', 'kr_drawdown', 'nb_kmom',
            'score', 'rank_pct', 'pred_breakout_top10', 'rank_in_ksic']
    te = te.sort_values('score', ascending=False)
    te[cols].to_csv(os.path.join(HERE, OUT+'.csv'), index=False, encoding='utf-8-sig')
    te.groupby('ksic').agg(n_cells=('score', 'size'), n_pred_breakout=('pred_breakout_top10', 'sum'), kr_W_sum=('kr_W', 'sum')) \
      .assign(pred_rate=lambda d: d.n_pred_breakout / d.n_cells).sort_values('pred_rate', ascending=False) \
      .to_csv(os.path.join(HERE, OUT+'_ksic.csv'), encoding='utf-8-sig')
    te[te.kr_W >= 5][cols].to_csv(os.path.join(HERE, OUT+'_krge5.csv'), index=False, encoding='utf-8-sig')  # 보고서용: KR 과거 3년 ≥5건(실질 부상) 셀만
    json.dump(dict(mode='future', t0=ft, forecast_window=[ft + 1, ft + 3], train_t0=[SPEC['train']['first_t0'], last_train],
                   n_train=int(len(tr)), n_cells=int(len(te)), nb=info, created=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(os.path.join(HERE, OUT+'_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('저장:', OUT+'.csv', OUT+'_ksic.csv', OUT+'_meta.json', info)

t_start = time.time()
os.makedirs(os.path.join(HERE, 'preds'), exist_ok=True)
if args.mode == 'dryrun':
    D = get_data(range(SPEC['train']['first_t0'], 2016))
    run_eval(D, SPEC['validation']['t0'], 'dryrun_result.json', 'final_dryrun.parquet')
elif args.mode in ('holdout', 'selftest'):
    h = SPEC['holdout']['t0'] if args.mode == 'holdout' else 2015
    assert (args.mode == 'holdout' and h == 2018) or (args.mode == 'selftest' and h + 3 <= 2018)
    D = get_data(list(range(SPEC['train']['first_t0'], h - 3 + 1)) + [h])   # 학습 t0 2005..h-3 + 평가 t0 h
    assert D[D.t0 < h].t0.max() + 3 <= h
    if args.mode == 'holdout': run_eval(D, [h], 'holdout_result.json', 'HOLDOUT_t0_2018.parquet')
    else: run_eval(D, [h], 'selftest_holdout.json', 'selftest_holdout.parquet')
if args.mode == 'future': run_future(SPEC['future']['t0'], SPEC['future']['last_train_t0'], 'future_forecast')
if args.mode == 'selftest': run_future(2015, 2012, 'selftest_future')
print(f'총 {time.time()-t_start:.0f}s')
