# r3kip_ 원점 시점 피인용 수 재구성 + KSIC 지표 + 개발기간 선행검정(T4 O16, T3 t0=2016)
import os, json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(os.path.dirname(D))
S = pd.read_parquet(os.path.join(D, 'r3kip_sample.parquet'))
ci = pd.read_parquet(os.path.join(D, 'r3kip_citing_edges.parquet'))
cn = pd.read_parquet(os.path.join(D, 'r3kip_citation_edges.parquet'))

# (피인용, 인용) 쌍 단위로 접기: 심사측(E0801 발송문서/E0802 선행기술조사보고서/E0805 선행기술조사문헌), 출원인(E0806)
EX = ['E0801', 'E0802', 'E0805']
g = ci.groupby(['cited_appl', 'citing_appl']).agg(
    exam=('ctype', lambda s: s.isin(EX).any()), appl=('ctype', lambda s: (s == 'E0806').any()),
    vis_cons=('vis_cons', 'first'), vis_lib=('vis_lib', 'first'), src=('src', 'first'),
    self_ksic=('self_ksic', 'first')).reset_index()
g = g[g.cited_appl != g.citing_appl]
g.to_parquet(os.path.join(D, 'r3kip_citing_pairs.parquet'), index=False)

# 원점별 as-of 피인용 수 (매년 1월 1일 기준)
asof = [f'{y}-01-01' for y in range(2008, 2020)]
rows = []
for a in S.applNo:
    e = g[g.cited_appl == a]
    for d in asof:
        dd = pd.Timestamp(d)
        vc = e.vis_cons < dd
        rows.append(dict(applNo=a, asof=d, fwd_cons=int(vc.sum()), fwd_lib=int((e.vis_lib < dd).sum()),
                         fwd_cons_exam=int((vc & e.exam).sum()), fwd_cons_appl=int((vc & e.appl).sum())))
F = pd.DataFrame(rows)
tot = g.groupby('cited_appl').size().rename('fwd_total_snapshot')
F = F.merge(S[['applNo', 'set', 'ksic']], on='applNo').merge(tot, left_on='applNo', right_index=True, how='left')
F['fwd_total_snapshot'] = F.fwd_total_snapshot.fillna(0).astype(int)
F.to_parquet(os.path.join(D, 'r3kip_patent_fwd_asof.parquet'), index=False)

# 후방 인용 지표(150건)
FOR = ['JP', 'US', 'EP', 'WO', 'DE', 'CN', 'GB', 'FR']
cn['kind'] = np.select([cn.cited_cc == 'KR', cn.cited_cc.isin(FOR), cn.cited_cc == ''], ['KR', 'foreign', 'npl'], 'other')
bw = cn.drop_duplicates(['citing_appl', 'cited_std_no']).groupby('citing_appl').agg(
    n_back=('cited_std_no', 'size'), sh_foreign=('kind', lambda s: (s == 'foreign').mean()),
    sh_npl=('kind', lambda s: (s == 'npl').mean()), sh_kr=('kind', lambda s: (s == 'KR').mean()),
    n_exam=('ctype', lambda s: s.isin(EX).sum()), n_applcite=('ctype', lambda s: (s == 'E0806').sum()),
    med_prior_pub=('cited_pub_dt', 'median')).reset_index().rename(columns={'citing_appl': 'applNo'})
qb = S[S.bw][['applNo', 'ksic']].merge(bw, on='applNo', how='left')
qb['n_back'] = qb.n_back.fillna(0)
qb.to_parquet(os.path.join(D, 'r3kip_patent_backward.parquet'), index=False)

# KSIC 지표(해석층): T4 사례-대조 표본을 가중(모집단 크기/표본 크기)해 O16 위험집합 대표값으로
t4 = S[S.set == 'T4'].merge(F[F['asof'] == '2016-01-01'][['applNo', 'fwd_cons', 'fwd_lib', 'fwd_total_snapshot']], on='applNo')
K = t4.groupby('ksic').apply(lambda x: pd.Series(dict(
    n=len(x), w_fwd_cons_2016=np.average(x.fwd_cons, weights=x.w),
    w_share_cited_2016=np.average(x.fwd_cons > 0, weights=x.w),
    w_fwd_snapshot=np.average(x.fwd_total_snapshot, weights=x.w))), include_groups=False).reset_index()
K = K.merge(qb.groupby('ksic').agg(bw_n=('applNo', 'size'), bw_mean_refs=('n_back', 'mean'),
                                   bw_sh_foreign=('sh_foreign', 'mean'), bw_sh_npl=('sh_npl', 'mean')).reset_index(),
            on='ksic', how='left')
K.to_parquet(os.path.join(D, 'r3kip_ksic_indicators.parquet'), index=False)

out = {}
# ---------------- 선행검정 1: T4 O16 (L=2016-01-01, 3년 내 등록료 불납 포기, 결과 ≤2018-12)
fe = pd.read_parquet(os.path.join(B, 'T4_특허포기예측', 'R2', 'cache', 'feat_20160101.parquet'),
                     columns=['pid', 'age', 'claims', 'paid_ahead', 'yrs_since_pay', 'n_renew', 'own_corp', 'own_foreign',
                              'port_alive', 'a_lapse_rate', 'n_transfer', 'n_ipc'])
t = t4.merge(fe.drop(columns=['age', 'claims']), on='pid', how='left')
t['lf'] = np.log1p(t.fwd_cons); t['lfl'] = np.log1p(t.fwd_lib); t['anyf'] = (t.fwd_cons > 0).astype(float)
t['lsnap'] = np.log1p(t.fwd_total_snapshot)
cov = ['age', 'claims', 'paid_ahead', 'yrs_since_pay', 'n_renew', 'own_corp', 'own_foreign', 'port_alive', 'a_lapse_rate', 'n_transfer', 'n_ipc']
X = t[cov].astype(float)
X = X.fillna(X.median())
X = (X - X.mean()) / X.std().replace(0, 1)


def clogit(extra):
    Z = X.copy()
    for c in extra:
        Z[c] = t[c].values
    return ConditionalLogit(t.y.values, Z, groups=t.ksic.values).fit(disp=0, method='bfgs', maxiter=1000)


r0 = clogit([])
res = {}
for v in ['lf', 'lfl', 'anyf', 'lsnap']:
    r1 = clogit([v]); lr = 2 * (r1.llf - r0.llf)
    res[v] = dict(coef=round(float(r1.params[v]), 4), se=round(float(r1.bse[v]), 4), p=round(float(r1.pvalues[v]), 4),
                  LR=round(float(lr), 3), p_LR=round(float(stats.chi2.sf(lr, 1)), 4))
res['mean_fwd_cons_lapse'] = float(t[t.y == 1].fwd_cons.mean()); res['mean_fwd_cons_keep'] = float(t[t.y == 0].fwd_cons.mean())
res['share_cited_lapse'] = float((t[t.y == 1].fwd_cons > 0).mean()); res['share_cited_keep'] = float((t[t.y == 0].fwd_cons > 0).mean())
res['mw_p'] = float(stats.mannwhitneyu(t[t.y == 1].fwd_cons, t[t.y == 0].fwd_cons).pvalue)
d0, d1 = [], []
X1 = X.assign(lf=t.lf.values)
for seed in range(20):
    p0 = np.zeros(len(t)); p1 = np.zeros(len(t))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, t.y):
        p0[te] = LogisticRegression(max_iter=2000).fit(X.iloc[tr], t.y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        p1[te] = LogisticRegression(max_iter=2000).fit(X1.iloc[tr], t.y.iloc[tr]).predict_proba(X1.iloc[te])[:, 1]
    d0.append(roc_auc_score(t.y, p0)); d1.append(roc_auc_score(t.y, p1))
d0 = np.array(d0); d1 = np.array(d1)
res.update(cv_auc_base=round(float(d0.mean()), 4), cv_auc_plus_fwd=round(float(d1.mean()), 4),
           cv_auc_diff=round(float((d1 - d0).mean()), 4), cv_diff_pos_share=float((d1 > d0).mean()),
           n=int(len(t)), n_lapse=int(t.y.sum()))
out['T4_O16'] = res

# ---------------- 선행검정 2: T3 t0=2016 (적격셀 150개, 셀당 KR 등록특허 1건, as-of 2017-01-01 = t0년 말)
dp = pd.read_parquet(os.path.join(B, 'T3_기술셀부상예측', 'R2', 'preds', 'dev_preds.parquet'))
dp = dp[dp.t0 == 2016][['cell', 'p', 'p_base', 'y', 'yb']]
t3 = S[S.set == 'T3'][['applNo', 'cell']].merge(F[F['asof'] == '2017-01-01'][['applNo', 'fwd_cons', 'fwd_lib']], on='applNo').merge(dp, on='cell')
t3['lf'] = np.log1p(t3.fwd_cons)
r3 = {'n_cells': int(len(t3)), 'n_emerge': int(t3.yb.sum()), 'share_cited': float((t3.fwd_cons > 0).mean())}
sp = stats.spearmanr(t3.fwd_cons, t3.y); r3['spearman_fwd_y'] = [round(float(sp[0]), 4), round(float(sp[1]), 4)]
res_y = t3.y - np.poly1d(np.polyfit(t3.p, t3.y, 1))(t3.p)
sp = stats.spearmanr(t3.fwd_cons, res_y); r3['spearman_fwd_resid_given_champion_p'] = [round(float(sp[0]), 4), round(float(sp[1]), 4)]
ols = sm.OLS(t3.y, sm.add_constant(t3[['p', 'lf']])).fit()
r3['ols_lf_coef'] = round(float(ols.params['lf']), 4); r3['ols_lf_t'] = round(float(ols.tvalues['lf']), 3)
r3['auc_fwd_for_yb'] = round(float(roc_auc_score(t3.yb, t3.fwd_cons)), 4) if t3.yb.nunique() > 1 else None
out['T3_t0_2016'] = r3
json.dump(out, open(os.path.join(D, 'r3kip_pretest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print('T4 표본 as-of 2016 피인용>0 비율', round(float((t.fwd_cons > 0).mean()), 4), '스냅샷 총 피인용 평균', round(float(t.fwd_total_snapshot.mean()), 3),
      'as-of 2016 평균', round(float(t.fwd_cons.mean()), 3))
print(K.describe().round(3).T)
