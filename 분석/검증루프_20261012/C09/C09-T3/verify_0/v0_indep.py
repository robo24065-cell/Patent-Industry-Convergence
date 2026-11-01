"""verify_0: S1·S4 독립 재현 (최종 원점 t0=2021).
- S1 피처: base_rows 의 US 행을 pandas 로 직접 읽어(duckdb·s1_panel·s2_feats 미사용) mgn·dedupe·첫 등장을 다시 계산.
  비US 국가의 신규 출원인 수는 공식 panel_ccy(트랙 캐시) 값을 그대로 쓴다(S1 은 US 만 바꾸므로).
- 학습·채점: c09core/final_run 함수 없이 xgboost·sklearn 직접 호출, 자체 레이블 경계·확대 표본·NB 꼬리확률·순위평균·AUC.
입력: v0_D_final.parquet, v0_W_final.parquet(v0_export.py), 트랙 R3/cache_pub_final/base_rows.parquet·panel_ccy.parquet (읽기 전용)
"""
import os, sys, json, re, time, numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, nbinom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
MY = os.path.dirname(os.path.abspath(__file__)); WD = os.path.dirname(MY)
R3 = os.path.abspath(os.path.join(WD, '..', '..', '..', '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3'))
CF = os.path.join(R3, 'cache_pub_final')
t = time.time(); out = {}
# ---- 1) US S0/S1 신규 출원인 재계산 (pandas) ----
us = pd.read_parquet(os.path.join(CF, 'base_rows.parquet'), columns=['ksic', 'cc', 'year', 'appl_year', 'appl_no', 'ipc_raw', 'app1'], filters=[('cc', '==', 'US')])
u = pd.Series(us.ipc_raw.unique())
def mgn(s):
    if s is None: return None
    s = s.replace(' ', '')
    m = re.match(r'^([A-H][0-9]{2}[A-Z])([0-9]{1,4})/', s)
    if m: return m.group(1) + ' ' + str(int(m.group(2)))
    if re.match(r'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$', s): return s[:4] + ' ' + str(int(s[4:len(s) - 2]))
    return None
mp = dict(zip(u, [mgn(x) for x in u])); us['mg'] = us.ipc_raw.map(mp); us = us[us.mg.notna()]
us = us.groupby(['ksic', 'mg', 'year', 'appl_no'], sort=False).agg(app1=('app1', 'first'), appl_year=('appl_year', 'min')).reset_index()
us['miss'] = us.app1.isna() | (us.app1 == ''); us['old'] = us.appl_year <= 2011
nm = us[~us.miss].copy()
nm['fy0'] = nm.groupby(['ksic', 'mg', 'app1']).year.transform('min')
nm1 = nm[~nm.old].copy(); nm1['fy1'] = nm1.groupby(['ksic', 'mg', 'app1']).year.transform('min')
g0 = nm[nm.fy0 == nm.year].groupby(['ksic', 'mg', 'year']).agg(new_apps0=('appl_no', 'size'), new0=('app1', 'nunique'))
g1 = nm1[nm1.fy1 == nm1.year].groupby(['ksic', 'mg', 'year']).agg(new_apps1=('appl_no', 'size'), new1=('app1', 'nunique'))
go = us[us.old].groupby(['ksic', 'mg', 'year']).size().rename('n_old')
gn = us.groupby(['ksic', 'mg', 'year']).size().rename('n_us')
G = pd.concat([gn, go, g0, g1], axis=1).fillna(0).reset_index()
off = pd.read_parquet(os.path.join(CF, 'panel_ccy.parquet'))
ou = off[off.cc == 'US'].merge(G, on=['ksic', 'mg', 'year'], how='outer')
out['us_parity_vs_official_panel'] = dict(rows=int(len(ou)), n_maxdiff=float((ou.n.fillna(0) - ou.n_us.fillna(0)).abs().max()),
    newapp_apps_maxdiff=float((ou.n_newapp_apps.fillna(0) - ou.new_apps0.fillna(0)).abs().max()), newapp_maxdiff=float((ou.n_newapp.fillna(0) - ou.new0.fillna(0)).abs().max()))
print(out, round(time.time() - t), flush=True)
# 셀×연도 합계(5개국) + US 교체
agg = off.groupby(['ksic', 'mg', 'year'])[['n', 'n_newapp_apps', 'n_newapp']].sum().reset_index().merge(G, on=['ksic', 'mg', 'year'], how='left').fillna(0)
agg['num1'] = agg.n_newapp_apps - agg.new_apps0 + agg.new_apps1; agg['nn1'] = agg.n_newapp - agg.new0 + agg.new1; agg['den1'] = agg.n - agg.n_old
agg['cell'] = agg.ksic + '|' + agg.mg
def win(t0):
    a = agg[agg.year.between(t0 - 2, t0)].groupby('cell')[['n', 'n_newapp_apps', 'n_newapp', 'num1', 'nn1', 'den1']].sum()
    return pd.DataFrame(dict(t0=t0, sh0=a.n_newapp_apps / np.maximum(a.n, 1), n0=np.log1p(a.n_newapp), sh1=a.num1 / np.maximum(a.den1, 1), n1=np.log1p(a.nn1))).reset_index()
S1F = pd.concat([win(t0) for t0 in list(range(2005, 2019)) + [2021]], ignore_index=True)
D = pd.read_parquet(os.path.join(MY, 'v0_D_final.parquet')); Wd = pd.read_parquet(os.path.join(MY, 'v0_W_final.parquet'))
mine = pd.read_parquet(os.path.join(WD, 'cache', 'scen_feats_fin.parquet'))
chk = {}
for nm_, X in [('D', D), ('W', Wd)]:
    m = X[['cell', 't0', 'all_newapp_sh', 'all_newapp_n']].merge(S1F, on=['cell', 't0'], how='left').merge(mine[['cell', 't0', 'all_newapp_sh_S1', 'all_newapp_n_S1']], on=['cell', 't0'], how='left')
    chk[nm_] = dict(rows=int(len(m)), s0_sh_vs_official=float(np.abs(m.sh0 - m.all_newapp_sh).max()), s0_n_vs_official=float(np.abs(m.n0 - m.all_newapp_n).max()),
                   s1_sh_indep_vs_main=float(np.abs(m.sh1 - m.all_newapp_sh_S1).max()), s1_n_indep_vs_main=float(np.abs(m.n1 - m.all_newapp_n_S1).max()),
                   s1_rows_changed=float((np.abs(m.sh1 - m.sh0) > 1e-9).mean()))
out['s1_feature_check'] = chk; print(chk, round(time.time() - t), flush=True)
# ---- 2) 독립 학습·채점 ----
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8')); comps = SPEC['cfg']['components']
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def run(D, W, drop=()):
    tr = D[(D.t0 >= 2005) & (D.t0 <= 2018)]; te = D[D.t0 == 2021]; wtr = W[(W.t0 >= 2005) & (W.t0 <= 2018)]
    thr = D[D.yb == 1].groupby('t0').y.min()
    ranks = []
    for c in comps:
        fs = [f for f in c['features'] if f not in drop]
        if c['type'] == 'logit':
            m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=c['C'], max_iter=3000)).fit(tr[fs].values, tr.yb.values)
            p = m.predict_proba(te[fs].values)[:, 1]
        else:
            w = wtr.copy(); prm = dict(c['params']); prm['seed'] = 0
            if c['type'] == 'xgbc':
                w['yb'] = (w.y >= w.t0.map(thr)).astype(int)
                X = np.vstack([tr[fs].values, w[fs].values]); yv = np.r_[tr.yb.values, w.yb.values]
                bst = xgb.train(prm, xgb.DMatrix(X, label=yv), c['rounds']); p = bst.predict(xgb.DMatrix(te[fs].values))
            else:
                X = np.vstack([tr[fs].values, w[fs].values]); yv = np.r_[tr.kr_fut.values, w.kr_fut.values]
                dm = xgb.DMatrix(X, label=yv); bst = xgb.train(prm, dm, c['rounds']); mu_tr = bst.predict(dm)
                ll = [(nbinom.logpmf(yv, 1 / a, (1 / a) / (1 / a + mu_tr)).sum(), a) for a in np.exp(np.linspace(np.log(0.01), np.log(5), 60))]
                a = max(ll)[1]; r = 1 / a
                tau = float(np.median([np.quantile(g.y.values, 0.9) for _, g in tr.groupby('t0')]))
                mu = bst.predict(xgb.DMatrix(te[fs].values)); need = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9)
                p = nbinom.sf(need - 1, r, r / (r + mu))
        ranks.append(rankdata(p) / len(te))
    p = np.mean(ranks, 0)
    b = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values).predict_proba(te[['kr_mom']].values)[:, 1]
    a, a0 = auc(te.yb.values, p), auc(te.yb.values, b)
    return dict(auc=float(a), auc_base=float(a0), skill=float((a - a0) / (1 - a0))), pd.DataFrame(dict(cell=te.cell.values, p=p))
AF = ['all_newapp_sh', 'all_newapp_n', 'kr_newapp_sh', 'kr_newk_sh', 'kr_napp', 'kr_newapp_n', 'kr_hhi', 'app_mom', 'app_size', 'big_sh', 'top1_sh']
res = {}
r0, p0 = run(D, Wd); res['S0'] = r0; print('S0', r0, round(time.time() - t), flush=True)
D1 = D.merge(S1F[['cell', 't0', 'sh1', 'n1']], on=['cell', 't0'], how='left'); W1 = Wd.merge(S1F[['cell', 't0', 'sh1', 'n1']], on=['cell', 't0'], how='left')
for X in (D1, W1): X['all_newapp_sh'] = X.sh1.astype(np.float32); X['all_newapp_n'] = X.n1.astype(np.float32)
r1, p1 = run(D1, W1); res['S1'] = r1; print('S1', r1, round(time.time() - t), flush=True)
r4, p4 = run(D, Wd, drop=AF); res['S4'] = r4; print('S4', r4, round(time.time() - t), flush=True)
main = {s: json.load(open(os.path.join(WD, 'cache', f'res_{s}_final.json'), encoding='utf-8'))['by_origin']['2021']['skill'] for s in ['S0', 'S1', 'S4']}
out['indep_final'] = res; out['main_final_skill'] = main
out['diff_indep_minus_main'] = {s: res[s]['skill'] - main[s] for s in main}
out['delta_indep'] = {s: res[s]['skill'] - res['S0']['skill'] for s in ['S1', 'S4']}
out['delta_main'] = {s: main[s] - main['S0'] for s in ['S1', 'S4']}
out['pass'] = bool(all(abs(v) < 0.002 for v in out['diff_indep_minus_main'].values()) and chk['D']['s1_sh_indep_vs_main'] < 1e-6 and chk['W']['s1_sh_indep_vs_main'] < 1e-6)
json.dump(out, open(os.path.join(MY, 'v0_indep_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)); print('총', round(time.time() - t))
