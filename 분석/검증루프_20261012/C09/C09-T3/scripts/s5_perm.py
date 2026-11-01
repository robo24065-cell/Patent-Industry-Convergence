"""s5: 출원인 피처 기여 (공식 모형 S0)
 --mode dev  : 개발 원점 2010–2018 permutation importance (그룹 AF·AF5·AFK·참고 그룹, 5회 반복, 앙상블 순위평균 AUC 하락). 최종 원점은 쓰지 않는다.
 --mode final: 최종 학습 표본(t0 2005–2018) 모형의 XGB 분류·음이항 gain 비중(피처 단위·그룹 단위). 최종 원점 레이블은 쓰지 않는다.
성분 학습은 c09core.comp_pred 와 같은 표본 구성(train_rows + 확대 표본)을 그대로 따른다. 비순열 예측이 S0 저장 예측과 같은지 대조.
"""
import os, sys, json, argparse, time, copy
ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True, choices=['dev', 'final']); ap.add_argument('--origins', default=None)
args = ap.parse_args()
MY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, MY)
from common import R3, CACHE
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))
if args.mode == 'dev':
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub', T3R3_MAXYEAR='2021', T3R3_LABEL_MAX='2021', T3R3_EXT_SUFFIX='', T3R3_EXTB_SUFFIX='')
    os.environ.pop('T3R3_FINAL_UNLOCK', None); os.environ.pop('T3R3_TRAIN_MAX_T0', None)
else:
    os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
                      T3R3_TRAIN_MAX_T0='2018', T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, nbinom
import c09lib as L, c09core as C
SUF = os.environ['T3R3_EXT_SUFFIX']; CFG = SPEC['cfg']
def prep(D):
    cf = pd.read_parquet(os.path.join(R3, f'extB_cell_pub{SUF}.parquet')); kf = pd.read_parquet(os.path.join(R3, f'extB_ksic_pub{SUF}.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    D['oa_mg_rel'] = D.oa_mg_g - D.oa_all_g if 'oa_all_g' in D else np.nan
    D['ix_sci_fx'] = D.ros_sci_rel * D.fx_mom; D['ix_sci_kr'] = D.ros_sci_rel * D.kr_mom; D['ix_oa_gap'] = D.oa_mg_g - D.kr_mom
    return D
_gw = C.get_wide
def gw(Dx, mt):
    w = _gw(Dx, mt)
    if 'ros_sci' not in w: w = prep(w)
    C._WIDE[mt] = w; return w
C.get_wide = gw
AF5 = ['all_newapp_sh', 'all_newapp_n']
AFK = ['kr_newapp_sh', 'kr_newk_sh', 'kr_napp', 'kr_newapp_n', 'kr_hhi', 'app_mom', 'app_size', 'big_sh', 'top1_sh']
GROUPS = {'AF(11)': AF5 + AFK, 'AF5(2)': AF5, 'AFK(9)': AFK, 'all_newapp_sh': ['all_newapp_sh'],
          '참고_우선권·IPC쌍(5)': ['kr_prio_for_sh', 'kr_forin_mom', 'kr_dom_mom', 'kr_newpair_sh', 'all_newpair_sh'],
          '참고_US셀이력(6)': ['US_mom', 'US_mom_l1', 'US_mom_l2', 'US_lvl', 'US_g1', 'US_slope5'], '참고_KOSIS(3)': ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']}
def train_set(c, D, v):
    tr = C.train_rows(D, v, CFG)
    if c.get('train_min_total'):
        W = C.get_wide(D, c['train_min_total']); W = C.train_rows(W, v, CFG)
        if c['type'] != 'nb':
            thr = D[D.yb == 1].groupby('t0').y.min(); W = W.copy(); W['yb'] = (W.y >= W.t0.map(thr)).astype(int)
        tr = pd.concat([tr, W[tr.columns.intersection(W.columns)]], ignore_index=True)
        tr['_elig'] = np.r_[np.ones(len(tr) - len(W)), np.zeros(len(W))]
    return tr
def fit(c, tr):
    """성분별 학습 → 예측 함수. c09core.comp_* 와 같은 연산(단일 seed=0)."""
    if c['type'] == 'logit':
        from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=c.get('C', 1.0), max_iter=3000))
        m.fit(tr[c['features']].values, tr.yb.values); return (lambda te: m.predict_proba(te[c['features']].values)[:, 1]), None
    p = dict(c['params']); p['seed'] = 0
    if c['type'] == 'xgbc':
        m = xgb.train(p, xgb.DMatrix(tr[c['features']].values, label=tr.yb.values), c['rounds'])
        return (lambda te: m.predict(xgb.DMatrix(te[c['features']].values))), m
    d = xgb.DMatrix(tr[c['features']].values, label=tr.kr_fut.values); m = xgb.train(p, d, c['rounds'])
    a = C.fit_alpha(tr.kr_fut.values, m.predict(d)); r = 1 / a
    trq = tr[tr['_elig'] == 1] if '_elig' in tr else tr
    tau = float(np.median([np.quantile(g.y.values, 1 - C.TOP) for _, g in trq.groupby('t0')]))
    def pr(te):
        mu = m.predict(xgb.DMatrix(te[c['features']].values)); need = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9)
        return nbinom.sf(need - 1, r, r / (r + mu))
    return pr, m
def auc(y, p):
    rr = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (rr[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
t = time.time(); out = {}
if args.mode == 'dev':
    D = prep(C.get_D()); origins = SPEC['dev']['origins'] if not args.origins else [int(x) for x in args.origins.split(',')]
    S0 = pd.concat([pd.read_parquet(f) for f in [os.path.join(CACHE, 'preds_S0_dev_2010-2011-2012-2013-2014.parquet'), os.path.join(CACHE, 'preds_S0_dev_2015-2016-2017-2018.parquet')]])
    rng = np.random.default_rng(42); rows = []; chk = {}
    for v in origins:
        te = D[D.t0 == v].reset_index(drop=True); fns = [fit(c, train_set(c, D, v))[0] for c in CFG['components']]
        base = [f(te) for f in fns]
        s0 = S0[S0.t0 == v].set_index('cell').loc[te.cell.values]
        chk[v] = dict(xgbc=float(np.abs(base[1] - s0.p_xgbc.values).max()), nb=float(np.abs(base[2] - s0.p_nb.values).max()), logit=float(np.abs(base[0] - s0.p_logit.values).max()))
        ens = lambda comps: np.mean([rankdata(p) / len(te) for p in comps], 0)
        a_full = auc(te.yb.values, ens(base))
        for g, fs in GROUPS.items():
            drops = []
            for rep in range(5):
                tp = te.copy(); perm = rng.permutation(len(tp))
                for f in fs: tp[f] = te[f].values[perm]          # 그룹 내 같은 순열(피처 간 상관 유지)
                comps = [fns[i](tp) if any(f in CFG['components'][i]['features'] for f in fs) else base[i] for i in range(3)]
                drops.append(a_full - auc(te.yb.values, ens(comps)))
            rows.append(dict(t0=v, group=g, auc_full=a_full, auc_drop_mean=float(np.mean(drops)), auc_drop_sd=float(np.std(drops))))
        print(v, round(time.time() - t), chk[v], flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(CACHE, f"s5_perm_dev{'' if not args.origins else '_' + args.origins.replace(',', '-')}.csv"), index=False, encoding='utf-8-sig')
    out = dict(check_vs_S0=chk, summary=R.groupby('group').auc_drop_mean.agg(['mean', 'min', 'max']).round(5).to_dict('index'))
    json.dump(out, open(os.path.join(CACHE, f"s5_perm_dev{'' if not args.origins else '_' + args.origins.replace(',', '-')}.json"), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
else:
    ft = 2021; D = prep(C.get_D(t0s=list(range(2005, 2019)) + [ft])); rows = []
    for i, c in enumerate(CFG['components']):
        if c['type'] == 'logit': continue
        tr = train_set(c, D, ft); _, m = fit(c, tr)
        gs = m.get_score(importance_type='total_gain'); tot = sum(gs.values())
        for j, f in enumerate(c['features']):
            rows.append(dict(component=c['type'], feature=f, total_gain_share=gs.get(f'f{j}', 0.0) / tot))
    G = pd.DataFrame(rows); G.to_csv(os.path.join(CACHE, 's5_gain_final_train.csv'), index=False, encoding='utf-8-sig')
    AF = AF5 + AFK
    out = {comp: dict(AF=float(g[g.feature.isin(AF)].total_gain_share.sum()), AF5=float(g[g.feature.isin(AF5)].total_gain_share.sum()),
                      AFK=float(g[g.feature.isin(AFK)].total_gain_share.sum()), top10=g.sort_values('total_gain_share', ascending=False).head(10)[['feature', 'total_gain_share']].round(4).values.tolist(),
                      AF_ranks={f: int(g.total_gain_share.rank(ascending=False)[g.feature == f].iloc[0]) for f in AF}, n_feats=int(len(g)))
           for comp, g in G.groupby('component')}
    json.dump(out, open(os.path.join(CACHE, 's5_gain_final_train.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:5000]); print('총', round(time.time() - t))
