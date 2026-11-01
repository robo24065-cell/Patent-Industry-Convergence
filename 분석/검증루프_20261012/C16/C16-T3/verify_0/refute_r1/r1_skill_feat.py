"""반박 r1: (A) 예측 파일에서 sklearn AUC로 skill·Δ 재계산 (B) 공식 panel_ccy에서 S0·S2a 피처를 pandas로 독립 재계산해 예측 파일 피처와 대조
(C) 다른 seed·셀 단위 부트스트랩으로 S2a 최종 Δ CI 재확인. 읽기 전용."""
import os, sys, glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
WD = sys.argv[1]; R3 = sys.argv[2]; OUT = sys.argv[3]
C = os.path.join(WD, 'cache')
def load(s, mode):
    fs = sorted(glob.glob(os.path.join(C, f'preds_{s}_{mode}*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs]); return d.drop_duplicates(['cell','t0']).sort_values(['t0','cell']).reset_index(drop=True)
res = {}
P = {m: {s: load(s, m) for s in ['S0','S1','S2a','S2b']} for m in ['final','dev']}
for m in P:
    for s, d in P[m].items():
        sk = {}
        for t0, g in d.groupby('t0'):
            a = roc_auc_score(g.yb, g.p); b = roc_auc_score(g.yb, g.p_base); sk[int(t0)] = (a-b)/(1-b)
        res[f'{m}_{s}'] = dict(mean=float(np.mean(list(sk.values()))), by=sk)
for m in P:
    for s in ['S1','S2a','S2b']:
        res[f'{m}_{s}']['delta'] = res[f'{m}_{s}']['mean'] - res[f'{m}_S0']['mean']
        res[f'{m}_{s}']['n_up'] = int(sum(res[f'{m}_{s}']['by'][k] > res[f'{m}_S0']['by'][k] for k in res[f'{m}_S0']['by']))
# official preds
off = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
mm = P['final']['S0'].merge(off, on=['cell','t0'], suffixes=('','_o'))
res['final_S0_vs_off_pmax'] = float((mm.p-mm.p_o).abs().max()); res['final_off_skill'] = float((roc_auc_score(off.yb, off.p)-roc_auc_score(off.yb, off.p_base))/(1-roc_auc_score(off.yb, off.p_base)))
# (B) independent features
def feats(src, t0s, preds_by_s):
    pc = pd.read_parquet(os.path.join(R3, src, 'panel_ccy.parquet'), columns=['ksic','mg','cc','year','n','n_newapp_apps','n_newapp'])
    pc['cell'] = pc.ksic + '|' + pc.mg
    out = {}
    for t0 in t0s:
        w = pc[pc.year.between(t0-2, t0)]
        a = w.groupby('cell')[['n','n_newapp_apps','n_newapp']].sum()
        b = w[w.cc != 'CN'].groupby('cell')[['n','n_newapp_apps','n_newapp']].sum().reindex(a.index).fillna(0)
        f = pd.DataFrame({'sh_S0': a.n_newapp_apps/np.maximum(a.n,1), 'n_S0': np.log1p(a.n_newapp),
                          'sh_S2a': b.n_newapp_apps/np.maximum(b.n,1), 'n_S2a': np.log1p(b.n_newapp)})
        for s in ['S0','S2a']:
            g = preds_by_s[s]; g = g[g.t0 == t0].set_index('cell')
            j = g.join(f, how='left')
            out[f'{t0}_{s}'] = dict(n=int(len(j)), nmiss=int(j[f'sh_{s}'].isna().sum()),
                                   max_sh=float((j.all_newapp_sh - j[f'sh_{s}']).abs().max()), max_n=float((j.all_newapp_n - j[f'n_{s}']).abs().max()))
        g0 = preds_by_s['S0']; g0 = g0[g0.t0 == t0].set_index('cell').join(f)
        out[f'{t0}_change'] = dict(mean_sh_S0=float(g0.sh_S0.mean()), mean_sh_S2a=float(g0.sh_S2a.mean()),
                                   spearman_sh=float(g0.sh_S0.rank().corr(g0.sh_S2a.rank())), cn_share_in_newapp=float(1-(g0.n_S2a.map(np.expm1).sum()/g0.n_S0.map(np.expm1).sum())))
    return out
res['feat_final'] = feats('cache_pub_final', [2021], P['final'])
res['feat_dev'] = feats('cache_pub', [2010, 2014, 2018], P['dev'])
# (C) bootstrap alternatives for final S2a/S1/S2b vs S0
def auc_fast(y, p):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
d0 = P['final']['S0']
for unit in ['mg', 'cell', 'ksic']:
    for seed in [7, 2026]:
        rng = np.random.default_rng(seed)
        grp = d0[unit].values; ug, inv = np.unique(grp, return_inverse=True)
        idx_by = pd.Series(np.arange(len(d0))).groupby(inv).apply(lambda s: s.values).to_dict()
        D = {s: [] for s in ['S1','S2a','S2b']}
        for _ in range(1000):
            pick = rng.integers(0, len(ug), len(ug)); ii = np.concatenate([idx_by[k] for k in pick])
            y = d0.yb.values[ii]; b = auc_fast(y, d0.p_base.values[ii]); a0 = auc_fast(y, d0.p.values[ii])
            for s in D:
                a = auc_fast(y, P['final'][s].p.values[ii]); D[s].append((a-b)/(1-b) - (a0-b)/(1-b))
        res[f'boot_final_{unit}_seed{seed}'] = {s: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(np.array(v) > 0))] for s, v in D.items()}
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or 'by' not in v else {kk: vv for kk, vv in v.items()}) for k, v in res.items()}, ensure_ascii=False, indent=1))
