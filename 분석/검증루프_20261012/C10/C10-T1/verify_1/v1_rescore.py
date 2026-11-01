# -*- coding: utf-8 -*-
"""verify_1: S3 claim independent rescoring (no import of worker/frozen code). Writes only into verify_1."""
import json, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
B = pathlib.Path(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석')
W = B/'검증루프_20261012/C10/C10-T1'; R3 = B/'모델경쟁_20260927/T1_산업대기실/R3'; OUT = W/'verify_1'
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def pct(v): return rankdata(v) / len(v)
def bl(o): return 0.9 * pct(o.px.values) + 0.1 * pct(o.pl.values)
res = {}
L = pd.read_parquet(W/'cache/final_labels.parquet')
fz = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
res['frozen_y_u_equals_final_labels'] = bool(np.array_equal(np.nan_to_num(fz.y_u.values, nan=-1), np.nan_to_num(L.y_u.values, nan=-1)))
res['frozen_y_u_nan'] = int(np.isnan(fz.y_u.values.astype(float)).sum())
keep = ~L.prior_any.values.astype(bool)
y = L.y_u.values[keep].astype(int); nc = L.newc.values[keep].astype(bool); firm = L.firm.values[keep]
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet').loc[keep].reset_index(drop=True)
base = max(auc(y, bs[k].values) for k in bs.columns); base_new = max(auc(y[nc], bs[k].values[nc]) for k in bs.columns)
res['base_auc'] = base; res['base_auc_new'] = base_new
off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
res['official'] = dict(auc=off['model']['auc_all'], skill=off['skill']['auc_all'])
S = {}
for v in ['FROZEN', 'S0', 'S3', 'S1']:
    p = R3/'runs/FROZEN_final/o2019.parquet' if v == 'FROZEN' else W/f'runs/{v}/o2019.parquet'
    o = pd.read_parquet(p); assert (o.firm.values == L.firm.values).all() and (o.k.values == L.k.values).all()
    o = o[keep].reset_index(drop=True); s = bl(o); S[v] = s
    a = auc(y, s); an = auc(y[nc], s[nc])
    res[v] = dict(auc=a, skill=(a - base) / (1 - base), auc_new=an, skill_new=(an - base_new) / (1 - base_new))
res['dS3'] = res['S3']['skill'] - res['S0']['skill']; res['dS3_new'] = res['S3']['skill_new'] - res['S0']['skill_new']
res['px_S3_vs_S0_maxdiff'] = float(np.abs(S['S3'] - S['S0']).max())
# independent firm-cluster bootstrap (own RNG), baseline recomputed per resample (strongest of 6)
fr_, inv = np.unique(firm, return_inverse=True); order = np.argsort(inv, kind='stable')
st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
rng = np.random.RandomState(777); D = []; DA = []
BSv = bs.values
for r in range(120):
    idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
    bm = max(auc(yy, BSv[idx, j]) for j in range(BSv.shape[1]))
    a0 = auc(yy, S['S0'][idx]); a3 = auc(yy, S['S3'][idx])
    D.append((a3 - bm) / (1 - bm) - (a0 - bm) / (1 - bm)); DA.append(a3 - a0)
D = np.array(D); res['boot_own'] = dict(R=len(D), d_skill_q=[float(np.percentile(D, q)) for q in (2.5, 50, 97.5)], share_neg=float((D < 0).mean()),
                                         d_auc_q=[float(np.percentile(DA, q)) for q in (2.5, 50, 97.5)])
# gain share
g0 = pd.read_csv(W/'runs/S0/o2019_gain.csv', header=None, names=['f', 'g']).dropna(); g0 = g0[g0.f.astype(str).str.len() > 0]
g0 = g0.sort_values('g', ascending=False).reset_index(drop=True)
m = g0.f.str.startswith('ds_') | g0.f.str.startswith('um_')
res['gain_share_dsum'] = float(g0.g[m].sum() / g0.g.sum()); res['gain_rank_dsum'] = {r.f: int(i + 1) for i, r in g0[m].iterrows()}
g3 = pd.read_csv(W/'runs/S3/o2019_gain.csv', header=None, names=['f', 'g']).dropna()
res['S3_gain_has_dsum'] = g3.f.astype(str).str.match(r'^(ds_|um_)').sum(); res['n_feat_S0'] = int(len(g0)); res['n_feat_S3'] = int(len(g3))
res['S0_other_design_like'] = [f for f in g0.f if ('des' in f or 'um' in f.lower()) and not (f.startswith('ds_') or f.startswith('um_'))]
# dev origins independent
dev = []
for c in range(2008, 2013):
    rows = {}
    for v, p in (('FULL', R3/f'runs/FROZEN_dev/o{c}.parquet'), ('S3', W/f'runs/S3/o{c}.parquet')):
        o = pd.read_parquet(p); o = o[~o.prior_any.values.astype(bool)].reset_index(drop=True)
        rows[v] = (auc(o.y_u.values.astype(int), bl(o)), len(o), int(o.y_u.sum()))
    dev.append(dict(c=c, auc_full=rows['FULL'][0], auc_s3=rows['S3'][0], d_auc=rows['S3'][0] - rows['FULL'][0], n=rows['FULL'][1], n_s3=rows['S3'][1]))
res['dev'] = dev
res['S3_final_d_auc'] = res['S3']['auc'] - res['S0']['auc']
res['dev_mean_d_auc'] = float(np.mean([d['d_auc'] for d in dev]))
json.dump(res, open(OUT/'v1_rescore_out.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
print(json.dumps(res, indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
