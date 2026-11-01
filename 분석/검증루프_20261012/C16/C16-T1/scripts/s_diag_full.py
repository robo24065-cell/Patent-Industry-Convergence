# -*- coding: utf-8 -*-
"""S1-full과 S1-fixed 차이 진단(사후): C 법인 공식 행의 점수 변화가 어떤 피처에서 오는가(ctrlF 대비 s1F 피처 차)"""
import sys, pathlib, pickle; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
PP = pickle.load(open(CC/'build'/'pop.pkl', 'rb')); fs = set(PP['firms_sel'])
c = pd.read_parquet(CC/'build'/'ctrlF'/'te_cols.parquet', filters=[('firm', 'in', list(fs))])
s = pd.read_parquet(CC/'build'/'s1F'/'te_cols.parquet', filters=[('firm', 'in', list(fs))])
sc_c = pd.read_parquet(CC/'build'/'ctrlF'/'scores.parquet', filters=[('firm', 'in', list(fs))]); sc_s = pd.read_parquet(CC/'build'/'s1F'/'scores.parquet', filters=[('firm', 'in', list(fs))])
key = lambda d: pd.MultiIndex.from_arrays([d.firm.values, d.k.values])
si = s.set_index(['firm', 'k']).reindex(key(c)); ssi = sc_s.set_index(['firm', 'k']).reindex(key(sc_c))
L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].set_index(['firm', 'k']).reindex(key(c))
out = {}
for col in c.columns:
    if col in ('firm', 'k'): continue
    a = c[col].values.astype(float); b = si[col].values.astype(float); d = np.abs(np.nan_to_num(a, nan=-999) - np.nan_to_num(b, nan=-999))
    if (d > 0).any(): out[col] = dict(rows_changed=int((d > 0).sum()), mean_signed=float(np.nanmean(b - a)), max_abs=float(d.max()))
keep = ~L.prior_any.values; y = L.y_u.values
dpx = ssi.px.values - sc_c.px.values
res = dict(rows=int(len(c)), changed_features=out, px_change_mean=float(dpx[keep].mean()), px_change_pos_mean=float(dpx[keep & (y == 1)].mean()), px_change_neg_mean=float(dpx[keep & (y == 0)].mean()),
           pos_rows=int((keep & (y == 1)).sum()), pos_rate_Cfirms_official_rows=float(y[keep].mean()),
           px_rel_change_pos_median=float(np.median((ssi.px.values / np.maximum(sc_c.px.values, 1e-9))[keep & (y == 1)])),
           px_rel_change_neg_median=float(np.median((ssi.px.values / np.maximum(sc_c.px.values, 1e-9))[keep & (y == 0)])))
jdump(res, CC/'s1full_diag.json'); print(json.dumps(res, ensure_ascii=False, indent=1))
