# -*- coding: utf-8 -*-
# independent lift: share of pairs with max design/um P>0.2 (rgst<=cut-12-31) and their entry rate / overall
import pathlib, json, numpy as np, pandas as pd
B = pathlib.Path(r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석')
W = B/'검증루프_20261012/C10/C10-T1'; R3 = B/'모델경쟁_20260927/T1_산업대기실/R3'
out = {}
for c, FC in ((2012, R3/'cache/o2012'), (2019, R3/'cache_final/o2019')):
    if c == 2019:
        te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k']); L = pd.read_parquet(W/'cache/final_labels.parquet')
        te['y'] = L.y_u.values; te['pa'] = L.prior_any.values
    else:
        te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'y_u', 'prior_any']).rename(columns={'y_u': 'y', 'prior_any': 'pa'})
    te = te[~te.pa.values.astype(bool)].reset_index(drop=True)
    for kind in ('design', 'um'):
        P = np.load(FC/f'P_{kind}.npy'); ix = pd.read_parquet(FC/f'P_{kind}_index.parquet')
        rg = ix.rgst_date.astype(str).str[:8].astype(np.int64).values
        firms = pd.Index(te.firm.unique()); fi = pd.Series(np.arange(len(firms)), index=firms)
        m = (rg <= int(f'{c}1231')) & ix.applicant_cd.isin(firms).values
        r = fi[ix.applicant_cd.values[m]].values; MX = np.zeros((len(firms), P.shape[1]), np.float32); np.maximum.at(MX, r, P[m].astype(np.float32))
        v = MX[fi[te.firm.values].values, te.k.values.astype(int)]; hi = v > 0.2; y = te.y.values.astype(float)
        out[f'o{c}_{kind}'] = dict(share_hi=float(hi.mean()), lift=float(y[hi].mean() / y.mean()))
print(json.dumps(out, indent=1)); json.dump(out, open(W/'verify_1/v1_lift_out.json', 'w'), indent=1)
