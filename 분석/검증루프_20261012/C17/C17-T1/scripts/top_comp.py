# -*- coding: utf-8 -*-
"""공식 상위 100·1,000 구성: 신규진입자·N1 비가시 신규진입자 행·양성 수 → cache/top_comp.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
O = official_frame(); Ok = O[~O.prior_any.values].reset_index(drop=True)
s = 0.9 * pd.Series(Ok.px.values).rank(pct=True).values + 0.1 * pd.Series(Ok.pl.values).rank(pct=True).values
o = np.argsort(-s, kind='stable'); E = pd.read_parquet(CC/'evid_all.parquet'); E = E[E.is_newc_off]; inv = set(E.firm[~E.N1_primary])
nc = Ok.newc.values; iv = Ok.firm.isin(inv).values & nc; y = Ok.y_off.values; out = {}
for K_ in (100, 1000):
    t = o[:K_]; out[f'top{K_}'] = dict(hits=int(y[t].sum()), newc_rows=int(nc[t].sum()), newc_hits=int((y[t] * nc[t]).sum()), invis_newc_rows=int(iv[t].sum()), invis_newc_hits=int((y[t] * iv[t]).sum()))
out['newc_share_rows'] = float(nc.mean()); out['newc_share_pos'] = float(y[nc].sum() / y.sum())
jdump(out, CC/'top_comp.json'); print(out)
