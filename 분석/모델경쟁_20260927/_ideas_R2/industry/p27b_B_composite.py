# -*- coding: utf-8 -*-
"""아이디어 B 타깃 변형: '산업 활력 복합지수' = 3년 상대성장 순위(종사자·급여·출하·부가가치) 평균. 동일 6피처, 순위 Ridge."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ilib import *
K = load_kosis50(); pv = load_pat()
F = ["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom", "cnsh", "dcnsh"]
rows = []
for t0 in range(2008, 2019):
    if t0 == 2010: continue
    d = pat_feats(pv, t0)[["lvl_KR", "frgn_mom"]].join(pat_feats(pv, t0, 2)[["mom_KR", "dsh_KR"]].add_suffix("_L2")).join(trade_feats(t0)[["cnsh", "dcnsh"]])
    for it in ["T02", "T03", "T04", "T06"]: d[it] = lg(K[it], t0+3, t0).reindex(d.index).rank(pct=True)
    d["y"] = d[["T02", "T03", "T04", "T06"]].mean(1, skipna=False); d["t0"] = t0
    d["mom"] = pd.concat([lg(K[it], t0, max(t0-3, 2008)).reindex(d.index).rank(pct=True) for it in ["T02", "T03", "T04", "T06"]], axis=1).mean(1) if t0 > 2008 else np.nan
    rows.append(d.dropna(subset=["y"]).reset_index())
P = pd.concat(rows); Q = pd.concat([g.assign(**{c: g[c].rank(pct=True).fillna(.5) for c in F}) for _, g in P.groupby("t0")])
rm, rb, rp = [], [], []
for v in range(2011, 2019):
    tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
    p = Ridge(alpha=5).fit(tr[F], tr.y).predict(te[F]); rm.append(spearmanr(te.y, p)[0]); rb.append(spearmanr(te.y, te.mom.fillna(.5))[0])
    pp = Ridge(alpha=5).fit(tr[F[:4]], tr.y).predict(te[F[:4]]); rp.append(spearmanr(te.y, pp)[0])
rm, rb, rp = map(np.array, (rm, rb, rp))
print(f"복합지수 3년: ρ_모형 {rm.mean():.3f} (원점별 {rm.round(2).tolist()}) | 특허만 {rp.mean():.3f} | ρ_모멘텀 {rb.mean():.3f} | 승률 {np.mean(rm>rb):.2f}")
