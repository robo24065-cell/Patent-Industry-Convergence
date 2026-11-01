# -*- coding: utf-8 -*-
"""아이디어 B 정식화 민감도: 결과 항목(종사자·사업체·급여·출하·부가가치) × 시계(2·3·4년). 결과 ≤2021.
피처 = 최종 챔피언 6개(특허 L2 모멘텀·점유변화, 특허 규모, 해외 모멘텀, 대중 수입점유 수준·변화). 비교: 모멘텀(과거 h년 성장 순위)."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ilib import *
K = load_kosis50(); pv = load_pat()
F = ["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom", "cnsh", "dcnsh"]
feat = {}
for t0 in range(2008, 2020):
    if t0 == 2010: continue
    a = pat_feats(pv, t0)[["lvl_KR", "frgn_mom"]].join(pat_feats(pv, t0, 2)[["mom_KR", "dsh_KR"]].add_suffix("_L2")).join(trade_feats(t0)[["cnsh", "dcnsh"]])
    feat[t0] = a
for it, nm in [("T02", "종사자"), ("T01", "사업체"), ("T03", "급여"), ("T04", "출하"), ("T06", "부가가치")]:
    for h in [2, 3, 4]:
        rows = []
        for t0 in feat:
            if t0+h > DEV_MAX or t0+h == 2010: continue
            d = feat[t0].copy(); d["y"] = lg(K[it], t0+h, t0).reindex(d.index); d["mom"] = lg(K[it], t0, max(t0-h, 2008)).reindex(d.index) if t0 > 2008 else np.nan
            d["t0"] = t0; rows.append(d.dropna(subset=["y"]).reset_index())
        P = pd.concat(rows)
        Q = pd.concat([g.assign(**{c: g[c].rank(pct=True).fillna(.5) for c in F+["y"]}) for _, g in P.groupby("t0")])
        rm, rb = [], []
        for v in sorted(P.t0.unique()):
            tr, te = Q[Q.t0 <= v-h], Q[Q.t0 == v]
            if tr.t0.nunique() < 2 or v < 2011: continue
            p = Ridge(alpha=5).fit(tr[F], tr.y).predict(te[F]); yr = P[P.t0 == v].y.values
            rm.append(spearmanr(yr, p)[0]); rb.append(spearmanr(yr, P[P.t0 == v].mom.fillna(0).values)[0])
        rm, rb = np.array(rm), np.array(rb)
        print(f"{nm:4s} h={h} 원점 {len(rm)}  ρ_모형 {rm.mean():.3f} (양수 {np.mean(rm>0):.2f})  ρ_모멘텀 {np.nanmean(rb):.3f}  승률 {np.mean(rm>rb):.2f}")
