# -*- coding: utf-8 -*-
"""한국발 해외출원 피처(kab_*)를 아이디어 B(고용)·D(수출) 챔피언에 추가하는 도전 실험."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ilib import *
def attach(P):
    E = []
    for t0 in sorted(P.t0.unique()):
        f = kab_feats(t0); f["t0"] = t0; E.append(f.reset_index())
    return P.merge(pd.concat(E), on=["ksic", "t0"], how="left")
def run(P, F, VS, tg="y", a=5.0):
    D = P.dropna(subset=[tg]); Q = pd.concat([g.assign(**{c: g[c].rank(pct=True).fillna(.5) for c in F+[tg]}) for _, g in D.groupby("t0")]); o = {}
    for v in VS:
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
        o[v] = spearmanr(D[D.t0 == v].set_index("ksic").loc[te.ksic, tg].values, Ridge(alpha=a).fit(tr[F], tr[tg]).predict(te[F]))[0]
    return pd.Series(o)
# B
PB = pd.read_parquet(f"{CACHE}/panel_kosis3y_ext.parquet"); pv = load_pat()
L2 = [pat_feats(pv, t0, 2)[["mom_KR", "dsh_KR"]].add_suffix("_L2").assign(t0=t0).reset_index() for t0 in sorted(PB.t0.unique())]
PB = attach(PB.merge(pd.concat(L2), on=["ksic", "t0"], how="left"))
for s in ["kab_mom", "kab_sh", "kab_dsh", "kab_lvl"]:
    r = np.array([spearmanr(g.y_emp, g[s], nan_policy="omit")[0] for _, g in PB.groupby("t0")]); print(f"[B 고용] {s} 단일 ρ {r.mean():+.3f} 양수 {np.mean(r>0):.2f}")
CB = ["mom_KR_L2", "dsh_KR_L2", "cnsh", "dcnsh", "lvl_KR", "frgn_mom"]; VB = range(2011, 2019); chB = run(PB, CB, VB, "y_emp"); print(f"B 챔피언 ρ {chB.mean():.3f}")
for s in ["kab_mom", "kab_sh", "kab_dsh", "kab_lvl"]:
    x = run(PB, CB+[s], VB, "y_emp"); d = x-chB; print(f"  B +{s} ρ {x.mean():.3f} Δ {d.mean():+.3f} 승률 {(d>0).mean():.2f} {'ACCEPT' if d.mean()>=0.003 and (d>0).mean()>=0.6 else 'ROLLBACK'}")
# D
PD = attach(pd.read_parquet(f"{CACHE}/panel_krexp_ksic.parquet"))
for s in ["kab_mom", "kab_sh", "kab_dsh", "kab_lvl"]:
    r = np.array([spearmanr(g.y, g[s], nan_policy="omit")[0] for _, g in PD.groupby("t0")]); print(f"[D 수출] {s} 단일 ρ {r.mean():+.3f} 양수 {np.mean(r>0):.2f}")
CD = ["cnsh", "dcnsh", "mom", "frgn_mom", "xlv"]; VD = range(2008, 2019); chD = run(PD, CD, VD); print(f"D 챔피언 ρ {chD.mean():.3f}")
for s in ["kab_mom", "kab_sh", "kab_dsh", "kab_lvl"]:
    x = run(PD, CD+[s], VD); d = x-chD; print(f"  D +{s} ρ {x.mean():.3f} Δ {d.mean():+.3f} 승률 {(d>0).mean():.2f} {'ACCEPT' if d.mean()>=0.003 and (d>0).mean()>=0.6 else 'ROLLBACK'}")
