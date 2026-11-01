# -*- coding: utf-8 -*-
"""아이디어 D 파일럿: KSIC 단위(48개) 한국 수출 3년 상대성장 순위 — 특허(KR·해외) + 대중 수입점유.
수출 = _shared/ext/R2global_kr_export_hs6.csv(HS6→KSIC 합산, 다중대응은 1/m 가중). y = log(Σ수출[t0+1..t0+3]/Σ수출[t0-2..t0]) − 원점 중앙값.
결과 ≤2021 → t0 ≤ 2018. 원점 2005..2018, 평가 v=2008..2018, 학습 t0 ≤ v−3. 지표: 원점별 Spearman ρ, 기준선 = 과거 3년 수출성장 순위."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ilib import *
x = pd.read_csv(os.path.join(B, "_shared", "ext", "R2global_kr_export_hs6.csv"), dtype={"hs6": str}); x = x[x.year <= DEV_MAX]; x["hs6"] = x.hs6.str.zfill(6)
mp = load_map(); mp["w"] = 1/mp.groupby("hs6").k5.transform("count")
k = x.merge(mp, on="hs6").assign(v=lambda d: d.value*d.w).pivot_table(index="k5", columns="year", values="v", aggfunc="sum")
pv = load_pat(); rows = []
for t0 in range(2005, 2019):
    d = pat_feats(pv, t0).join(pat_feats(pv, t0, 2)[["mom_KR", "dsh_KR"]].add_suffix("_L2")).join(app_feats(t0))
    c = pd.read_csv(os.path.join(BASE, "분석", "fusion", "cn_import_yearly_ksic5.csv"), dtype={"ksic": str})
    c = c[(c.year <= t0) & (c.ksic_level == 5)].pivot_table(index="ksic", columns="year", values="cn_share_imp")
    d["cnsh"] = c[t0]; d["dcnsh"] = c[t0]-c[t0-3]
    fut = k[[t0+1, t0+2, t0+3]].sum(1); now = k[[t0-2, t0-1, t0]].sum(1); past = k[[t0-5, t0-4, t0-3]].sum(1)
    d["y"] = np.log(fut/now).reindex(d.index); d["mom"] = np.log(now/past).reindex(d.index); d["xlv"] = np.log(now).reindex(d.index)
    d["t0"] = t0; rows.append(d.dropna(subset=["y"]).reset_index())
P = pd.concat(rows)
for s in ["mom_KR", "dsh_KR", "mom_KR_L2", "dsh_KR_L2", "frgn_mom", "mom_CN", "mom_ALL", "cnsh", "dcnsh", "mom", "xlv", "newmom_KR"]:
    r = np.array([spearmanr(g.y, g[s], nan_policy="omit")[0] for _, g in P.groupby("t0")]); print(f"{s:10s} ρ {r.mean():+.3f} 양수 {np.mean(r>0):.2f}")
def run(F, a=5.0):
    Q = pd.concat([g.assign(**{cc: g[cc].rank(pct=True).fillna(.5) for cc in F+["y"]}) for _, g in P.groupby("t0")]); o = {}
    for v in range(2008, 2019):
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]; p = Ridge(alpha=a).fit(tr[F], tr.y).predict(te[F])
        o[v] = spearmanr(P[P.t0 == v].y.values, p)[0]
    return pd.Series(o)
base = pd.Series({v: spearmanr(g.y, g.mom, nan_policy="omit")[0] for v, g in P.groupby("t0") if v >= 2008})
print(f"기준선(과거 수출성장 순위) ρ {base.mean():.3f}")
for lab, F in [("특허만(B의 4개)", ["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom"]), ("B 최종 6피처", ["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom", "cnsh", "dcnsh"]),
               ("B6 + 과거 수출성장", ["mom_KR_L2", "dsh_KR_L2", "lvl_KR", "frgn_mom", "cnsh", "dcnsh", "mom"]), ("무역만(cnsh,dcnsh,mom)", ["cnsh", "dcnsh", "mom"])]:
    s = run(F); d = s - base; print(f"{lab:24s} ρ {s.mean():.3f} (원점 양수 {np.mean(s>0):.2f}) vs 기준선 Δ {d.mean():+.3f} 승률 {np.mean(d>0):.2f}")
P.to_parquet(f"{CACHE}/panel_krexp_ksic.parquet")
