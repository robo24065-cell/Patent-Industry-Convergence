# -*- coding: utf-8 -*-
"""진단: 후보 피처 단일 신호 Spearman ρ(원점별, y_emp) + 챔피언 잔차와의 상관(추가 정보 여부). 개발 원점 2008~2018."""
from engine import *
st = champ_state(); F = st["F"]; o = preds(F, alpha=st.get("alpha", 5.0))
def _res(d):
    a, b = d.y.rank(pct=True), d.p.rank(pct=True); bb = np.cov(a, b)[0, 1]/np.var(b, ddof=1)
    return a - a.mean() - bb*(b - b.mean())   # 원점 내 순위 회귀 잔차(챔피언 예측으로 설명되지 않는 부분)
res = pd.concat([d.assign(t0=v, r=_res(d)) for v, d in o.items()])
Q = P.merge(res[["ksic", "t0", "r"]], on=["ksic", "t0"], how="left")
skip = {"ksic", "t0", "r"} | {c for c in P.columns if c.startswith("y_")}
out = []
for c in P.columns:
    if c in skip or P[c].dtype == object: continue
    r = [spearmanr(g.y_emp, g[c], nan_policy="omit")[0] for _, g in P.dropna(subset=["y_emp"]).groupby("t0") if g[c].notna().sum() > 10]
    rr = [spearmanr(g.r, g[c], nan_policy="omit")[0] for _, g in Q.dropna(subset=["r"]).groupby("t0") if g[c].notna().sum() > 10]
    out.append((c, np.nanmean(r), np.mean(np.array(r) > 0) if r else np.nan, np.nanmean(rr) if rr else np.nan, np.mean(np.array(rr) > 0) if rr else np.nan, len(r)))
U = pd.DataFrame(out, columns=["feat", "rho_y", "pos_y", "rho_resid", "pos_resid", "n_orig"]).sort_values("rho_resid", key=abs, ascending=False)
U.to_csv(f"{R2}/diag_univar.csv", index=False, encoding="utf-8-sig")
print(U.round(3).head(45).to_string())
