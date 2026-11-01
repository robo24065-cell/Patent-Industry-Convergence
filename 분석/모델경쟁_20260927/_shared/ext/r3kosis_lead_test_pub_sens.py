# -*- coding: utf-8 -*-
"""공개연도 선행 검정 민감도: 원점 부분집합(2009–16 / 2011–18), 성장률 윈저라이즈(|g|≤0.5), 짝지은 품목≥3 조건.
입력: r3kosis_lead_test_pub_panel.parquet (KR 공개연도 ≤2021만으로 만든 패널)."""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata
EXT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
D = pd.read_parquet(os.path.join(EXT, "r3kosis_lead_test_pub_panel.parquet")); D["ksic"] = D.ksic.astype(str)
CH = pd.read_parquet(os.path.join(EXT, "r3kosis_ksic50_items_chain.parquet")); CH["ksic"] = CH.ksic.astype(str)
npair = CH.set_index(["ksic", "year"])["n_pairs"]
D["npair_L2"] = [npair.get((k, t - 2), np.nan) for k, t in zip(D.ksic, D.t0)]
def pc(x, y, m):
    x, y, m = rankdata(x), rankdata(y), rankdata(m); X = np.c_[np.ones_like(m), m]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])
def test(G, f, nperm=2000):
    gs = [g for _, g in G.dropna(subset=[f]).groupby("t0") if len(g) >= 15]
    per = [pc(g[f].values, g.Y.values, g.M.values) for g in gs]
    obs = np.mean(per)
    null = np.array([np.mean([pc(rng.permutation(g[f].values), g.Y.values, g.M.values) for g in gs]) for _ in range(nperm)])
    return dict(origins=len(gs), n_obs=int(sum(len(g) for g in gs)), mean=round(obs, 4), perm_p=round(float((np.abs(null) >= abs(obs)).mean()), 4),
                n_pos=int(sum(v > 0 for v in per)))
res = []
for f in ["g_prod_2y_L2", "g_ship_2y_L2", "inv_ship_lvl_L2"]:
    for lab, G in {"all_2009_2018": D, "origins_2009_2016": D[D.t0 <= 2016], "origins_2011_2018": D[D.t0 >= 2011],
                   "npair>=3": D[D.npair_L2 >= 3],
                   "winsor_0.5": D.assign(**{f: D[f].clip(-0.5, 0.5) if f.startswith("g_") else D[f]})}.items():
        r = test(G, f); r.update(feature=f, subset=lab); res.append(r); print(r)
pd.DataFrame(res).to_csv(os.path.join(EXT, "r3kosis_lead_test_pub_sens.csv"), index=False, encoding="utf-8-sig")
