# -*- coding: utf-8 -*-
"""아이디어 B 해석: 최종 6피처 Ridge 계수(학습 t0 ≤2015) + 개발 원점 2018(→2021) 상·하위 산업 예시."""
import numpy as np, pandas as pd, json
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
exec(open("p25_B_final_pilot.py", encoding="utf-8").read().split("# 기준선")[0])
F = ["mom_KR_L2", "dsh_KR_L2", "cnsh", "dcnsh", "lvl_KR", "frgn_mom"]
D = P.dropna(subset=[TG]); Q = pd.concat([g.assign(**{c: g[c].rank(pct=True).fillna(.5) for c in F+[TG]}) for _, g in D.groupby("t0")])
m = Ridge(alpha=5).fit(Q[Q.t0 <= 2015][F], Q[Q.t0 <= 2015][TG]); print("계수", dict(zip(F, m.coef_.round(3))))
te = Q[Q.t0 == 2018].copy(); te["p"] = m.predict(te[F]); te["y"] = D[D.t0 == 2018].set_index("ksic").loc[te.ksic, TG].values
nm = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str}).drop_duplicates("세세분류코드").set_index("세세분류코드").산업명
te["name"] = te.ksic.map(nm); te = te.sort_values("p", ascending=False)
print("상위5", te.head(5)[["ksic", "name", "p", "y"]].round(3).to_string()); print("하위5", te.tail(5)[["ksic", "name", "p", "y"]].round(3).to_string())
print("ρ 2018", round(spearmanr(te.y, te.p)[0], 3))
