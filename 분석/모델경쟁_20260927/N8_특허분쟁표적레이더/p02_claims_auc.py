# -*- coding: utf-8 -*-
"""청구항 사후값 누수 크기 진단: 양성을 '원점 이후(≤2022) 정정·심결확정·일부말소 사건 있음/없음'으로 나눠
-claims의 단변량 AUC(같은 원점 음성 전체 대비)를 비교. 봉인 캐시만 사용."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
rt = pd.read_parquet(f"{MC}/_ideas_R4/cache/reg_right_lt2023.parquet"); assert rt.d.max() < "20230101"
rt["d"] = pd.to_numeric(rt.d)
ev = rt[rt.cs.isin(["권리의 일부말소등록", "정정(심판)예고등록", "정정등록", "심결확정(초심)등록", "심결일부확정(초심)등록", "등록권리의 일부말소등록"])].groupby("rgst_no").d.max()
D = pd.read_parquet(f"{MC}/_ideas_R5/cache/J2_panel.parquet", columns=["rgst_no", "t0", "y", "claims", "utility"])
D = D[D.t0.between(2012, 2020)].copy()
D["post"] = D.rgst_no.map(ev).fillna(0) > D.t0 * 10000 + 1231
out = []
for t0, g in D.groupby("t0"):
    neg = g[g.y == 0]
    for lab, pos in (("all", g[g.y == 1]), ("post_event", g[(g.y == 1) & g.post]), ("no_post_event", g[(g.y == 1) & ~g.post])):
        s = pd.concat([pos, neg]); out.append(dict(t0=t0, grp=lab, npos=len(pos), auc=roc_auc_score(s.y, -s.claims.fillna(0))))
o = pd.DataFrame(out); print(o.pivot(index="t0", columns="grp", values="auc").round(3)); print(o.groupby("grp")[["npos", "auc"]].mean().round(3))
print("median claims pos post/no-post:", D[(D.y == 1) & D.post].claims.median(), D[(D.y == 1) & ~D.post].claims.median(), "neg", D[D.y == 0].claims.median())
