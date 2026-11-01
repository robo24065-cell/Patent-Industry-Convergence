# -*- coding: utf-8 -*-
"""청구항 수(현재 등록원부 값) 사후 변경 점검: 원점 이후(봉인 캐시 범위 ≤2022-12-31) 일부말소·정정·심결확정 사건이
분쟁 양성에서 얼마나 더 흔한가. 파일럿 패널(_ideas_R5 cache, 읽기 전용)의 y를 쓴다."""
import pandas as pd, numpy as np
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
rt = pd.read_parquet(f"{MC}/_ideas_R4/cache/reg_right_lt2023.parquet"); assert rt.d.max() < "20230101"
rt["d"] = pd.to_numeric(rt.d)
D = pd.read_parquet(f"{MC}/_ideas_R5/cache/J2_panel.parquet", columns=["rgst_no", "t0", "y", "claims", "utility", "own_disp"])
for nm in [["권리의 일부말소등록"], ["정정(심판)예고등록", "정정등록"], ["심결확정(초심)등록", "심결일부확정(초심)등록"]]:
    ev = rt[rt.cs.isin(nm)].groupby("rgst_no").d.apply(list).to_dict()
    res = []
    for t0, g in D[D.t0.between(2012, 2018)].groupby("t0"):
        E = t0 * 10000 + 1231
        post = g.rgst_no.map(lambda r: any(x > E for x in ev.get(r, [])))
        res.append(dict(t0=t0, pos_rate=post[g.y == 1].mean(), neg_rate=post[g.y == 0].mean()))
    r = pd.DataFrame(res); print(nm, r.round(4).to_dict("list"))
# 청구항 분포: 양성 vs 음성
g = D[D.t0.between(2012, 2020)]
print(g.groupby("y").claims.describe())
