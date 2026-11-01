# -*- coding: utf-8 -*-
"""N2 누수 진단·재기준: 파일럿 재무(fin_first)는 회계연도 2016·17·19·20·22 값을 이후 보고서 비교수치에서 가져와
나우캐스트 시점(t+1년 4월)에 없던 기업(이후 상장·공시 시작)과 재작성 값을 포함한다. 실시간판 2종을 현 챔피언 위에서 비교하고
더 나은 실시간판으로 챔피언을 재기준한다(누수 보완은 채택 규칙 밖의 필수 수정)."""
import numpy as np, pandas as pd
import n_lib as L
A = L.Arena()
m_rt2, _, _ = A.trial("E11", "재무 → 실시간판2(t년 보고서 당기 / t−1년 보고서 당기)", fin="rt2")
m_rt, _, _ = A.trial("E12", "재무 → 실시간판(t년 보고서 당기/전기)", fin="rt")
# 진단: 기업 수·원점별 차이
for src in ["first", "rt", "rt2"]:
    fr = L.firm_growth(src)
    print(src, "firms with growth by t", fr.groupby("t").size().loc[2016:2023].to_dict())
best = "rt2" if m_rt2["mean_skill"] >= m_rt["mean_skill"] else "rt"
A.rebase("C1_rt", f"챔피언 재기준(누수보완): 재무 fin_first → {best}", note="fin_first는 이후 보고서 비교수치로 나우캐스트 시점에 없던 기업·값 포함", fin=best)
print(A.champ)
