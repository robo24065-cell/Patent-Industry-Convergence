# -*- coding: utf-8 -*-
"""N2 5차 배치. 챔피언 = E37(최근 8년 특허 KSIC 구성)."""
import n_lib as L
A = L.Arena()
f = lambda: A.champ["feats"]
#A.trial("E43", "+11차 조사체계 더미 d11(10차→11차 전환 수준 차이)", feats=f() + ["d11"])
#A.trial("E44", "+g_idx(출하지수) 병행", feats=f() + ["g_idx"])
A.trial("E45", "라벨 신뢰도 수축을 최근 8년 KSIC 건수 기준으로", conf_on="n_k_rec8")
A.trial("E46", "학습 시작 2015(2014 학습행은 2015 보고서 비교수치 → 실시간 아님, 제거)", train_start=2015)
A.trial("E47", "기업 성장 절단 ±1.5 (완화)", clip_gi=1.5)
print("champion now", A.champ_id, A.champ)
