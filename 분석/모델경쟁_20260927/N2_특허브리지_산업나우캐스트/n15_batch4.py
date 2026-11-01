# -*- coding: utf-8 -*-
"""N2 4차 배치. 챔피언 = E25(최근 5년 특허 KSIC 구성 가중)."""
import n_lib as L
A = L.Arena()
f = lambda: A.champ["feats"]
A.trial("E35", "gd 결측 보완값 g_idx → g_pidx", fill="g_pidx")
A.trial("E36", "최근 등록 창 5년 → 3년(w_pit_rec3)", w="pit_rec3")
A.trial("E37", "최근 등록 창 5년 → 8년(w_pit_rec8)", w="pit_rec8")
A.trial("E38", "+전체 생존 포트폴리오 배분 성장 gd2(w_pit) 병행", gd2="pit", feats=f() + ["gd2_f"])
A.trial("E39", "최근 연도 가중 감쇠 0.9 (E32 재도전)", decay=0.9)
A.trial("E40", "모형 QR+Huber 평균 앙상블", model="qr_huber")
A.trial("E41", "재무 rt2 → rt(같은 보고서 당기/전기)", fin="rt")
A.trial("E42", "라벨 신뢰도 c=3", conf_c=3)
print("champion now", A.champ_id, A.champ)
