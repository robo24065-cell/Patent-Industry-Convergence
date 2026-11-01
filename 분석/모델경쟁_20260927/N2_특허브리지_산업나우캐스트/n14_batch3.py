# -*- coding: utf-8 -*-
"""N2 3차 배치. 챔피언 = E24(g_pidx·gd·HS6수출, QR, 실시간 재무, 시점판 가중 c=2)."""
import n_lib as L
A = L.Arena()
f = lambda: A.champ["feats"]
A.trial("E25", "특허 배분가중 → 최근 5년 등록특허 KSIC 구성(w_pit_rec)", w="pit_rec")
A.trial("E26", "+작년 조사 성장 g_lag (E8 롤백 변형: QR로 이상치 둔감)", feats=f() + ["g_lag"])
A.trial("E27", "라벨 신뢰도 c=5 (E14 재도전)", conf_c=5)
A.trial("E28", "가중 w_pit_share (E22 재도전)", w="pit_share")
A.trial("E29", "+재고지수 12월 전년동월비 g_inv", feats=f() + ["g_inv"])
A.trial("E30", "+내수·수출 출하지수 (E06 재도전)", feats=f() + ["g_dom", "g_exp"])
A.trial("E31", "+하반기 생산지수 성장 g_pidx_h2", feats=f() + ["g_pidx_h2"])
A.trial("E32", "최근 연도 가중 감쇠 0.9 (E20 변형)", decay=0.9)
A.trial("E33", "+DART 업종코드 배분 병행(스냅샷, 파일럿 E12 재도전)", induty=True, feats=f() + ["gd_induty_f"])
A.trial("E34", "QR L1 벌점 α=0.002", alpha=0.002)
print("champion now", A.champ_id, A.champ)
