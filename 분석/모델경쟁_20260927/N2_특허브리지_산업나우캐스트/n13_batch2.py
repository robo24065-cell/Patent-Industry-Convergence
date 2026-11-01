# -*- coding: utf-8 -*-
"""N2 2차 배치: 롤백된 변경의 변형 재시도 + 신규 후보. 챔피언 = C1_rt(실시간 재무 rt2, 시점판 가중, QR, 라벨신뢰도 c=2, +HS6 수출)."""
import n_lib as L
A = L.Arena()
f = lambda: A.champ["feats"]
A.trial("E13", "+생산지수 성장 g_pidx (E03 재도전, QR·수출 위)", feats=f() + ["g_pidx"])
A.trial("E14", "라벨 신뢰도 수축 강화 c=5", conf_c=5)
A.trial("E15", "라벨 신뢰도 수축 완화 c=1", conf_c=1)
A.trial("E16", "+DART 자산성장 배분(E04 재도전)", assets=True, feats=f() + ["gd_assets_f"])
A.trial("E17", "gd를 기업수 기반으로 지수 쪽 수축 nf/(nf+5) (파일럿 E4 변형)", shrink_k=5)
A.trial("E18", "매출 규모 완화 a=w·rev^0.7", size_pow=0.7)
A.trial("E19", "기업 성장 절단 ±0.5", clip_gi=0.5)
A.trial("E20", "최근 연도 가중 학습(감쇠 0.85)", decay=0.85)
A.trial("E21", "3자리 계층 정합 후처리(λ=0.5)", post="hier", hier_lam=0.5)
A.trial("E22", "가중 w_pit → w_pit_share(제공범위 밖 축소 제거)", w="pit_share")
A.trial("E23", "+제조업 전체 출하지수 성장 g_mfg", feats=f() + ["g_mfg"])
A.trial("E24", "지수 피처 g_idx → g_pidx 대체", feats=["g_pidx" if x == "g_idx" else x for x in f()])
print("champion now", A.champ_id, A.champ)
