# -*- coding: utf-8 -*-
"""N2 챔피언/도전자 1차 배치 (PROTOCOL_R2 §3: +0.003 & 원점 60% 이상 → ACCEPT, 아니면 그 변경만 ROLLBACK)."""
import n_lib as L

A = L.Arena()
f = lambda: A.champ["feats"]
A.trial("E01", "DART 실시간판: 회계연도 t 성장 = t년 사업보고서 당기/전기(N2 추가 수집 6개 연도 포함)", fin="rt")
A.trial("E02", "모형 → 분위회귀(중앙값, MAE 직접 최소화)", model="qr", alpha=0.0)
A.trial("E03", "+생산지수(원지수) 3자리 t년 성장 g_pidx", feats=f() + ["g_pidx"])
A.trial("E04", "+DART 자산성장 배분(파일럿 E14 재도전, 시점판 가중)", assets=True, feats=f() + ["gd_assets_f"])
A.trial("E05", "특허 KSIC 라벨 신뢰도 가중 w·n_k/(n_k+2)", conf_c=2)
A.trial("E06", "+내수·수출 출하지수 성장(2자리/C26 3자리)", feats=f() + ["g_dom", "g_exp"])
A.trial("E07", "+한국 HS6 수출 t년 성장(Comtrade, KSIC-HS6 연계, 공통품목)", feats=f() + ["g_x_hs"])
A.trial("E08", "+하반기 출하지수 성장 g_idx_h2(월별 경로)", feats=f() + ["g_idx_h2"])
A.trial("E09", "배분 기반 기업 상한 cap=0.3(대기업 번들 완화)", cap=0.3)
A.trial("E10", "산업 잔차 수축 고정효과 ind_bias(λ=5)", ind_bias=5)
print("champion now", A.champ_id, A.champ)
