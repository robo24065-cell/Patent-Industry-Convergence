# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5\m5_dashboard.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""map5 5단계: 9개 트랙 미래 산출물(future_*) + 기술 지형도 → 50 KSIC '산업 대시보드' 초안.
트랙 파일은 읽기만. 사용 파일(R4 권고: 헤드라인 전망은 공식 성적 모형으로):
  T3 R3\future_forecast_r3_ksic.csv (E40-auditD, t0=2024 공개 → 2025–27) / T1 R3\future_forecast_r3.csv (R3E29, 진입 2025–28)
  N9 future_ksic_r6.csv (과제연도 2024) / N6 future_ksic_r5.csv (t0=2023 → 2024–25 출원) / N7 future_ksic_entry_r5.csv (2023–24 과제)
  N8 future_ksic_r6.csv (t0=2024 → 2025–26) / T4 R3\future_forecast_r3.csv (X26, L=2025-07-01, 공식 성적은 R2 E03 → 각주)
  N4 future_ksic_risk_r4.csv (해석층, 본문에서 내림) / N5는 26429 한 산업만이라 열에서 뺌.
모든 미래 값은 순위·상대 비교용(확률 보정 경고: N9 약 14% 과대, N8 2배 과대). 합성 지수는 만들지 않는다(가중을 사전 고정하지 않았으므로).
"""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = _C19_OUT
MC = os.path.abspath(os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5", r"..\..\.."))
J = lambda *p: os.path.join(MC, *p)
rd = lambda p, **k: pd.read_csv(J(*p.split("/")), encoding="utf-8-sig", **k)

prof = pd.read_csv(os.path.join(OUT, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")
q = pd.read_csv(os.path.join(OUT, "m4_quadrant_table.csv"), dtype={"ksic": str}).set_index("ksic")
D = pd.DataFrame(index=prof.index)
D["name"] = prof.name
# 지형도 열
D["kr_rta_P3"] = prof.rta_KR_P3; D["kr_rta_P4_posthoc"] = prof.rta_KR_P4
D["leader_P3"] = prof.leader_P3; D["kr_gap_log_P3"] = prof.kr_gap_log_P3
D["kr_cagr_P1P3"] = prof.cagr_KR_P1P3; D["kr_rel_growth_P1P3"] = prof.kr_rel_growth_P1P3
D["kr_origin_abroad_rta_P3"] = prof.kr_origin_abroad_rta_P3
D["quad_tech_export"] = q.quad_tech_export_P2; D["quad_momentum"] = q.quad_momentum
# 트랙 미래
t3 = rd("T3_기술셀부상예측/R3/future_forecast_r3_ksic.csv", dtype={"ksic": str}).set_index("ksic")
D["T3_breakout_cell_rate"] = t3.pred_rate; D["T3_n_cells"] = t3.n_cells
t1 = rd("T1_산업대기실/R3/future_forecast_r3.csv", dtype={"ksic": str}).set_index("ksic")
D["T1_waiting_room_index"] = t1.waiting_room_index; D["T1_index_share"] = t1.index_share; D["T1_mean_prob"] = t1.mean_prob
n9 = rd("N9_정부RnD과제_특허성과예측/future_ksic_r6.csv", dtype={"top_ksic": str})
n9 = n9[n9.t0 == n9.t0.max()].set_index("top_ksic")
D["N9_exp_pos_projects"] = n9.exp_pos; D["N9_exp_pos_rate"] = n9.exp_pos / n9.n_proj; D["N9_n_proj"] = n9.n_proj
n6 = rd("N6_정부RnD기업_첫특허전환/future_ksic_r5.csv", dtype={"ksic_text": str}).set_index("ksic_text")
D["N6_top10_share"] = n6.n_top10 / n6.n_firms; D["N6_n_firms"] = n6.n_firms
n7 = rd("N7_특허기업_정부RnD첫수행/future_ksic_entry_r5.csv", dtype={"ksic": str}).set_index("ksic")
D["N7_top10_share"] = n7.top10pct_share
n8 = rd("N8_특허분쟁표적레이더/future_ksic_r6.csv", dtype={"ksic": str}).set_index("ksic")
D["N8_exp_disputes_per1000"] = n8.exp_disputes / n8.n * 1000; D["N8_exp_disputes"] = n8.exp_disputes
t4 = rd("T4_특허포기예측/R3/future_forecast_r3.csv", dtype={"ksic": str})
t4 = t4[t4.L == t4.L.max()].set_index("ksic")
D["T4_lapse_risk_3y"] = t4.pred_risk; D["T4_L"] = t4.L
n4 = rd("N4_특허행태_기업폐업조기경보/future_ksic_risk_r4.csv", dtype={"ksic": str}).set_index("ksic")
D["N4_exit_risk_interp"] = n4.exp_risk
IND = ["T3_breakout_cell_rate", "T1_index_share", "N9_exp_pos_rate", "N6_top10_share", "N7_top10_share",
       "N8_exp_disputes_per1000", "T4_lapse_risk_3y", "N4_exit_risk_interp"]
for c in IND:
    D[c + "_pct"] = D[c].rank(pct=True)
D.to_csv(os.path.join(OUT, "m5_industry_dashboard.csv"), encoding="utf-8-sig")
meta = pd.DataFrame([
    ("T3_breakout_cell_rate", "T3 기술셀 부상", "KSIC 안 IPC 메인그룹 셀 중 2025–27 KR 공개 성장 상위 10% 예측 셀 비율", "R3 E40-auditD(공식 성적 모형)", "본문 주축"),
    ("T1_index_share", "T1 산업 대기실", "2025–28 첫 진입 예측 기업 기대 수 / 위험쌍(index_share)", "R3E29(공식)", "본문 주축"),
    ("N9_exp_pos_rate", "N9 정부R&D 성과", "과제연도 2024 과제 중 50-KSIC KR 특허 성과 기대 비율(확률 약 14% 과대)", "R6 공식 모형", "본문 주축"),
    ("N6_top10_share", "N6 R&D기업 첫 특허", "t0=2023 위험 기업 중 상위 10% 후보 비중(과제 텍스트 대응 KSIC)", "R5 공식 모형, 해석층 권고", "본문 주축"),
    ("N7_top10_share", "N7 특허기업 R&D 첫 수행", "특허 KSIC 가중 상위 10% 비중", "R5 공식 모형", "박스"),
    ("N8_exp_disputes_per1000", "N8 분쟁 표적", "살아 있는 권리 1,000건당 2025–26 기대 분쟁(확률 2배 과대)", "R6 공식 모형", "박스"),
    ("T4_lapse_risk_3y", "T4 특허 포기", "L=2025-07-01 살아 있는 등록특허의 3년 내 포기 위험", "R3 X26(공식 성적은 R2 E03 → 각주)", "박스"),
    ("N4_exit_risk_interp", "N4 기업 폐업", "특허기업 퇴조 위험 순위 평균(해석층)", "R4 C12", "본문에서 내림(해석층 열)"),
], columns=["column", "track", "definition", "model", "report_role"])
meta.to_csv(os.path.join(OUT, "m5_dashboard_columns.csv"), index=False, encoding="utf-8-sig")
# 지표 간 상관(서로 겹치지 않는 신호인지) + 지형도와의 관계
cols = ["kr_rta_P3", "kr_rel_growth_P1P3", "kr_origin_abroad_rta_P3"] + IND
C = D[cols].corr(method="spearman")
C.to_csv(os.path.join(OUT, "m5_indicator_spearman.csv"), encoding="utf-8-sig")
tri = C.loc[IND, IND].where(np.triu(np.ones((len(IND), len(IND))), 1).astype(bool)).stack()
res = {"n_ksic": int(len(D)), "coverage": {c: int(D[c].notna().sum()) for c in IND},
       "track_pair_spearman_range": [round(tri.min(), 3), round(tri.max(), 3)],
       "track_pair_abs_gt_0.5": tri[tri.abs() > 0.5].round(3).to_dict().__repr__(),
       "rta_vs_tracks": C.loc["kr_rta_P3", IND].round(3).to_dict(),
       "relgrowth_vs_tracks": C.loc["kr_rel_growth_P1P3", IND].round(3).to_dict()}
# 상위 10개(각 지표) 목록
res["top5"] = {c: [f"{k} {D.at[k, 'name']}" for k in D[c].nlargest(5).index] for c in IND}
# 조합 신호: 기술 강·수출 약 사분면 × T3 부상 비율 상위 1/3
m = (D.quad_tech_export == "기술 강·수출 약(괴리)") & (D.T3_breakout_cell_rate_pct >= 2 / 3)
res["tech_strong_export_weak_and_T3_top_third"] = [f"{k} {D.at[k, 'name']}" for k in D.index[m]]
m2 = (D.kr_gap_log_P3 < np.log(0.7)) & (D.T1_index_share_pct >= 2 / 3)
res["kr_gap_large_and_T1_top_third"] = [f"{k} {D.at[k, 'name']} (선도 {D.at[k, 'leader_P3']})" for k in D.index[m2]]
json.dump(res, open(os.path.join(OUT, "m5_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
