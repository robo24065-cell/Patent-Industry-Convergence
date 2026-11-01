# -*- coding: utf-8 -*-
"""숫자 고정표(facts.csv) 생성기. 결과 파일을 읽기 전용으로 열어 값을 그대로 옮긴다.
원자료·트랙 폴더에는 쓰지 않는다. 출력: 이 폴더의 facts.csv, facts_mismatch.csv
실행: python -X utf8 -I build_facts.py
"""
import csv, json, os, sqlite3, statistics
from decimal import Decimal, ROUND_HALF_UP

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
R = BASE + r"\분석\모델경쟁_20260927"
OUT = os.path.dirname(os.path.abspath(__file__))

def P(rel):
    return os.path.join(R, rel)

def S(rel):  # 출처 표기(프로젝트 루트 기준 상대경로)
    return "분석\\모델경쟁_20260927\\" + rel.replace("/", "\\")

def jl(rel):
    with open(P(rel), encoding="utf-8") as f:
        return json.load(f)

def rows(rel):
    with open(P(rel), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def r(x, nd=3):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def pc(x, nd=1):  # 비율 → %
    return r(Decimal(str(x)) * 100, nd)

def sg(x, nd=3):  # 부호 포함
    s = r(x, nd)
    return s if s.startswith("-") else "+" + s

def spc(x, nd=1):
    return sg(Decimal(str(x)) * 100, nd)

def ci(lo, hi, nd=3, pct=False, sign=False):
    f = (lambda v: (spc(v, nd) if sign else pc(v, nd))) if pct else (lambda v: (sg(v, nd) if sign else r(v, nd)))
    return f"[{f(lo)}, {f(hi)}]"

FACTS = []
MISMATCH = []
IDS = set()

import re
def F(fid, topic, val, unit, phrase, src, cond="", note=""):
    fid = fid.replace(" ", "_")
    phrase = re.sub(r"(^|(?<=[\s\[\(,~:→]))-(?=\d)", "−", phrase)
    assert fid not in IDS, fid
    IDS.add(fid)
    FACTS.append([fid, topic, val, unit, phrase, src, cond, note])
    if note and ("불일치" in note or "다름" in note or "차이" in note):
        MISMATCH.append([fid, note])

# ---------------------------------------------------------------- 1. 공식 성적
R6 = "R6_결과_20261004.md"
GEN = "공식 성적은 첫 잠긴 최종시험 1회 값. 트랙끼리 AUC·skill로 순위 매기지 않음. AUC는 기준선과 나란히, skill은 보수적 수치와 함께 쓴다"

# T1
t1 = jl("T1_산업대기실/R3/final_result_r3.json")
t1rob = "T1_산업대기실/VERIFY_R5/robustness_table.md"
t1s = S("T1_산업대기실/R3/final_result_r3.json")
b = t1["bootstrap_firm_cluster"]["skill_vs_strongest__model"]["auc_all"]
F("OFF-T1-WIN", "공식성적·T1 산업 대기실", "2019→2020–23", "원점→결과창", "원점 c=2019 위험쌍의 2020–23년 첫 진입(R3E29)", t1s, GEN)
F("OFF-T1-N", "공식성적·T1", f"{t1['n']['rows']:,}", "위험쌍", f"위험쌍 {t1['n']['rows']:,}개, 양성 {t1['n']['pos']:,}개(기저율 {pc(t1['n']['pos']/t1['n']['rows'],3)}%)", t1s)
F("OFF-T1-AUC", "공식성적·T1", r(t1["model"]["auc_all"]), "AUC", f"AUC {r(t1['model']['auc_all'])} (기준선 {r(t1['baseline_strongest']['auc_all'])})", t1s, "기준선=공식 6변형 중 최강(xgb_y_u)")
F("OFF-T1-SKILL", "공식성적·T1", r(t1["skill"]["auc_all"]), "skill", f"skill {r(t1['skill']['auc_all'])} {ci(b[0], b[2])}", t1s, "CI=법인 군집 부트스트랩. \"최종이 개발보다 크게 올랐다\" 금지(개발 대비 +0.126은 기준선 효과)")
tk = t1["topk"]
F("OFF-T1-TOP1000", "공식성적·T1", pc(tk["model"]["1000"]["precision"]), "%", f"상위 1,000쌍 적중 {pc(tk['model']['1000']['precision'])}%(기준선 {pc(tk['strongest_auc_baseline']['1000']['precision'])}%, 무작위 0.56%)", t1s, "쉬운 말: 1,000개 중 313개 진입, 무작위면 약 6개")
F("OFF-T1-TOP1PCT", "공식성적·T1", "24.0", "% 포착", "상위 1%가 전체 진입의 24.0% 포착(기준선 10.7%)", S(t1rob), "robustness_table.md 표값(기계 판독 JSON 미확인)")
F("OFF-T1-SBB", "보수적수치·T1", "0.395", "skill", "고정 엄격 기준선 SBB(0.835) 대비 skill 0.395 [0.379, 0.412]", S(t1rob), "검증 등급 '견고'")

# T3
t3 = jl("T3_기술셀부상예측/R3/final_result_r3.json")
t3s = S("T3_기술셀부상예측/R3/final_result_r3.json")
t3rob = S("T3_기술셀부상예측/VERIFY_R5/robustness_table.md")
sm = t3["summary"]; c = t3["ci95"]
F("OFF-T3-WIN", "공식성적·T3 기술셀 부상", "2021→2022–24", "원점→결과창", "공개연도 t0=2021 → KR 공개 2022–24(E40-auditD)", t3s, GEN)
F("OFF-T3-N", "공식성적·T3", "5,041", "셀", f"{t3['by_origin']['2021']['model']['n']:,}셀 중 부상 {t3['by_origin']['2021']['model']['n_evt']}셀", t3s)
F("OFF-T3-AUC", "공식성적·T3", r(sm["auc"]), "AUC", f"AUC {r(sm['auc'])} (기준선 {r(sm['auc_baseline'])})", t3s, "기준선=kr_mom 단독 로지스틱")
F("OFF-T3-SKILL", "공식성적·T3", r(sm["mean_skill"]), "skill", f"skill {r(sm['mean_skill'])} {ci(*c['skill'])}", t3s, "CI=IPC 메인그룹 군집 1,000회")
F("OFF-T3-TOP10", "공식성적·T3", pc(sm["p_at10"]), "%", f"상위 10% 셀의 {pc(sm['p_at10'])}% 부상(기준선 {pc(sm['p_at10_baseline'])}%, 무작위 10%)", t3s,
  "", "R6 문서는 기준선을 23.8–26.0%로 범위 표기. 공식 JSON 기준선 값은 26.0%(문서 범위의 상단과 같음, 하단 23.8%는 robustness 다른 기준선)")
F("OFF-T3-TOP100", "공식성적·T3", "50", "%", "상위 100셀 적중 50%(기준선 33%)", t3rob)
F("OFF-T3-SB3", "보수적수치·T3", "0.182", "skill", "KR 이력 XGB 기준선(0.748) 대비 skill 0.182 [0.124, 0.243]", t3rob, "검증 등급 '견고'")
F("OFF-T3-NEWIPC", "보수적수치·T3", "0.406", "skill", "2022년 이후 신설 IPC 셀 제외 시 skill 0.406", t3rob)
t3b = jl("T3_기술셀부상예측/R2/final_result.json")
F("OFF-T3APP", "공식성적·T3 출원연도판", r(t3b["summary"]["mean_skill"]), "skill",
  f"출원연도판(R2 t0=2019·2020) AUC {r(t3b['summary']['auc'])} (기준선 {r(t3b['summary']['auc_baseline'])}), skill {r(t3b['summary']['mean_skill'])} {ci(*t3b['ci95']['skill'])}, 상위 10% {pc(t3b['summary']['p_at10'])}% / {pc(t3b['summary']['p_at10_baseline'])}%",
  S("T3_기술셀부상예측/R2/final_result.json"), "실시간성 경고: 피처 창 KR 문서 약 46%가 원점 시점 미공개. 본문 대표값은 공개연도판",
  "R6·robustness 문서는 상위 10% 37.1% / 24.8%로 표기. 공식 JSON p_at10 0.3715(→37.2%), 기준선 0.2541(→25.4%)로 다름 → 파일 값 사용(기준선 정의 차이 가능)")

# T4
t4 = jl("T4_특허포기예측/R2/final_result.json")
t4s = S("T4_특허포기예측/R2/final_result.json"); t4rob = S("T4_특허포기예측/VERIFY_R5/robustness_table.md")
F("OFF-T4-WIN", "공식성적·T4 특허 포기", "2022-01→3년", "기준일→결과창", "기준일 L=2022-01-01 → 3년 내 등록료 불납 포기(R2-E03)", t4s, GEN)
F("OFF-T4-N", "공식성적·T4", f"{t4['n']:,}", "건", f"KR 등록특허 {t4['n']:,}건, 포기 기저율 {pc(t4['base_rate'])}%", t4s)
F("OFF-T4-AUC", "공식성적·T4", r(t4["model"]["auc"]), "AUC", f"AUC {r(t4['model']['auc'])} (기준선 {r(t4['baseline']['auc'])})", t4s)
F("OFF-T4-SKILL", "공식성적·T4", r(t4["skill_auc"]), "skill", f"skill {r(t4['skill_auc'])} {ci(*t4['bootstrap_95ci']['skill'])}", t4s, "CI=1순위 출원인 군집")
F("OFF-T4-TOP1PCT", "공식성적·T4", "45.5", "%", "상위 1%(2,875건) 적중 45.5%(기준선 38.9%, 기저율 18.8%)", t4rob)
F("OFF-T4-TOP100", "공식성적·T4", "84", "건/100", "포기 위험 상위 100건 중 84건이 3년 안에 포기(평균 19%)", t4rob)
F("OFF-T4-STRONG", "보수적수치·T4", "0.124", "skill", "13변수 로지스틱 강화 기준선(0.659) 대비 skill 0.124 [−0.003, 0.208]", t4rob, "조건부(CI 하한 0 접촉)")

# N4
n4 = jl("N4_특허행태_기업폐업조기경보/final_result_r4.json")["results"][0]
n4s = S("N4_특허행태_기업폐업조기경보/final_result_r4.json"); n4rob = S("N4_특허행태_기업폐업조기경보/VERIFY_R5/robustness_table.md")
F("OFF-N4-WIN", "공식성적·N4 기업 폐업", "2023→2024–25", "원점→결과창", "t0=2023 → 국세청 폐업 2024–25(C12)", n4s, GEN + ". N4는 본문 제외(트렌드 행렬 해석층 열), 개별 기업명 금지")
F("OFF-N4-AUC", "공식성적·N4", r(n4["auc"]["champ"]), "AUC", f"AUC {r(n4['auc']['champ'])} (M0 {r(n4['auc']['M0_xgb'])})", n4s)
F("OFF-N4-SKILL", "공식성적·N4", r(n4["skill_vs_M0"]), "skill", f"skill {r(n4['skill_vs_M0'])} {ci(*n4['skill_vs_M0_ci95'])}", n4s, "CI=사업자번호 군집 1,000회")
F("OFF-N4-TOP1PCT", "공식성적·N4", pc(n4["top1pct_precision"]["champ"]), "%", f"상위 1%(248곳) 적중 {pc(n4['top1pct_precision']['champ'])}%(M0 {pc(n4['top1pct_precision']['M0'])}%, 기저율 {pc(n4['base_rate'],2)}%)", n4s, "NPS 보정 패널 31.0% / 19.0%")
F("OFF-N4-ALIAS", "보수적수치·N4", "0.140", "skill", "NPS 별칭 보정 skill 0.140 [0.111, 0.168](공식 0.177 → −0.037)", n4rob, "하향 보정 후 유지")
F("OFF-N4-PATINC", "보수적수치·N4", "0.052", "skill", "외부데이터 전부 대비 특허 순증분 0.052 [0.034, 0.069]", n4rob, "공식 0.177의 약 70%는 외부데이터(NPS 탈퇴 이력) 몫")

# N5
n5 = jl("N5_표준특허선언레이더/final_result_r4.json")
n5s = S("N5_표준특허선언레이더/final_result_r4.json"); n5rob = S("N5_표준특허선언레이더/VERIFY_R5/robustness_table.md")
F("OFF-N5-WIN", "공식성적·N5 표준특허 선언", "2022→2023–24", "원점→결과창", "t0=2022 → ETSI 반영 2023–24(E21, KSIC 26429)", n5s, GEN + ". N5는 본문 제외")
F("OFF-N5-AUC", "공식성적·N5", r(n5["auc"]["champ"]), "AUC", f"AUC {r(n5['auc']['champ'])} (M0 {r(n5['auc']['M0'])})", n5s)
F("OFF-N5-SKILL", "공식성적·N5", r(n5["skill_champ_vs_M0"]), "skill", f"skill {r(n5['skill_champ_vs_M0'])} {ci(*n5['ci95_cluster_app'])}", n5s, "CI 하한이 0에 가장 가까운 트랙")
F("OFF-N5-TOP100", "공식성적·N5", str(n5["top100_hits"]["champ"]), "건/100", f"상위 100건 중 {n5['top100_hits']['champ']}건 ETSI 반영(M0 {n5['top100_hits']['M0']}건, 기저율 {pc(n5['base_rate'],2)}%)", n5s)
F("OFF-N5-STRONG", "보수적수치·N5", "0.000", "skill", "M0+최근 선언률 4개 기준선 대비 skill 0.000 [−0.072, 0.070]", n5rob, "조건부(외부데이터 기여 0). 상위 100도 기준선 57 대 챔피언 50", "robustness 원값 0.0002")

# N6
n6 = jl("N6_정부RnD기업_첫특허전환/final_result_r5.json")
n6s = S("N6_정부RnD기업_첫특허전환/final_result_r5.json")
F("OFF-N6-WIN", "공식성적·N6 R&D기업 첫 특허", "2021→2022–23", "원점→결과창", "t0=2021 → 50개 산업 KR 첫 출원 2022–23(C11)", n6s, GEN)
F("OFF-N6-AUC", "공식성적·N6", r(n6["auc"]), "AUC", f"AUC {r(n6['auc'])} (M0 {r(n6['auc_M0'])})", n6s)
F("OFF-N6-SKILL", "공식성적·N6", r(n6["skill"]), "skill", f"skill {r(n6['skill'])} {ci(*n6['skill_ci95'])}", n6s, "학습 원점 NTIS 상호 시점 미점검")
F("OFF-N6-TOP1PCT", "공식성적·N6", pc(n6["top1"]), "%", f"상위 1%(123곳)의 {pc(n6['top1'])}%가 2년 안에 첫 특허(M0 {pc(n6['top1_M0'])}%, 기저율 {pc(n6['base_rate'],2)}%)", n6s)
F("OFF-N6-ENS", "보수적수치·N6", r(n6["secondary_vs_M0_ensemble"]["skill"]), "skill", f"M0-ens({r(n6['secondary_vs_M0_ensemble']['auc_ref'])}) 대비 skill {r(n6['secondary_vs_M0_ensemble']['skill'])} {ci(*n6['secondary_vs_M0_ensemble']['skill_ci95'])}", n6s, "견고")

# N7
n7 = jl("N7_특허기업_정부RnD첫수행/final_result_r5.json")["results"][0]
n7s = S("N7_특허기업_정부RnD첫수행/final_result_r5.json"); n7rob = S("N7_특허기업_정부RnD첫수행/VERIFY_R6/robustness_table.md")
F("OFF-N7-WIN", "공식성적·N7 특허기업 R&D 첫 수행", "2022→2023–24", "원점→결과창", "t0=2022 → NTIS 과제연도 2023–24 첫 수행(R03)", n7s, GEN)
F("OFF-N7-AUC", "공식성적·N7", r(n7["auc"]["champ"]), "AUC", f"AUC {r(n7['auc']['champ'])} (M0 {r(n7['auc']['M0'])})", n7s)
F("OFF-N7-SKILL", "공식성적·N7", r(n7["skill_vs_M0"]), "skill", f"skill {r(n7['skill_vs_M0'])} {ci(*n7['skill_vs_M0_ci95'])}", n7s, "공식값과 NPS 보정값(0.251)을 함께 인용")
tk7 = n7["topk"]
F("OFF-N7-TOP1PCT", "공식성적·N7", pc(tk7["champ"]["top1pct"]["precision"]), "%", f"상위 1%(123곳) {pc(tk7['champ']['top1pct']['precision'])}%(M0 {pc(tk7['M0']['top1pct']['precision'])}%, 기저율 {pc(n7['base_rate'],2)}%), 상위 10% 재현율 {pc(tk7['champ']['top10pct']['recall'])}%", n7s,
  "상위 1% 이득은 특허 고유 기여 아님(외부데이터 전부 모형 21 대 21, 보정 21 대 18). 보정 패널 14.6%(18/123) / M0 9.8%")
F("OFF-N7-ALIAS", "보수적수치·N7", "0.251", "skill", "NPS 별칭 보정 skill 0.251 [0.180, 0.324](공식 0.286 → −0.035)", n7rob, "견고(하향 보정 후)")
F("OFF-N7-SB", "보수적수치·N7", "0.176", "skill", "보정 후 SB 대비 0.176 [0.108, 0.238], 외부데이터 전부 대비(특허 순증분) 0.124 [0.079, 0.167]", n7rob)
F("OFF-N7-NONNPS", "공식성적·N7 경고", "0.789", "AUC", "NPS 미가입 구간 AUC 0.79 [0.73, 0.85](M0 0.49)", n7rob, "'NPS 미가입 AUC 0.806' 문구 사용 금지(오분류로 부풀려짐)", f"공식 JSON nonnps.auc_champ {r(n7['nonnps']['auc_champ'])}는 보정 전 값")

# N8
n8 = jl("N8_특허분쟁표적레이더/final_result_r6.json")
n8s = S("N8_특허분쟁표적레이더/final_result_r6.json")
F("OFF-N8-WIN", "공식성적·N8 특허 분쟁 표적", "2022→2023–24", "원점→결과창", "t0=2022 → 무효·권리범위확인심판 예고등록 2023–24(E01)", n8s, GEN + ". 개별 기업·특허명 금지(KSIC 집계만)")
F("OFF-N8-N", "공식성적·N8", f"{n8['n']:,}", "건", f"시험 {n8['n']:,}건, 양성 {n8['pos']}건(기저율 {r(n8['base_rate_permil']/10,3)}%)", n8s, "양성 108건이라 CI가 넓다")
F("OFF-N8-AUC", "공식성적·N8", r(n8["champ_auc"]), "AUC", f"AUC {r(n8['champ_auc'])} (M0 {r(n8['M0_auc'])})", n8s, "M0만으로 0.78이므로 AUC 상당 부분은 기준선 수준 정보")
F("OFF-N8-SKILL", "공식성적·N8", r(n8["skill_vs_M0"]), "skill", f"skill {r(n8['skill_vs_M0'])} {ci(*n8['skill_vs_M0_ci'])}", n8s, "CI=권리자 군집 46,370개 1,000회")
F("OFF-N8-CAP1PCT", "공식성적·N8", pc(n8["champ_top1pct_capture"]), "% 포착", f"감시 목록을 1/100({n8['k_top1pct']:,}건)로 줄여도 2년 내 분쟁의 {pc(n8['champ_top1pct_capture'])}%({n8['champ_top1pct_hits']}/{n8['pos']}건) 포착(M0 {pc(n8['M0_top1pct_capture'])}%, SB {pc(n8['SB_top1pct_capture'])}%)", n8s,
  "\"분쟁될 특허를 맞힌다\" 금지. 맨 위 목록은 SB와 거의 같음(39 대 33)")
F("OFF-N8-PREC1PCT", "공식성적·N8", pc(n8["champ_top1pct_prec"], 2), "%", f"상위 1% 적중률 {pc(n8['champ_top1pct_prec'],2)}%(기저율 {r(n8['base_rate_permil']/10,3)}%의 36배)", n8s)
F("OFF-N8-TOP100", "공식성적·N8", str(n8["champ_hit100"]), "건/100", f"상위 100건 {n8['champ_hit100']}건(M0 {n8['M0_hit100']}, SB {n8['SB_hit100']})", n8s)
F("OFF-N8-CAP510", "공식성적·N8", pc(n8["champ_top5pct_capture"]), "% 포착", f"상위 5% 포착 {pc(n8['champ_top5pct_capture'])}%(M0 {pc(n8['M0_top5pct_capture'])}%, SB {pc(n8['SB_top5pct_capture'])}%), 상위 10% {pc(n8['champ_top10pct_capture'])}%({pc(n8['M0_top10pct_capture'])}%, {pc(n8['SB_top10pct_capture'])}%)", n8s)
F("OFF-N8-SB", "보수적수치·N8", r(n8["skill_vs_SB"]), "skill", f"SB(M0+과거 분쟁, AUC {r(n8['SB_auc'])}) 대비 skill {r(n8['skill_vs_SB'])} {ci(*n8['skill_vs_SB_ci'])}", n8s, "\"최종에서 SB 대비 더 좋아졌다(0.270→0.312)\" 금지")
sub = n8["subgroups"]["신규(이력 없음)"]
F("OFF-N8-NEW", "보수적수치·N8", r(sub["skill_vs_SB"]), "skill", f"신규 분쟁(양성 {sub['pos']}) SB 대비 {r(sub['skill_vs_SB'])} {ci(*sub['skill_vs_SB_ci'])}", n8s)
F("OFF-N8-CALIB", "공식성적·N8 경고", "2.04", "배", "확률 2.04배 과대(예측 합 220.3 대 실제 108)", S("R6_결과_20261004.md"), "확률을 '예상 분쟁 건수'로 읽지 않음", "R6 문서 값(FINAL_REPORT_R6 근거), JSON 미확인")

# N9
n9 = jl("N9_정부RnD과제_특허성과예측/final_result_r6.json")
n9s = S("N9_정부RnD과제_특허성과예측/final_result_r6.json")
F("OFF-N9-WIN", "공식성적·N9 R&D 과제 특허 성과", "2022→2022–23", "원점→결과창", "t=2022 과제 → 50개 산업 KR 특허 출원 2022–23(C10, 보고 지연판)", n9s, GEN + ". 지연판을 주 모형으로 한 것은 사후 결정")
F("OFF-N9-N", "공식성적·N9", f"{n9['n']:,}", "과제", f"과제 {n9['n']:,}건, 양성 {n9['pos']:,}({pc(n9['base_rate'],2)}%)", n9s)
F("OFF-N9-AUC", "공식성적·N9", r(n9["auc"]), "AUC", f"AUC {r(n9['auc'])} (M0_L2 {r(n9['auc_M0L2'])})", n9s, "\"N9 AUC 0.89\" 단독 금지")
F("OFF-N9-SKILL", "공식성적·N9", r(n9["skill"]), "skill", f"skill {r(n9['skill'])} {ci(*n9['skill_ci95'])}", n9s, "CI=수행기관 군집 16,239개. final_audit_r6.json의 [0.359, 0.418]은 공식값 아님")
F("OFF-N9-TOP1PCT", "공식성적·N9", pc(n9["top1"]), "%", f"상위 1%(751건)의 {pc(n9['top1'])}%가 특허 성과(M0_L2 {pc(n9['top1_M0L2'])}%, 기저율 {pc(n9['base_rate'],1)}%)", n9s, "최상위 우위 주장 금지: 1년 더 쓰는 기준선(파일럿 원판) 상위 1% 80.6%")
F("OFF-N9-TOP100", "공식성적·N9", str(n9["hit100"]), "건/100", f"상위 100건 중 {n9['hit100']}건(M0_L2 {n9['hit100_M0L2']})", n9s, "파일럿 원판 96건, 강한 원판 97건(R6 계산)이 더 높음")
F("OFF-N9-TOP510", "공식성적·N9", pc(n9["top10"]), "%", f"상위 5% 적중 {pc(n9['top5'])}%(M0_L2 {pc(n9['top5_M0L2'])}%), 상위 10% {pc(n9['top10'])}%({pc(n9['top10_M0L2'])}%), 상위 10% 재현율 {pc(n9['recall_top10'])}%({pc(n9['recall_top10_M0L2'])}%)", n9s)
F("OFF-N9-STRONG", "보수적수치·N9", r(n9["skill_vs_strong_L2"]), "skill", f"강한 기준선 지연판({r(n9['auc_strong_L2'])}) 대비 {r(n9['skill_vs_strong_L2'])} {ci(*n9['skill_vs_strong_L2_ci95'])}", n9s, "견고(같은 정보 시점)")
F("OFF-N9-PILOT", "보수적수치·N9", r(n9["skill_vs_pilot_orig"]), "skill", f"파일럿 원판({r(n9['auc_pilot_orig'])}, 정보 1년 더) 대비 {r(n9['skill_vs_pilot_orig'])}", n9s)
sg9 = n9["subgroups"]
F("OFF-N9-SUB", "공식성적·N9 하위집단", r(sg9["대학"]["skill"]), "skill", f"대학 {r(sg9['대학']['skill'])}, 출연연 {r(sg9['출연연·국공립연']['skill'])}, 기업 {r(sg9['기업']['skill'])}, 신규 수행기관 {r(sg9['신규수행기관']['skill'])}", n9s, "CI는 FINAL_REPORT_R6.md 표: 대학 [0.447, 0.512], 기업 [0.221, 0.292], 신규 [0.078, 0.314]. \"기업 과제·신규 기관에서도 잘 맞는다\" 금지")
F("OFF-DEV2FINAL", "공식성적·개발 대비 최종", "T3 −0.005 … N9 −0.014", "skill 차", "개발 대비 최종: T3 −0.005, T1 +0.126(기준선 효과), N4 −0.018, N5 −0.029, N6 −0.027, N7 +0.069, N8 −0.014, N9 −0.014", S(R6), "9개 트랙 모두 공식 CI 하한 > 0, 잠글 구간 소진", "R6 문서 값")

# ---------------------------------------------------------------- 2. 라벨 재현성
t0 = jl("T1_산업대기실/TXT_R9/res/t0_txt_agree.json")
t0b = jl("T1_산업대기실/TXT_R9/res/t0b_txtfree_agree.json")
lv = rows("_shared_report/label_validity/lv1_overall_agreement.csv")
cc = jl("_shared_report/audit_r7/concordance_resolution_check.json")
F("LBL-CW-GROUP", "라벨 재현성", pc(t0["group_agree"]), "%", f"공식 IPC-KSIC 연계표와 그룹 수준 일치 {pc(t0['group_agree'])}%(κ {r(float(lv[0]['kappa']))})", S("T1_산업대기실/TXT_R9/res/t0_txt_agree.json"), "본문 표준값(R7 §3.4). KR 출원, 27그룹 라벨공간")
F("LBL-CW-5D", "라벨 재현성", pc(t0["cw5_agree_all(16개만 지목)"]), "%", f"5자리 일치 {pc(t0['cw5_agree_all(16개만 지목)'])}%(연계표가 지목하는 16개 산업만)", S("T1_산업대기실/TXT_R9/res/t0_txt_agree.json"), "분모 KR 출원 1,087,511(미배정=불일치)",
  f"R7·label_validity는 5자리 {pc(float(lv[5]['일치율']))}%(분모 배정된 1,039,164건). 분모 차이로 값이 다름. R9·R10 본문 표준은 7.1%")
F("LBL-CW-ASSIGN", "라벨 재현성", pc(t0["cw_assigned"]), "%", f"연계표가 KR 출원의 {pc(t0['cw_assigned'])}%에 배정", S("T1_산업대기실/TXT_R9/res/t0_txt_agree.json"))
F("LBL-CW-27", "라벨 재현성", str(cc["n_concordance_codes_covering_50"]), "코드", f"제공 50개 세세분류가 연계표 {cc['n_concordance_codes_covering_50']}개 코드로 뭉침, {50-cc['ksic_alone_in_code']}/50은 서로 구분 불가(단독 {cc['ksic_alone_in_code']}개)", S("_shared_report/audit_r7/concordance_resolution_check.json"), "예: 메모리·비메모리·LED·기타 반도체가 모두 C2610")
F("LBL-CW-MULTI", "라벨 재현성", pc(cc["kr_2005_2022"]["share_ipc_multi_industry"]), "%", f"연계표에 있는 출원의 {pc(cc['kr_2005_2022']['share_ipc_multi_industry'])}%는 IPC가 2개 이상 산업에 걸침(다대다), 메인 IPC 포함률 {pc(cc['kr_2005_2022']['matched_ipc'])}%", S("_shared_report/audit_r7/concordance_resolution_check.json"), "KR 출원 2005–2022 842,412건")
F("LBL-CW-TOPAGREE", "라벨 재현성", pc(cc["kr_2005_2022"]["top_agree_at_concordance_level"]), "%", f"연계표 최빈 산업이 제공 라벨 상위 코드와 일치 {pc(cc['kr_2005_2022']['top_agree_at_concordance_level'])}%", S("_shared_report/audit_r7/concordance_resolution_check.json"),
  "R7 대조표 값. TXT-FREE 5자리 40.4%와 숫자가 우연히 같지만 다른 지표 — 혼동 주의", "값 같음·정의 다름(대조표 40.4%=연계표 최빈 일치, R9 40.4%=TXT-FREE 5자리)")
F("LBL-SINGLE", "라벨 재현성", "1.0", "라벨/출원", "제공 데이터에서 KR 출원 1건이 갖는 KSIC 라벨은 평균 1.0(복수 라벨 0%)", S("_shared_report/audit_r7/concordance_resolution_check.json"), "배타적 단일 라벨 → 비율 지표 분모 정의")
F("LBL-TXT", "라벨 재현성", pc(t0["txt5_agree_all(미배정=불일치)"]), "%", f"해설서 텍스트 배정(TXT, 연계표 그룹 안) 5자리 일치 {pc(t0['txt5_agree_all(미배정=불일치)'])}%", S("T1_산업대기실/TXT_R9/res/t0_txt_agree.json"), "본문 표준 '29~40%'의 하단. T3 32.3%, R8 29.6%는 모집단·어휘창 차이")
F("LBL-FREE", "라벨 재현성", pc(t0b["free5_agree_all"]), "%", f"해설서 텍스트 단독(TXT-FREE, 공개 최강) 5자리 일치 {pc(t0b['free5_agree_all'])}%, 그룹 {pc(t0b['free_group_agree_all'])}%", S("T1_산업대기실/TXT_R9/res/t0b_txtfree_agree.json"), "\"공개 자료로는 5자리 배정 불가\" 금지. c5_R10 KR 출원 2015–19는 39.2%(모집단 차이)")
F("LBL-TXT-INGROUP", "라벨 재현성", pc(t0["txt5_agree_given_group_correct_multi34"]), "%", f"그룹이 맞은 경우 TXT 5자리 일치 {pc(t0['txt5_agree_given_group_correct_multi34'])}%(무작위 {pc(t0['random_within_group_expect'])}%)", S("T1_산업대기실/TXT_R9/res/t0_txt_agree.json"), "", "R8 문서 68.3%는 R8 원판 TXT 값(누출 제거판과 98% 같음)")
lc = rows("_shared_report/label_validity/lv1_by_confidence.csv")
F("LBL-CONF", "라벨 재현성", "15.0→90.0", "%", "연계표 확신도 구간별 일치 " + " → ".join(pc(float(x["일치율"])) + "%" for x in lc), S("_shared_report/label_validity/lv1_by_confidence.csv"), "두 분류는 같은 대상을 재고 IPC가 모호한 곳에서 갈린다")

# ---------------------------------------------------------------- 3. 근거 A (기업 업종)
s19 = rows("T1_산업대기실/TXT_R9/res/s1_firm_o2019.csv")
s15 = rows("T1_산업대기실/TXT_R9/res/s1_firm_o2015.csv")
def pick(rs, base, cond_prefix, w="기업평균"):
    for x in rs:
        if x["기준"] == base and x["조건"].startswith(cond_prefix) and x["가중"] == w:
            return x
    raise KeyError((base, cond_prefix, w))
s1src = S("T1_산업대기실/TXT_R9/res/s1_firm_o2019.csv")
for base, tag in [("NPS", "NPS"), ("DART", "DART")]:
    x = pick(s19, base, "(d)")
    F(f"A-{tag}-RATES", f"근거A·기업 5자리 업종 일치({tag})", pc(x["P"]), "%",
      f"{tag} 등록업종이 50개 안인 기업({int(x['기업수']):,}사) 평균 5자리 일치: 제공 {pc(x['P'])}%, 연계표 {pc(x['CW'])}%, TXT {pc(x['TXT'])}%, TXT-FREE {pc(x['FREE'])}%", s1src,
      "원점 2019, 기업 평균. NPS 업종=원점 시점(2019-12), DART=2026 스냅샷")
    F(f"A-{tag}-PFREE", f"근거A({tag})", spc(x["P-FREE"]), "%p", f"제공 − TXT-FREE {spc(x['P-FREE'])}%p {ci(x['P-FREE_lo'], x['P-FREE_hi'], 1, True, True)}", s1src, "TXT-FREE 비교는 사후 추가 분석(R9 표기)")
    F(f"A-{tag}-PTXT", f"근거A({tag})", spc(x["P-TXT"]), "%p", f"제공 − TXT {spc(x['P-TXT'])}%p {ci(x['P-TXT_lo'], x['P-TXT_hi'], 1, True, True)}", s1src)
    F(f"A-{tag}-PCW", f"근거A({tag})", spc(x["P-CW"]), "%p", f"제공 − 연계표 {spc(x['P-CW'])}%p {ci(x['P-CW_lo'], x['P-CW_hi'], 1, True, True)}", s1src)
    y = pick(s15, base, "(d)")
    F(f"A-{tag}-O2015", f"근거A({tag}) 원점 2015", pc(y["P"]), "%", f"원점 2015: 제공 {pc(y['P'])}%, TXT {pc(y['TXT'])}%, FREE {pc(y['FREE'])}% (제공−TXT {spc(y['P-TXT'])}%p, 제공−FREE {spc(y['P-FREE'])}%p, CI 하한 > 0)", S("T1_산업대기실/TXT_R9/res/s1_firm_o2015.csv"))
    z = pick(s19, base, "(d2)")
    F(f"A-{tag}-SMALL", f"근거A({tag}) 특허 1~20건 기업", spc(z["P-TXT"]), "%p", f"특허 1~20건 기업만: 제공−TXT {spc(z['P-TXT'])}%p", s1src)
    e = pick(s19, base, "(e)")
    F(f"A-{tag}-34", f"근거A({tag}) 34개 업종", spc(e["P-TXT"]), "%p", f"연계표 그룹 공유 34개 업종 기업: 제공−TXT {spc(e['P-TXT'])}%p, 제공−FREE {spc(e['P-FREE'])}%p (CI > 0)", s1src)
xf = pick(s19, "NPS", "(f) ")
F("A-NPS-INGROUP", "근거A 단서(NPS 그룹 안 선택)", spc(xf["P-TXT"]), "%p", f"NPS 그룹 안 5자리 선택력({xf['기업수']}사): 제공 {pc(xf['P'])}% 대 TXT {pc(xf['TXT'])}%, 차이 {spc(xf['P-TXT'])}%p {ci(xf['P-TXT_lo'], xf['P-TXT_hi'],1,True,True)}", s1src, "NPS 기준 우위는 주로 산업군을 맞히는 데서 나옴. \"세부 업종까지 해설서보다 잘 고른다\" 금지(DART 기준이라고 밝히면 가능)")
xf2 = pick(s19, "NPS", "(f2)", "특허가중")
F("A-NPS-INGROUP-PW", "근거A 단서(특허 가중, TXT 우세)", spc(xf2["P-TXT"]), "%p", f"네 그룹 모두 맞은 특허, 특허 가중: 제공−TXT {spc(xf2['P-TXT'])}%p {ci(xf2['P-TXT_lo'], xf2['P-TXT_hi'],1,True,True)}", s1src, "불리한 결과(TXT 우세)")
xd = pick(s19, "DART", "(f) ")
F("A-DART-INGROUP", "근거A(DART 그룹 안 선택)", spc(xd["P-TXT"]), "%p", f"DART 그룹 안 5자리 선택력({xd['기업수']}사): 제공 {pc(xd['P'])}% 대 TXT {pc(xd['TXT'])}%, {spc(xd['P-TXT'])}%p {ci(xd['P-TXT_lo'], xd['P-TXT_hi'],1,True,True)}", s1src, "", "R9 문서 CI [+12.4, +20.0]와 파일 CI가 다름 → 파일 값 사용")
xp = pick(s19, "NPS", "(d)", "특허가중")
F("A-NPS-PW", "근거A 단서(특허 가중)", spc(xp["P-TXT"]), "%p", f"NPS 5자리 50개 안 특허 가중: 제공 {pc(xp['P'])}% 대 TXT {pc(xp['TXT'])}%, {spc(xp['P-TXT'])}%p {ci(xp['P-TXT_lo'], xp['P-TXT_hi'],1,True,True)}", s1src, "CI가 0 포함(대기업에서 차이 작음)")
xk = pick(s19, "KIND", "(b)")
F("A-KIND", "근거A(KIND 3자리)", spc(xk["P-FREE"]), "%p", f"KIND 상장사({xk['기업수']}사) 3자리: 제공 {pc(xk['P'])}%, 연계표 {pc(xk['CW'])}%, FREE {pc(xk['FREE'])}%; 제공−연계표 {spc(xk['P-CW'])}%p {ci(xk['P-CW_lo'],xk['P-CW_hi'],1,True,True)}, 제공−FREE {spc(xk['P-FREE'])}%p {ci(xk['P-FREE_lo'],xk['P-FREE_hi'],1,True,True)}", s1src,
  "R7 KIND(521곳, +4.2%p)와 표본이 다름. 연계표 대비 본문 표준=R7 값, TXT·FREE 대비=이 값")
meta = jl("T1_산업대기실/TXT_R9/res/s1_meta_o2019.json")
F("A-COVER", "근거A 단서(대상 범위)", pc(meta["NPS"]["share_5digit_in50"]), "%", f"주 조건이 다루는 기업은 특허 보유 법인 {meta['NPS']['firms_with_patents']:,}사 중 NPS {pc(meta['NPS']['share_5digit_in50'])}%, DART {pc(meta['DART']['share_5digit_in50'])}%", S("T1_산업대기실/TXT_R9/res/s1_meta_o2019.json"), "본문 각주")
# R7 label_validity
lv2 = rows("_shared_report/label_validity/lv2_firm_agreement_o2019.csv")
def lvp(base, ind):
    for x in lv2:
        if x["기준"] == base and x["지표"] == ind: return x
for base in ["NPS", "DART"]:
    x = lvp(base, "연계표그룹(27) 기업평균")
    F(f"A-R7-{base}", f"근거A·R7 연계표 대비 그룹 일치({base})", spc(x["차이(제공-연계표)"]), "%p", f"{base} 등록업종(27그룹, 기업 평균 {int(x['기업수']):,}사): 제공 {pc(x['제공라벨_일치율'])}% 대 연계표 {pc(x['연계표_일치율'])}%, {spc(x['차이(제공-연계표)'])}%p {ci(x['차이_95CI_lo'],x['차이_95CI_hi'],1,True,True)}", S("_shared_report/label_validity/lv2_firm_agreement_o2019.csv"), "원점 2019, 연계표 대비만")
    y = lvp(base, "5자리 특허가중(등록업종 5자리 기업)")
    F(f"A-R7-{base}-5D", f"근거A·R7 5자리 특허가중({base})", spc(y["차이(제공-연계표)"]), "%p", f"5자리 특허 가중 {base}: 제공 {pc(y['제공라벨_일치율'])}% 대 연계표 {pc(y['연계표_일치율'])}%", S("_shared_report/label_validity/lv2_firm_agreement_o2019.csv"))
kind = jl("_ideas_R7/validity_kind_industry.json")
F("A-R7-KIND", "근거A·R7 KIND", "+4.2", "%p", f"KIND 상장법인 3자리(521곳): 제공 {pc(kind['소분류 3자리 일치 — 제공 KSIC 라벨'])}% 대 연계표 {pc(kind['소분류 3자리 일치 — 공식 연계표'])}%, 차이 CI {ci(*kind['소분류 3자리 일치율 차이(제공−연계표) 95% CI'],1,True,True)}", S("_ideas_R7/validity_kind_industry.json"), "", "파일 값 25.5%−21.3%=4.2%p로 문서와 일치")
evf = jl("N4_특허행태_기업폐업조기경보/ESSENTIAL_R7/ext_validity_firm.json")
F("A-N4-16PP", "근거A·사용 금지 값", "16", "%p", f"N4 파일의 '16%p 더 잘 맞힌다'(DART {pc(evf[0]['top_unit_match_provided'])}% 대 {pc(evf[0]['top_unit_match_crosswalk'])}%, NPS {pc(evf[1]['top_unit_match_provided'])}% 대 {pc(evf[1]['top_unit_match_crosswalk'])}%)", S("N4_특허행태_기업폐업조기경보/ESSENTIAL_R7/ext_validity_firm.json"), "본문 사용 금지(국민연금 2024-12 업종, 50 KSIC 대응 기업만 → 제공 라벨에 유리). 쓰려면 조건 명시")

# ---------------------------------------------------------------- 4. 순환 점검 (R10)
LO = "_shared_report/label_origin_R10/"
p1 = rows(LO + "res_p1.csv"); p1d = rows(LO + "res_p1_disagree.csv")
def p1r(base, fam="가족 다수결", w="기업평균"):
    for x in p1:
        if x["기준"] == base and x["패밀리"] == fam and x["가중"] == w: return x
fc = jl(LO + "res_fam_check.json"); tests = jl(LO + "res_tests.json")
lk = {x[0]: x for x in fc["links"]}
F("CIRC-LINKS", "순환점검·패밀리 연결", f"{lk['US'][1]:,}/{lk['JP'][1]:,}/{lk['EP'][1]:,}", "쌍", f"등록번호로 다시 만든 패밀리 연결: US {lk['US'][1]:,}, JP {lk['JP'][1]:,}, EP {lk['EP'][1]:,}쌍", S(LO + "res_fam_check.json"), "R8 fam_link.parquet(공개번호를 출원번호로 오독)는 결함 — 그 기반 숫자 인용 금지")
F("CIRC-AGREE-CC", "순환점검·같은 발명 라벨 일치", f"{pc(lk['US'][4])}/{pc(lk['JP'][4])}/{pc(lk['EP'][4])}", "%", f"같은 발명의 KR·해외 라벨 일치(문서쌍): US {pc(lk['US'][4])}%, JP {pc(lk['JP'][4])}%, EP {pc(lk['EP'][4])}%", S(LO + "res_fam_check.json"))
F("CIRC-AGREE-KR", "순환점검·같은 발명 라벨 일치", pc(tests["P0"]["agree_kr_vs_famMajor"]), "%", f"KR 문서 기준 해외 다수결 일치 {pc(tests['P0']['agree_kr_vs_famMajor'])}%(1:1 연결 {pc(tests['P0']['agree_kr_vs_fam_one2one'])}%, 패밀리 보유 KR {tests['P0']['kr_with_family']:,}건)", S(LO + "res_tests.json"), "사전 기준 95% 이하 → 해외 비교 '정보 있음'. 1:1에서도 10%가 다름(단순 복사 아님)", "원값 0.8965(반올림 89.7%)")
old = {x[0]: x for x in fc["old_r8_method"]}
F("CIRC-R8DEFECT", "순환점검·결함 발견", f"{pc(old['US'][2])}/{pc(old['EP'][2])}/{pc(old['JP'][2])}", "%", f"R8 결함 연결의 라벨 일치율 US {pc(old['US'][2])}%, EP {pc(old['EP'][2])}%, JP {pc(old['JP'][2])}%(CN {pc(old['CN'][2])}%는 정상)", S(LO + "res_fam_check.json"), "결함 연결 기반 숫자(res_p0.json, c1_stats family_pkr_agree)는 패밀리 일치로 인용 금지")
for base in ["NPS", "DART"]:
    x = p1r(base)
    F(f"CIRC-P1-{base}", f"순환점검·P1 {base}", spc(x["KR-해외"], 2), "%p", f"{base} 기업 평균({x['기업']}사, {int(x['특허']):,}건): KR 라벨 {pc(x['KR라벨'],2)}%, 해외 라벨 {pc(x['해외라벨'],2)}%, 차이 {spc(x['KR-해외'],2)}%p {ci(x['KR-해외_lo'],x['KR-해외_hi'],2,True,True)}", S(LO + "res_p1.csv"), "판정 문구는 \"순환 가능성은 낮다\"(단정형 \"순환이 아니다\" 금지, 사전 규칙상 '불명(경계)')")
    F(f"CIRC-P1-{base}-FREE", f"순환점검·P1 {base}", spc(x["해외-FREE"], 2), "%p", f"해외 라벨 − TXT-FREE(같은 KR 문서) {spc(x['해외-FREE'],2)}%p {ci(x['해외-FREE_lo'],x['해외-FREE_hi'],2,True,True)}", S(LO + "res_p1.csv"), "근거 A의 우위가 해외 라벨에서도 같다")
    y = p1r(base, "가족 1:1만")
    F(f"CIRC-P1-{base}-1to1", f"순환점검·P1 {base} 1:1", spc(y["KR-해외"], 2), "%p", f"1:1 연결만: {spc(y['KR-해외'],2)}%p {ci(y['KR-해외_lo'],y['KR-해외_hi'],2,True,True)}", S(LO + "res_p1.csv"))
pw = p1r("NPS", w="특허가중")
F("CIRC-P1-NPS-PW", "순환점검·P1 경고", spc(pw["KR-해외"], 2), "%p", f"NPS 특허 가중 KR−해외 {spc(pw['KR-해외'],2)}%p {ci(pw['KR-해외_lo'],pw['KR-해외_hi'],2,True,True)}; 해외−FREE {spc(pw['해외-FREE'],2)}%p {ci(pw['해외-FREE_lo'],pw['해외-FREE_hi'],2,True,True)}", S(LO + "res_p1.csv"), "CI가 점추정에 비대칭(점추정이 CI 밖 근처). 본문은 기업 평균을 주로 쓴다")
for x in p1d:
    if x["패밀리"] != "가족 다수결": continue
    F(f"CIRC-P1-R-{x['기준']}", f"순환점검·P1 불일치 사례({x['기준']})", r(float(x["r=KR/해외"]), 2), "비", f"불일치 사례 r=P(KR=등록)/P(해외=등록) {x['기준']} {r(float(x['r=KR/해외']),2)} [{r(float(x['r_lo']),2)}, {r(float(x['r_hi']),2)}] ({x['KR=등록']} 대 {x['해외=등록']}건)", S(LO + "res_p1_disagree.csv"),
      "DART r 1.84 > 1.5 → P1 '부분/불명', 종합 '불명(경계)'" if x["기준"] == "DART" else "0.67 ≤ r ≤ 1.5 → 낮음")
ph2 = jl(LO + "res_posthoc2.json")
rn = [x for x in ph2["rows"] if x["US 출원인칸=발명자만"]]
F("CIRC-INVONLY", "순환점검·사후(발명자만 US)", pc(ph2["inventor_only_share"]), "%", f"[사후] 출원인 칸이 발명자뿐인 US 문서({pc(ph2['inventor_only_share'])}%): NPS KR {pc(rn[0]['KR라벨'])}% 대 US {pc(rn[0]['US라벨'])}%, DART {pc(rn[1]['KR라벨'])}% 대 {pc(rn[1]['US라벨'])}%", S(LO + "res_posthoc2.json"), "사후 분석")
p2 = rows(LO + "res_p2.csv")
F("CIRC-P2", "순환점검·P2 출원인 집중도", r(tests["P2_ratio_family"][0]["비율"]), "비", "IPC 통제 후 초과 집중도 E(KR)/E(해외): " + ", ".join(f"{x['층'].split()[0]} {r(x['비율'])}" + (f" [{r(x['lo'])}, {r(x['hi'])}]" if i == 0 else "") for i, x in enumerate(tests["P2_ratio_family"])), S(LO + "res_tests.json"), "≤ 1.2 → 낮음 쪽")
F("CIRC-P2-EH0", "순환점검·P2", r(float(p2[0]["상대초과E/H0"]), 2), "비", f"초과 집중 E/H0 제공 {r(float(p2[0]['상대초과E/H0']),2)}, 순수 내용 라벨 TXT-FREE {r(float(p2[3]['상대초과E/H0']),2)}", S(LO + "res_p2.csv"), "초과 집중은 출원인별 기술 특화로 설명")
p3 = rows(LO + "res_p3.csv")
F("CIRC-P3", "순환점검·P3 다업종 출원인", r(float(p3[0]["b/a"])), "비", f"다업종 출원인 {int(p3[0]['대상출원인']):,}개({int(p3[0]['IPC최빈≠출원인최빈 특허']):,}건): 라벨=IPC 최빈 {pc(p3[0]['a=라벨=IPC최빈'])}%, =출원인 최빈 {pc(p3[0]['b=라벨=출원인최빈'])}%, b/a {r(float(p3[0]['b/a']))} [{p3[0]['b/a_lo']}, {p3[0]['b/a_hi']}] (TXT-FREE {r(float(p3[1]['b/a']))})", S(LO + "res_p3.csv"), "제공 ≤ FREE → 낮음 쪽")
ph = jl(LO + "res_posthoc.json")
a = ph["a_agree_by_ipc"]
F("CIRC-IPC", "순환점검·사후(IPC 차이별 일치)", pc(a["메인IPC서브그룹 같음"][0]), "%", f"[사후] 같은 발명 KR·해외 라벨 일치: 메인 서브그룹 같음 {pc(a['메인IPC서브그룹 같음'][0])}%, 서브클래스만 같음 {pc(a['서브그룹 다름·서브클래스 같음'][0])}%, 서브클래스도 다름 {pc(a['서브클래스도 다름'][0])}%", S(LO + "res_posthoc.json"), "라벨이 문서 내용을 따라간다")
bt = ph["b_agree_by_type"]
F("CIRC-TYPE", "순환점검·사후(출원인 유형)", pc(bt["등록업종 50개 안 법인(KR)"][0]), "%", f"[사후] 등록업종 법인 {pc(bt['등록업종 50개 안 법인(KR)'][0])}% 대 그 밖 KR 법인 {pc(bt['그 밖 KR 법인'][0])}%(차이 CI [{spc(ph['b_diff_registered_minus_other_corp'][0])}, {spc(ph['b_diff_registered_minus_other_corp'][1])}]%p)", S(LO + "res_posthoc.json"), "")
F("CIRC-BOUND", "순환점검·결론", "약 1", "%p", "등록업종 참조로 설명될 수 있는 몫은 많아야 약 1%p(CI 상한)", S(LO + "RESULT_R10.md"), "R10 확정 문장")
F("CIRC-AIHUB", "순환점검·사용 금지 값", "0.934 / 90", "F1 / Top-1 %", "AI Hub 학습자료: KR KoBERT F1 0.934, 해외 다국어 BERT Top-1 90%", S(LO + "RESULT_R10.md"), "\"제공 데이터는 이 모델의 출력\", \"제공 라벨 정확도 93%\" 금지(근거 없음). 한계 각주에서만")

# ---------------------------------------------------------------- 5. 근거 B (T1 라벨 무관 결과)
cs = rows("T1_산업대기실/TXT_R9/res/C_summary.csv")
pool = {x["pair"]: x for x in rows("T1_산업대기실/TXT_R9/res/C_pool_summary.csv")}
def mauc(arm, key="auc"):
    return statistics.mean(float(x[key]) for x in cs if x["arm"] == arm)
csrc = S("T1_산업대기실/TXT_R9/res/C_summary.csv"); psrc = S("T1_산업대기실/TXT_R9/res/C_pool_summary.csv")
F("B-AUCS", "근거B·T1 라벨 무관 결과", r(mauc("P50"), 4), "AUC", f"새 IPC 서브클래스 첫 출원 AUC: 제공 {r(mauc('P50'),4)}, 해설서 단독 {r(mauc('FREE50'),4)}, 연계표+해설서 {r(mauc('TXT50'),4)}, 연계표 {r(mauc('CW27'),4)}, 맥락없음 {r(mauc('NONE'),4)}", csrc, "개발 원점 c 2008–2012 5개 평균")
for pr, lab in [("P50-CW27", "연계표"), ("P50-TXT50", "연계표+해설서(TXT)"), ("P50-FREE50", "해설서 단독(FREE)")]:
    x = pool[pr]
    F(f"B-{pr}", "근거B·T1", sg(float(x["mean_d"]), 4), "ΔAUC", f"제공 − {lab} {sg(float(x['mean_d']),4)} [{sg(float(x['ci_lo']),4)}, {sg(float(x['ci_hi']),4)}], {x['wins']}", psrc,
      "크기 AUC 0.002~0.004로 작다(문장에서 빼지 않음)" + (". FREE 대비 하한 +0.0004로 0에 가깝고 2008 원점 +0.0001" if "FREE" in pr else ""))
for pr, lab in [("P50TXT-P50", "제공에 TXT 더함"), ("P50FREE-P50", "제공에 FREE 더함"), ("P50TXT-TXT50", "TXT에 제공 더함"), ("P50FREE-FREE50", "FREE에 제공 더함")]:
    x = pool[pr]
    F(f"B-{pr}", "근거B·T1 포함 관계", sg(float(x["mean_d"]), 4), "ΔAUC", f"{lab} {sg(float(x['mean_d']),4)} [{sg(float(x['ci_lo']),4)}, {sg(float(x['ci_hi']),4)}], {x['wins']}", psrc, "공개 배정의 정보는 제공 라벨에 거의 포함된다")
gain = jl("T1_산업대기실/NONCIRC_R8/res/A_pool_P50_NONE.json")["mean_d"]
F("B-CTXGAIN", "근거B·T1", sg(gain, 4), "ΔAUC", f"산업 맥락 이득(제공 − 맥락없음) {sg(gain,4)}; 공개 텍스트 재현 몫 TXT {pc(float(pool['TXT50-NONE']['mean_d'])/gain,0)}%, FREE {pc(float(pool['FREE50-NONE']['mean_d'])/gain,0)}%", S("T1_산업대기실/NONCIRC_R8/res/A_pool_P50_NONE.json") + "; " + psrc, "나머지 44% 이상은 제공 라벨에만 있음", "R9 문서 FREE 56%는 반올림한 0.0053으로 나눈 값(차이 반올림)")
F("B-TOP1000", "근거B·T1", pc(mauc("P50", "top1000")), "%", f"top1000 적중 제공 {pc(mauc('P50','top1000'))}% 대 FREE {pc(mauc('FREE50','top1000'))}%", csrc, "비슷하다(상위 목록 우위 주장 금지)")
ap = jl("T1_산업대기실/NONCIRC_R8/res/A_pool_P27_CW27.json")
F("B-R8-P27", "근거B·R8 같은 해상도", sg(ap["mean_d"], 4), "ΔAUC", f"제공27 − 연계표27(같은 해상도) {sg(ap['mean_d'],4)} {ci(*ap['ci95'],4,False,True)}, {ap['wins']}/5", S("T1_산업대기실/NONCIRC_R8/res/A_pool_P27_CW27.json"), "R8 판정 기준은 T3 기준을 사후에 빌림")
c2 = rows("T1_산업대기실/NONCIRC_R8/res/C_summary.csv")
def m2(Y, arm): return statistics.mean(float(x["auc"]) for x in c2 if x["Y"] == Y and x["arm"] == arm)
F("B-R8-2x2", "근거B·R8 대칭 2×2", sg(m2("Y_C", "F_PC") - m2("Y_C", "F_C"), 4), "ΔAUC", f"연계표 정의 결과에 제공 피처 더하면 {sg(m2('Y_C','F_PC')-m2('Y_C','F_C'),4)}, 제공 정의 결과에 연계표 피처 더하면 {sg(m2('Y_P','F_PC')-m2('Y_P','F_P'),4)}", S("T1_산업대기실/NONCIRC_R8/res/C_summary.csv"), "연계표 정보는 제공 라벨에 거의 포함(T1 비대칭, T3는 대칭)")

# ---------------------------------------------------------------- 6. 근거 C (R10)
C5 = "_shared_report/fusion_advantage/c5_R10/"
c1 = rows(C5 + "c1_label_agreement.csv")
win = {x["cc"]: x for x in c1 if x["window"].startswith("출원2015")}
allc = {x["cc"]: x for x in c1 if x["window"] == "전체"}
for k in ["US", "JP", "CN", "EP", "KR"]:
    x = win[k]
    F(f"C-AGREE-{k}", f"근거C·해외 라벨 재현율({k})", pc(x["free5"]), "%", f"{k} 출원 2015–19 발명특허({int(x['n']):,}건): 연계표 그룹 {pc(x['cw_group'])}%, 연계표 5자리 {pc(x['cw5'])}%, TXT {pc(x['txt5'])}%, FREE-IPC {pc(x['free5'])}%, 진단용 PKR-IPC {pc(x['pkr5'])}%", S(C5 + "c1_label_agreement.csv"),
      "KR은 FREE 텍스트 단독 값" if k == "KR" else "FREE-IPC=공개 최강. PKR-IPC는 비공개 진단용(판정 제외)")
F("C-AGREE-RANGE", "근거C·해외 라벨 재현율", "48–54", "%", f"해외 4개국 제공 라벨 재현율: 연계표 5자리 {pc(min(float(win[k]['cw5']) for k in 'US JP CN EP'.split()))}–{pc(max(float(win[k]['cw5']) for k in 'US JP CN EP'.split()))}%, 텍스트+IPC 표 {pc(min(float(win[k]['txt5']) for k in 'US JP CN EP'.split()))}–{pc(max(float(win[k]['txt5']) for k in 'US JP CN EP'.split()))}%, 해설서 단독+IPC 표 {pc(min(float(win[k]['free5']) for k in 'US JP CN EP'.split()))}–{pc(max(float(win[k]['free5']) for k in 'US JP CN EP'.split()))}%", S(C5 + "c1_label_agreement.csv"),
  "\"해외 특허 KSIC 라벨은 공개 자료로 재현되지 않는다\"(일반형) 금지 → \"절반 정도만 재현\". 불리한 점: 해외가 KR(39%)보다 더 잘 재현됨")
F("C-AGREE-UB", "근거C·결함 상한 보정", "52.3/49.3/55.2", "%", "결함 패밀리 연결 상한 보정 FREE-IPC: US 52.3%, JP 49.3%, EP 55.2%(JP 여전히 50% 미만)", S("R10_결과_20261005.md"), "", "R10 문서 계산값(alt_assign 읽기 집계, 결과 파일 없음)")
c2r = jl(C5 + "c2_results.json")
mm = c2r["main_abroadRTA_P3_vs_RCA"]
F("C-RCA-PROV", "근거C·외부 기준(해외 출원 RTA↔수출 RCA)", r(mm["free"]["rho_prov"]), "ρ", f"우선권 KR 해외 출원 RTA(2017–19) ↔ 수출 RCA(2015–17) 제공 ρ {r(mm['free']['rho_prov'])}(재현 {r(c2r['repro_rho_abroad_P3'],4)}, n 48)", S(C5 + "c2_results.json"),
  "\"0.545는 제공 라벨이어야 나온다\" 금지. '5개국 데이터가 있으면 나오는 지표'로만",
  f"제공 ρ CI가 비교마다 다름(부트스트랩): {ci(*mm['cweq']['ci_prov'])}, {ci(*mm['txt']['ci_prov'])}, {ci(*mm['free']['ci_prov'])}; R10 문서 [0.28, 0.74], map5 m4_summary [0.276, 0.732]")
for k, lab in [("cweq", "CW-eq"), ("txt", "TXT"), ("free", "FREE-IPC")]:
    x = mm[k]
    F(f"C-RCA-{lab}", "근거C·외부 기준", r(x["rho_alt"]), "ρ", f"{lab} ρ {r(x['rho_alt'])} {ci(*x['ci_alt'])}, 제공−대체 {sg(x['d'])} {ci(*x['ci_d'],3,False,True)} → {x['verdict']}", S(C5 + "c2_results.json"), "모두 '같다'. n=48 검정력 낮음")
F("C-RCA-PKR", "근거C·외부 기준(진단)", r(mm["pkr"]["rho_alt"]), "ρ", f"PKR-IPC(진단) ρ {r(mm['pkr']['rho_alt'])}, 차이 {sg(mm['pkr']['d'])}", S(C5 + "c2_results.json"), "비공개 진단용")
sb = c2r["sub_abroadRTA_P2_vs_RCA"]
F("C-RCA-SUB", "근거C·외부 기준 보조창", sg(sb["free"]["d"]), "Δρ", f"보조 창(2015–17) 제공−대체: CW-eq {sg(sb['cweq']['d'])}, TXT {sg(sb['txt']['d'])}, FREE {sg(sb['free']['d'])}(모두 CI 0 포함)", S(C5 + "c2_results.json"))
ls = c2r["landscape_P3"]
for k, lab in [("prov", "제공"), ("cweq_F", "CW-eq F"), ("txt_F", "TXT F"), ("free_F", "FREE-IPC F"), ("cweq_H", "KR제공+해외CW-eq H"), ("txt_H", "KR제공+해외TXT H"), ("free_H", "KR제공+해외FREE H"), ("pkr_H", "KR제공+해외PKR H(진단)")]:
    x = ls[k]
    extra = "" if k == "prov" else f", 제공 1위 7개 중 유지 {x['kr_leader_kept_of_prov7']}, 한국 RTA ρ(제공 대비) {r(x['rho_vs_prov']['KR'])}"
    F(f"C-LAND-{k}", "근거C·지형도 라벨 민감도", str(x["kr_leader"]), "산업", f"{lab}: 한국 RTA>1 {x['kr_rta_gt1']}개, 한국 1위 {x['kr_leader']}개{extra}", S(C5 + "c2_results.json"),
      "출원 2017–19 동일가중 RTA. 어느 지형도가 맞는지는 판정 안 됨. \"한국 7개 산업 1위\"는 라벨 민감성 각주 필수")
sp = c2r["sub_patRTA_P2_vs_RCA"]
F("C-KRRTA-RCA", "근거C·한국 특허 RTA↔RCA", r(sp["cweq_F"]["rho_prov"]), "ρ", f"한국 특허 RTA(2015–17) ↔ 수출 RCA 제공 {r(sp['cweq_F']['rho_prov'])}, 공개 F 시나리오 {r(min(sp[k]['rho_alt'] for k in ['cweq_F','txt_F','free_F']))}–{r(max(sp[k]['rho_alt'] for k in ['cweq_F','txt_F','free_F']))}(CI 0 포함), CW-eq H 차이 {sg(sp['cweq_H']['d'])} {ci(*sp['cweq_H']['ci_d'],3,False,True)}", S(C5 + "c2_results.json"),
  "F 시나리오 차이 CI 모두 0 포함", f"제공 CI도 비교마다 다름(예 {ci(*sp['cweq_F']['ci_prov'])}); R10 문서 [0.04, 0.62]")
F("C-ABROAD-N", "근거C·지표 분자", f"{c2r['abroad_kr_docs_P3']['prov']:,}", "건", f"우선권 KR 해외 출원 2017–19 {c2r['abroad_kr_docs_P3']['prov']:,}건", S(C5 + "c2_results.json"), "FREE-IPC 팔은 이 중 8.9%(5,002건)가 결함 연결로 배정(R10 §6-1 문서 계산, 결과 파일 없음)")
t3c = jl(C5 + "c5_t3_summary.json")
A_ = t3c["auc"]; Pp = t3c["pairs"]
F("C-T3-AUCS", "근거C·T3 예측 기여", r(A_["drop_ros"], 4), "AUC", f"T3(개발 원점 9개, RoS 제외판) AUC: 제공 {r(A_['drop_ros'],4)}, KR만 {r(A_['kr_rebuild'],4)}, hyb_txt {r(A_['hyb_txt'],4)}, hyb_free {r(A_['hyb_free'],4)}, hyb_pkr(진단) {r(A_['hyb_pkr'],4)}", S(C5 + "c5_t3_summary.json"), "개발 원점만, 공식 성적 아님")
for k, lab in [("drop_ros−hyb_txt", "제공 − hyb_txt"), ("drop_ros−hyb_free", "제공 − hyb_free"), ("drop_ros−hyb_pkr", "제공 − hyb_pkr(진단)"), ("hyb_txt−kr_rebuild", "hyb_txt − KR만"), ("hyb_free−kr_rebuild", "hyb_free − KR만"), ("drop_ros−kr_rebuild", "해외 기여(제공 − KR만)")]:
    x = Pp[k]
    F("C-T3-" + k.replace("−", "-"), "근거C·T3", sg(x["dAUC"], 4), "ΔAUC", f"{lab} {sg(x['dAUC'],4)} {ci(*x['ci'],4,False,True)}, {x['wins']}", S(C5 + "c5_t3_summary.json"),
      "체계 일관성의 몫(과제를 제공 라벨로 정의). \"5개국 없으면 −0.035라 대체 불가\" 금지" if "drop_ros−hyb" in k else "")
ret = t3c["retention"]
F("C-T3-RET", "근거C·T3", f"{pc(ret['hyb_txt'],0)}/{pc(ret['hyb_free'],0)}", "% 보존", f"해외 기여 보존율 hyb_txt {pc(ret['hyb_txt'],0)}%, hyb_free {pc(ret['hyb_free'],0)}%(공개 라벨로 바꾸면 해외 신호의 71–75% 상실), hyb_pkr {pc(ret['hyb_pkr'],0)}%", S(C5 + "c5_t3_summary.json"), "")
F("C-T3-COVD", "근거C·T3 셀 구성 진단", r(t3c["hyb_txt_coverage_D"]["alt_vs_target_totW_median_ratio"], 2), "중앙값 비", f"제공 셀 총가중 대비 대체 캐시 중앙값 비 hyb_txt {r(t3c['hyb_txt_coverage_D']['alt_vs_target_totW_median_ratio'],2)}, hyb_free {r(t3c['hyb_free_coverage_D']['alt_vs_target_totW_median_ratio'],2)}, hyb_pkr {r(t3c['hyb_pkr_coverage_D']['alt_vs_target_totW_median_ratio'],2)}", S(C5 + "c5_t3_summary.json"), "T3 우위는 라벨 정확도보다 평가 셀 정의 체계와의 일치")
p1c = jl(C5 + "c4_p1_results.json")["compare"]
for k, lab in [("FULL_prov5c-KRONLY_prov", "FULL − KR만(5개국 증분)"), ("FULL_prov5c-HYB_txt", "FULL − HYB_txt"), ("FULL_prov5c-HYB_free", "FULL − HYB_free"), ("HYB_txt-KRONLY_prov", "HYB_txt − KR만"), ("HYB_free-KRONLY_prov", "HYB_free − KR만")]:
    x = p1c[k]
    F("C-P1-" + k, "근거C·R8 P1 기술 이동", sg(x["mean"], 4), "ΔAUC", f"{lab} {sg(x['mean'],4)} {ci(*x['ci'],4 if abs(x['ci'][1])>1e-4 else 5,False,True)}, {x['pos_origins']} ({x['verdict']})", S(C5 + "c4_p1_results.json"),
      "공개 해외 라벨이 오히려 해롭다('사라진다'가 아니라 '나빠진다')" if k == "HYB_txt-KRONLY_prov" else "")
cnt = {k: int(allc[k]["n"]) for k in ["KR", "US", "JP", "CN", "EP"]}
tot = int(allc[""]["n"])
F("C-COUNTRY-SHARE", "데이터 규모·국가 구성", pc(cnt["KR"] / tot), "%", "제공 데이터 행 구성: " + ", ".join(f"{k} {pc(cnt[k]/tot)}%({cnt[k]:,})" for k in ["KR", "CN", "US", "JP", "EP"]), S(C5 + "c1_label_agreement.csv"), "행=(출원, KSIC) 쌍")
F("C-OLD-R7", "근거C·인용 제한(R7 5개국)", "−0.035/−0.016/−0.028", "ΔAUC/skill", "R7: T3 KR만 −0.035(9/9), T4 skill −0.016(4/4), N8 skill −0.028(7/9)", S("R7_결과_20261005.md"),
  "R10에서 그대로 인용 금지. T3는 공개 해외 배정 시 기여 25–29% 회복, T4·N8은 패밀리 필드(라벨 아님)")

F("C-VERDICT", "근거C·종합 판정", "2/5", "판정 항목", "사전 규칙 판정 항목 5개(주 지표 대체안 3, T3 hyb_txt·hyb_free) 중 '제공 우위' 2개(T3 둘) → '부분'. 근거 C는 '조건부(체계 일관성)'", S("R10_결과_20261005.md"), "단독 필수성 근거로 쓰지 않음. 근거 A와 이어서만")
F("A-VERDICT", "근거A·순환 판정", "불명(경계)/낮음", "판정", "사전 규칙 판정 '불명(경계)'(P1 부분/불명, P2·P3 낮음 쪽), 실질 판정 '순환 가능성 낮음'", S("R10_결과_20261005.md"), "\"순환이 아니다\"(단정형) 금지 — RESULT_R10.md 권고 문장이 판정보다 강함", "RESULT_R10.md 권고 문장('순환이 아니다')과 R10 확정 판정('순환 가능성 낮음')이 다름 → R10 문구 사용")
F("PAGE-ALLOC", "보고서 20쪽 배분(R10 §4.3)", "20", "쪽", "요약 1 · 서론·문제 정의 1 · 데이터·연계 프레임워크 2 · 필수성·라벨 외부 타당성 3(A 1.5, B 0.5, C 0.5, 종합표 0.5) · 트렌드 2 · 검증 설계 1.5 · T3 2 · T1 2 · N9·N6 1.5 · 박스(N7 0.5, N8·T4 0.5) 1 · 정책 제언 1.5 · 결론·한계 1 · 출처 0.5", S("R10_결과_20261005.md"), "N4·N5 본문 제외")

# ---------------------------------------------------------------- 7. T3 라벨 무관·외부 정합 (R8·R9)
mg = jl("T3_기술셀부상예측/TXT_R9/summary_mg_r9.json")
mgm = {x["model"]: x for x in mg["models"]}
F("T3NC-MG", "필수성·T3 메인그룹 성장(라벨 무관)", r(mgm["prov"]["auc"], 4), "AUC", f"메인그룹 성장 상위 10% AUC: 제공 {r(mgm['prov']['auc'],4)}, 연계표 {r(mgm['conc']['auc'],4)}, TXT {r(mgm['txt']['auc'],4)}, 맥락없음 {r(mgm['none']['auc'],4)}(기준선 {r(mg['baseline_auc'],4)})", S("T3_기술셀부상예측/TXT_R9/summary_mg_r9.json"), "산업 맥락은 어떤 배정이든 보태는 것이 없다")
for k, lab in [("prov_vs_txt", "제공 − TXT"), ("prov_vs_conc", "제공 − 연계표"), ("txt_vs_none", "TXT − 맥락없음")]:
    x = mg["pairs"][k]
    F("T3NC-" + k, "필수성·T3 메인그룹", sg(x["dAUC"], 4), "ΔAUC", f"{lab} {sg(x['dAUC'],4)} {ci(*x['ci'],4,False,True)}, {x['wins']}/{x['n_orig']} ({x['verdict']})", S("T3_기술셀부상예측/TXT_R9/summary_mg_r9.json"), "\"제공 라벨이 TXT보다 메인그룹 성장 예측을 낫게 한다\" 금지")
ea = rows("T3_기술셀부상예측/TXT_R9/external_alignment_r9.csv")
def ear(cmp_, ind, sig="cell_real"):
    for x in ea:
        if x["비교"] == cmp_ and x["외부지표"] == ind and x["신호"] == sig: return x
for cmp_, lab in [("prov-txt", "TXT"), ("prov-conc", "연계표")]:
    for ind in ["사업체수", "종사자수"]:
        x = ear(cmp_, ind)
        q = float(x["q_BH"])
        F(f"T3EXT-{cmp_}-{ind}", f"조건부·T3 셀 실현율↔이후 {ind}", sg(float(x["차이"])), "Δρ", f"제공 − {lab}: ρ {r(float(x['A_rho']))} 대 {r(float(x['B_rho']))}, 차이 {sg(float(x['차이']))} [{sg(float(x['CI_lo']))}, {sg(float(x['CI_hi']))}], {x['A우세_원점']}/9, BH q {('< 0.001' if q < 0.001 else r(q, 3 if q < 0.2 else 2))}", S("T3_기술셀부상예측/TXT_R9/external_alignment_r9.csv"),
          "BH 후 유의는 연계표 대비 사업체뿐. 필수성 근거로 앞세우지 않음(트렌드 장 기술통계)",
          ("R8 문서 CI [+0.142, +0.431](사업체)·[+0.018, +0.325](종사자)는 다른 난수열 — 평균 같음, 본문은 R8 값 사용 권고(R9 §5-2)" if cmp_ == "prov-conc" else ""))
xi = ear("prov-conc", "수입액", "cell_pred")
F("T3EXT-IMPORT", "조건부·T3 외부 정합", sg(float(xi["차이"])), "Δρ", f"모형 예측 ↔ 수입액 제공−연계표 {sg(float(xi['차이']))}, BH q {r(float(xi['q_BH']))}", S("T3_기술셀부상예측/TXT_R9/external_alignment_r9.csv"), "", "R9 문서 'BH 후 유의'로 적었으나 파일 판정은 '같다'(우세 원점 5/9 < 60%)")
ew = rows("T3_기술셀부상예측/TXT_R9/external_within_group_r9.csv")
x = [y for y in ew if y["외부지표"] == "사업체수" and y["신호"] == "cell_real"][0]
F("T3EXT-34", "조건부·T3 그룹 안 34 KSIC", sg(float(x["차이"])), "Δρ", f"연계표 그룹 안 34 KSIC 사업체(셀 실현율) 제공 {r(float(x['제공_r']))} 대 TXT {r(float(x['TXT_r']))}, {sg(float(x['차이']))} [{sg(float(x['CI_lo']),4)}, {sg(float(x['CI_hi']))}], {x['제공우세_원점']}/9", S("T3_기술셀부상예측/TXT_R9/external_within_group_r9.csv"), "경계")
F("T3NC-LINK", "필수성·연결 수", "50/48", "산업", "KOSIS 5자리 50/50, HS6 48/50 연결(TXT도 50/50, 48/50), 연계표 5자리 지목 16/50", S("R9_결과_20261005.md"), "\"KOSIS 50/50은 제공 데이터 고유 장점\" 금지(연계표 대비라고 밝히면 가능)")
r8s = jl("T3_기술셀부상예측/NONCIRC_R8/summary_r8.json")
cel = r8s["sym2x2"]["cells"]
F("T3NC-2x2", "필수성·R8 T3 대칭 2×2", r(cel["pcode_feat-pcode"]["auc"], 4), "AUC", f"제공 피처→제공 결과 {r(cel['pcode_feat-pcode']['auc'],4)}, 제공 피처→연계표 결과 {r(cel['pcode_feat-conc']['auc'],4)}, 연계표 피처→연계표 결과 {r(cel['conc_feat-conc']['auc'],4)}", S("T3_기술셀부상예측/NONCIRC_R8/summary_r8.json"), "두 결과 평균 +0.0089 [−0.0002, +0.0176] 7/9 → 같다(경계, R8 문서)")
et = rows("T3_기술셀부상예측/ESSENTIAL_R7/essential_table_pub.csv")
etd = {x["test"]: x for x in et}
F("R7-T3-CONC", "필수성·R7 T3 연계표 대체(순환)", f"{r(float(etd['full']['mean_auc']),3)}→{r(float(etd['conc_feat']['mean_auc']),3)}", "AUC", f"연계표 대체(결과 고정) {r(float(etd['full']['mean_auc']),3)}→{r(float(etd['conc_feat']['mean_auc']),3)}({etd['conc_feat']['d_auc']} {etd['conc_feat']['d_auc_ci95']}, {etd['conc_feat']['auc_lt_full']})", S("T3_기술셀부상예측/ESSENTIAL_R7/essential_table_pub.csv"),
  "순환 단서 없이 필수성 근거로 인용 금지(R8: 대부분 순환 효과)")
F("R7-T3-KRONLY", "필수성·R7 T3 KR만", etd["kr_rebuild"]["d_auc"], "ΔAUC", f"KR만 재계산 {r(float(etd['full']['mean_auc']),3)}→{r(float(etd['kr_rebuild']['mean_auc']),3)}({etd['kr_rebuild']['d_auc']} {etd['kr_rebuild']['d_auc_ci95']}, 9/9)", S("T3_기술셀부상예측/ESSENTIAL_R7/essential_table_pub.csv"), "단독 인용 금지(근거 C 단서)")
F("R7-T3-CONCTASK", "필수성·R7 T3 재정의", etd["conc_task"]["mean_auc"], "AUC", f"연계표로 과제 재정의 시 AUC {etd['conc_task']['mean_auc']}(기준선 {etd['conc_task']['mean_base_auc']}, skill {etd['conc_task']['mean_skill']}); 부상 셀 일치 13.5%", S("T3_기술셀부상예측/ESSENTIAL_R7/essential_table_pub.csv"), "부상 셀 일치 13.5%는 R7 문서 값")

# ---------------------------------------------------------------- 8. N9·N6 순환 없는 시험 / R7
NC = "_shared_report/noncirc_n9n6_R9/"
rt = rows(NC + "results_table.csv"); at = rows(NC + "auc_table.csv")
def rtr(s, c_):
    for x in rt:
        if x["set"] == s and x["comparison"] == c_: return x
def atr(s, a_):
    for x in at:
        if x["set"] == s and x["arm"] == a_: return x
for s_, tag, lab in [("n9_y_all", "N9", "분야 무관 KR 특허 성과"), ("n6_ntis_y_ntis", "N6", "NTIS 수행기업 첫 NTIS 보고 KR 특허")]:
    F(f"NC-{tag}-AUC", f"필수성·{tag} 라벨 무관 결과", r(float(atr(s_, "PROV")["auc_mean"]), 4), "AUC", f"{lab} AUC: 제공 {r(float(atr(s_,'PROV')['auc_mean']),4)}, 연계표 {r(float(atr(s_,'CW')['auc_mean']),4)}, TXT {r(float(atr(s_,'TXT')['auc_mean']),4)}", S(NC + "auc_table.csv"), "")
    x = rtr(s_, "PROV-TXT")
    F(f"NC-{tag}-PTXT", f"필수성·{tag}", sg(float(x["mean_dAUC"]), 4), "ΔAUC", f"제공 − TXT {sg(float(x['mean_dAUC']),4)} [{sg(float(x['ci_lo']),4)}, {sg(float(x['ci_hi']),4)}], {x['npos']}/{x['n_origins']}", S(NC + "results_table.csv"), "피처 필수성 사라짐. \"N9·N6 피처는 공개 자료로 대체 불가\" 금지")
    x = rtr(s_, "PROV-CW")
    F(f"NC-{tag}-PCW", f"필수성·{tag}", sg(float(x["mean_dAUC"]), 4), "ΔAUC", f"제공 − 연계표 {sg(float(x['mean_dAUC']),4)} [{sg(float(x['ci_lo']),4)}, {sg(float(x['ci_hi']),4)}], {x['npos']}/{x['n_origins']} ({x['verdict']})", S(NC + "results_table.csv"), "")
for s_, tag in [("n9_y_frozen", "N9"), ("n6_prov_y", "N6")]:
    x = rtr(s_, "PROV-TXT")
    F(f"NC-{tag}-FROZEN", f"필수성·{tag} 동결 결과(순환)", sg(float(x["mean_dAUC"]), 4), "ΔAUC", f"동결 결과(제공 라벨 정의)에서 제공 − TXT {sg(float(x['mean_dAUC']),4)}, {x['npos']}/{x['n_origins']}", S(NC + "results_table.csv"), "결과 정의를 바꾸면 사라지는 순환 효과")
ya = rows("N9_정부RnD과제_특허성과예측/ESSENTIAL_R7/yalt_dev.csv")
yal = [float(x["auc"]) for x in ya if x["variant"] == "D_YALT_altfeats"]
sd9 = {x["variant"]: x for x in rows("N9_정부RnD과제_특허성과예측/ESSENTIAL_R7/summary_dev.csv")}
F("R7-N9-YALT", "필수성·R7 N9 결과까지 연계표(순환)", f"{r(float(sd9['FULL']['mean_auc']),4)}→{r(statistics.mean(yal),4)}", "AUC", f"결과까지 연계표로 바꾸면 {r(float(sd9['FULL']['mean_auc']),4)}→{r(statistics.mean(yal),4)}(7/7)", S("N9_정부RnD과제_특허성과예측/ESSENTIAL_R7/yalt_dev.csv"), "순환 단서 없이 필수성 근거로 인용 금지")
sd6 = {x["variant"]: x for x in rows("N6_정부RnD기업_첫특허전환/ESSENTIAL_R7/summary_dev.csv")}
F("R7-N6-YALT", "필수성·R7 N6 결과까지 연계표(순환)", f"{r(float(sd6['FULL']['mean_auc']),4)}→{r(float(sd6['D_YALT_altfeats']['mean_auc']),4)}", "AUC", f"결과까지 연계표로 바꾸면 {r(float(sd6['FULL']['mean_auc']),4)}→{r(float(sd6['D_YALT_altfeats']['mean_auc']),4)}(6/6, skill {r(float(sd6['FULL']['mean_skill']),3)}→{r(float(sd6['D_YALT_altfeats']['mean_skill']),3)})", S("N6_정부RnD기업_첫특허전환/ESSENTIAL_R7/summary_dev.csv"), "순환 단서 없이 인용 금지")
sd1 = {x["variant"]: x for x in rows("T1_산업대기실/ESSENTIAL_R7/summary_dev.csv")}
F("R7-T1-ALT", "필수성·R7 T1 연계표+텍스트 대체", f"{r(float(sd1['FULL']['mean_auc']),4)}→{r(float(sd1['S_TXTIPC']['mean_auc']),4)}", "AUC", f"S_TXTIPC 대체 {r(float(sd1['FULL']['mean_auc']),4)}→{r(float(sd1['S_TXTIPC']['mean_auc']),4)}({sg(float(sd1['S_TXTIPC']['d_auc']),4)}, 5/5)", S("T1_산업대기실/ESSENTIAL_R7/summary_dev.csv"), "결과가 제공 라벨 정의(순환 가능). 순환 없는 근거는 B-* 행")
F("R7-N7", "필수성·R7 N7(외부 결과)", "−0.0102", "skill", "N7 KSIC 4개를 연계표로 대체: skill −0.0102(4/4)", S("N7_특허기업_정부RnD첫수행/ESSENTIAL_R7/essential_summary.md"), "\"연계표 대비 4/4\"로만. 공개 텍스트 대비 미시험")
F("R7-N8T4-FAM", "필수성·R7 N8·T4 5개국(패밀리)", "−0.0284/−0.0161", "skill", "5개국 패밀리 피처 제거: N8 skill −0.0284(7/9), T4 −0.0161(4/4)", S("N8_특허분쟁표적레이더/ESSENTIAL_R7/essential_summary.md") + "; " + S("T4_특허포기예측/ESSENTIAL_R7/essential_summary.md"), "패밀리 정보 기여(출원국 수). \"제공 KSIC 라벨 덕분\" 금지")

# ---------------------------------------------------------------- 9. R8 P1·P2
p1r8 = jl("_ideas_R8/p1_results.json"); fv = jl("_ideas_R8/p1_firm_validity.json"); p2r8 = jl("_ideas_R8/p2_results.json")
s8 = p1r8["summary"]
F("R8-P1-AUC", "R8 P1 기술의 산업 간 이동", r(s8["FULL_prov5c"]["auc"]), "AUC", f"P1 AUC FULL {r(s8['FULL_prov5c']['auc'])}, M0 {r(s8['M0_prov_KR6']['auc'])}, TXT {r(s8['TXT_sub']['auc'])}, CW {r(s8['CW_sub']['auc'])}", S("_ideas_R8/p1_results.json"), "결과를 제공 라벨로 정의(순환). 트랙 미개설, T3 해석층")
F("R8-P1-DART", "R8 P1 외부 점검(DART 도착 산업)", spc(fv["PROV-CW"]["diff"]), "%p", f"도착 산업 DART 일치 제공 {pc(fv['PROV']['dest_match_event_mean'])}%, CW {pc(fv['CW']['dest_match_event_mean'])}%, TXT {pc(fv['TXT']['dest_match_event_mean'])}%; 제공−CW {spc(fv['PROV-CW']['diff'])}%p {ci(*fv['PROV-CW']['ci'],1,True,True)}, 제공−TXT {spc(fv['PROV-TXT']['diff'])}%p", S("_ideas_R8/p1_firm_validity.json"), "TXT 대비 '같다'")
pc2 = p2r8["ship"]["compare"]["sig_PROV_KR-sig_TXT"]
F("R8-P2", "R8 P2 그룹 내 분화→KOSIS(불통과)", sg(pc2["mean"]), "ΔAUC", f"출하 몫 상승 AUC TXT {r(p2r8['ship']['auc_mean']['sig_TXT'])}, 제공 KR {r(p2r8['ship']['auc_mean']['sig_PROV_KR'])}; 제공 KR − TXT {sg(pc2['mean'])} {ci(*pc2['ci'],3,False,True)} ({pc2['pos']})", S("_ideas_R8/p2_results.json"), "TXT 우위(반대 방향). 불리한 결과로 기록")

# ---------------------------------------------------------------- 10. 34개 산업 분화 (split34)
SP = "_shared_report/fusion_advantage/split34/"
vd = {x["지표"]: x for x in rows(SP + "t1_variance_decomposition.csv")}
for k, lab in [("g_KR", "KR 출원 성장"), ("g_KR기원", "한국 기원 출원 성장"), ("한국기원점유_2015_17", "한국 기원 점유(2015–17 수준)"), ("한국기원점유_변화pp", "한국 기원 점유 변화"), ("g_5국", "5개국 합계 성장"), ("CN출원국점유_2015_17", "CN 출원국 점유"), ("x_g_출하액", "KOSIS 출하 성장"), ("x_g_종사자수", "KOSIS 종사자 성장"), ("x_g_사업체수", "KOSIS 사업체 성장")]:
    x = vd[k]
    F(f"SP-VAR-{k}", "트렌드·34개 산업 분화(그룹 안 분산 비중)", pc(x["그룹안_분산비중_34"], 1), "%", f"{lab}: 연계표 그룹 안 분산 비중 {pc(x['그룹안_분산비중_34'])}%(무작위 기대 {pc(x['무작위묶음_기대'])}%, p {x['p(더_동질)']})", S(SP + "t1_variance_decomposition.csv"),
      "34 KSIC·11그룹, 2005–07→2015–17. 연계표 대비로만 유효(TXT 대비 미시험) — 트렌드 장 기술통계로만",
      ("split34 SUMMARY의 '한국 기원 점유 53%'는 2015–17 수준값, R8의 0.628은 성장 — 둘 다 파일 값" if k == "한국기원점유_2015_17" else ""))
tc = rows(SP + "t2c_panel_two_windows.csv")
for x in tc:
    if x["특허"] != "KR출원": continue
    F(f"SP-FE-{x['외부']}", "트렌드·34개 산업 분화(그룹×창 고정효과 상관)", r(float(x["그룹×창 안 r"]), 2), "r", f"KR 출원 성장 ↔ {x['외부']}: 그룹×창 안 r {r(float(x['그룹×창 안 r']),2)}(순열 p {x['순열p(단측)']}, n 68)", S(SP + "t2c_panel_two_windows.csv"),
      "\"제공 라벨이어야 보인다\" 금지(TXT 대비 미시험, P2는 TXT 우위). 다중비교 미보정")
tw = rows(SP + "t2_within_group_external.csv")
xs = [y for y in tw if y["특허지표"] == "gw_KR_lag0" and y["외부지표"] == "x_g_사업체수"][0]
xe = [y for y in tw if y["특허지표"] == "gw_KR_lag0" and y["외부지표"] == "x_g_종사자수"][0]
F("SP-LONG", "트렌드·34개 산업 분화(단일 장기 창)", r(float(xs["그룹안_r"])), "r", f"단일 장기 창(n 34) KR 출원 ↔ 사업체 r {r(float(xs['그룹안_r']))} {xs['r_95CI_그룹부트']}, 종사자 {r(float(xe['그룹안_r']))} {xe['r_95CI_그룹부트']}(순열 p ≈ 0.09)", S(SP + "t2_within_group_external.csv"), "")
t5m = jl(SP + "t5_meta.json")
F("SP-PAIRS", "트렌드·34개 산업 분화(쌍 순서 일치)", t5m["쌍순서일치_종사자"]["일치"], "쌍", f"같은 그룹 쌍 순서 일치: 종사자 {t5m['쌍순서일치_종사자']['일치']}, 사업체 {t5m['쌍순서일치_사업체']['일치']}", S(SP + "t5_meta.json"), "쌍 비독립 → 이항 p(0.0018/0.048)는 참고값")
cases = {x["ksic"]: x for x in rows(SP + "t5_cases.csv")}
for k in ["26211", "26212", "26291", "29280", "26111"]:
    x = cases[k]
    F(f"SP-CASE-{k}", "트렌드·분화 사례(같은 창 08–09→16–17)", x["KR출원_연성장%(08-09→16-17)"], "%/년", f"{x['사례그룹']} {x['산업명']}: KR 출원 {x['KR출원_연성장%(08-09→16-17)']}%, KOSIS 출하 {x['KOSIS출하_연성장%']}%, 종사자 {x['KOSIS종사자_연성장%(08-09→16-17)']}%, 사업체 {x['KOSIS사업체_연성장%']}%(연평균)", S(SP + "t5_cases.csv"),
      "같은 창으로 맞춘 값. 장기 창(05–07→15–17) KR 출원 성장과 섞지 말 것" + ("; 반례" if k == "26111" else ""),
      (f"split34 SUMMARY 사례표는 KR 출원에 장기 창 값({x['KR출원_연성장%(05-07→15-17)']}%)을 KOSIS 단기 창과 섞어 표기 — 창 불일치" if k in ("26211", "26212", "29280", "26111") else ""))
dv = rows(SP + "t4_divergence_index.csv")
F("SP-DIV", "트렌드·분화 지수", ", ".join(f"{x['연계표코드']} {x['분화지수_평균절대편차%p']}" for x in dv[:3]), "%p", "분화 지수(그룹 안 KR 출원 성장 평균절대편차) 상위: " + ", ".join(f"{x['연계표코드']} {x['분화지수_평균절대편차%p']}%p" for x in dv[:4]) + f"; 부호 갈림 그룹 {sum(1 for x in dv if x['부호갈림']=='True')}개(" + "·".join(x['연계표코드'] for x in dv if x['부호갈림']=='True') + ")", S(SP + "t4_divergence_index.csv"), "")
t2b = rows(SP + "t2b_cross_section_prov_vs_cw.csv")
F("SP-XSEC", "트렌드·불리한 결과(34 단면)", "CI 0 포함", "", "34개 전체 단면 상관에서 제공 5자리는 연계표 그룹 합계(상한)를 넘지 못함: KR 출원 차이 " + f"{min(float(x['차이']) for x in t2b if x['특허']=='KR'):+.3f}~{max(float(x['차이']) for x in t2b if x['특허']=='KR'):+.3f}, 5개국 {min(float(x['차이']) for x in t2b if x['특허']=='5국'):+.3f}~{max(float(x['차이']) for x in t2b if x['특허']=='5국'):+.3f}", S(SP + "t2b_cross_section_prov_vs_cw.csv"), "장점은 산업 간 수준이 아니라 그룹 안을 가르는 데")
fa = {(x["목표"], x["L"], x["구간"]): x for x in rows(SP + "t3a_forecast_summary.csv")}
x0 = fa[("5국", "0", "개발")]; x2 = fa[("KR", "2", "개발")]
F("SP-FCST", "트렌드·특허 미래 예측(그룹 안 순위)", x0["쌍순서_M"], "쌍 순서", f"L=0(실시간 아님) 5개국 쌍 순서 {x0['쌍순서_M']}({x0['쌍순서_M>0.5원점']}), RMSE {x0['RMSE차이(M−G)']} {x0['차이_95CI(KSIC부트)']}; 공개 지연 2년(L=2) KR RMSE 차이 +{x2['RMSE차이(M−G)']}({x2['M우세원점']}), 쌍 순서 {x2['쌍순서_M']}", S(SP + "t3a_forecast_summary.csv"), "주 설정 L=2에서는 이득 없음. 선행지표 주장 금지")

# ---------------------------------------------------------------- 11. 5개국 지형도·사분면 (map5)
MP = "_shared_report/fusion_advantage/map5/"
chk = jl(MP + "m2_checks.json"); cs5 = {x["cc"]: x for x in rows(MP + "m2_country_summary.csv")}
F("MAP-KR-RTA", "트렌드·5개국 지형도", str(chk["kr_rta_gt1_P3"]), "산업", f"한국은 {chk['kr_rta_gt1_P3']}/50 산업에서 RTA>1, {chk['kr_leader_P3']}개 산업에서 5개국 1위(출원 2017–19, 동일가중 RTA)", S(MP + "m2_checks.json"), "라벨 민감성 각주 필수(공개 배정 시 한국 1위 8–16개, 한국 RTA ρ 0.20–0.36)")
F("MAP-LEADERS", "트렌드·5개국 지형도", "CN 18", "산업", "산업별 1위 수: " + ", ".join(f"{k} {cs5[k]['n_ksic_leader_P3']}" for k in ["CN", "EP", "JP", "US", "KR"]) + f"; 사후 구간(P4) KR {cs5['KR']['n_ksic_leader_P4']}개", S(MP + "m2_country_summary.csv"), "P4는 사후 관찰")
F("MAP-STAB", "트렌드·5개국 지형도", r(chk["rta_KR_P3_vs_P4_spearman"]), "ρ", f"한국 RTA 순위 두 기간 사이 ρ {r(chk['rta_KR_P3_vs_P4_spearman'])}", S(MP + "m2_checks.json"), "")
prof = rows(MP + "m2_ksic_profile.csv")
import math
kr1 = sorted([x for x in prof if x["leader_P3"] == "KR"], key=lambda x: -float(x["rta_KR_P3"]))
F("MAP-KR-TOP", "트렌드·5개국 지형도(한국 1위 산업)", r(float(kr1[0]["rta_KR_P3"]), 2), "RTA", "한국 1위: " + ", ".join(f"{x['ksic']} {x['name']} {r(float(x['rta_KR_P3']),2)}" for x in kr1), S(MP + "m2_ksic_profile.csv"), "라벨 민감성 각주 필수")
gap = [x for x in prof if x["leader_P3"] != "KR" and x["kr_gap_log_P3"] not in ("", None) and float(x["kr_gap_log_P3"]) < math.log(0.7)]
gs = sorted(gap, key=lambda x: float(x["kr_gap_log_P3"]))
F("MAP-GAP", "트렌드·5개국 지형도(격차)", str(len(gap)), "산업", f"한국 RTA가 선도국의 0.7배 미만인 산업 {len(gap)}개; 최대 격차 " + ", ".join(f"{x['name']} {r(math.exp(float(x['kr_gap_log_P3'])),2)}배({x['leader_P3']})" for x in gs[:3]), S(MP + "m2_ksic_profile.csv"), "kr_gap_log_P3 = log(한국 RTA/선도국 RTA)로 계산")
m7 = jl(MP + "m7_summary.json")
F("MAP-CAGR", "트렌드·5개국 총량 성장", pc(m7["total_cagr_P1P3"]["KR"]), "%/년", "총량 연평균 성장(2008–10→2017–19): " + ", ".join(f"{k} {spc(m7['total_cagr_P1P3'][k])}%" for k in ["KR", "US", "JP", "CN", "EP"]), S(MP + "m7_summary.json"), "")
F("MAP-RELG", "트렌드·한국 상대 성장", str(m7["kr_rel_growth_vs_USJPEP_pos_n"]), "산업", f"US·JP·EP 대비 한국 상대 성장 양수 {m7['kr_rel_growth_vs_USJPEP_pos_n']}/50(해외 4개국 평균 대비는 CN 지배로 {m7['kr_rel_growth_vs_4_pos_n']}/50)", S(MP + "m7_summary.json"), "본문은 CN 제외 기준")
F("MAP-ABROAD", "트렌드·5개국 데이터에서만 나오는 지표", r(chk["kr_rta_vs_origin_abroad_rta_spearman"]), "ρ", f"우선권 KR 해외 출원 RTA와 국내 관청 RTA 순위상관 {r(chk['kr_rta_vs_origin_abroad_rta_spearman'])}", S(MP + "m2_checks.json"), "")
F("MAP-WIPO", "트렌드·전처리(WIPO 보정)", f"{chk['coverage_flagged_cells_n']}/{chk['coverage_cells_n']}", "칸", f"WIPO 대조 ±15% 이탈 국가×연도 {len(chk['coverage_flagged_total'])}개({chk['coverage_flagged_cells_n']}/{chk['coverage_cells_n']}칸) 보정: JP 2021 계수 {chk['coverage_flagged_total'][2][4]}, CN 2010·2013, US 2003·2004", S(MP + "m2_checks.json"), "공개연도 계열에만 적용")
F("MAP-JP2021", "트렌드·전처리(JP 2021)", "49,397→63,886", "건", f"JP 2021 공개 총량 {m7['jp_pub_total_raw_corr_2019_2023']['2021'][0]:,}→{m7['jp_pub_total_raw_corr_2019_2023']['2021'][1]:,}, KSIC 순위 ρ {r(m7['jp_2021_ksic_rank_raw_vs_corr_spearman'])}", S(MP + "m7_summary.json"), "JP 2021–22 감소를 실제 일본 출원 감소로 해석 금지")
F("MAP-JPUNIF", "트렌드·전처리(JP 감소 균일성)", f"{min(chk['jp_sd_dlog_share_2019_2022']):.3f}–{max(chk['jp_sd_dlog_share_2019_2022']):.3f}", "SD", f"JP 2019–22 구성 변동 SD {min(chk['jp_sd_dlog_share_2019_2022']):.3f}–{max(chk['jp_sd_dlog_share_2019_2022']):.3f}(평시 중앙값 {chk['jp_sd_dlog_share_2005_2018_median']:.3f})", S(MP + "m2_checks.json"), "", "R8 문서는 0.098–0.112(파일 0.0982–0.1118 반올림)")
m3 = jl(MP + "m3_summary.json")
F("MAP-LEADLAG", "트렌드·선행-후행", f"{m3['dev_fwd_q<0.10']}/400", "쌍", f"산업별 그랜저형 시험 400개 중 BH q<0.10 통과 {m3['dev_fwd_q<0.10']+m3['dev_rev_q<0.10']}개; 원 p<0.05는 방향마다 {m3['dev_fwd_rawp<0.05(bonf)']}개(우연 기대 {m3['expected_false_at_0.05_of_200']})", S(MP + "m3_summary.json"), "\"해외 특허가 한국을 선행한다\", \"선행지표\" 금지")
pd_ = {(x[0], x[1], x[2]): x for x in m3["panel_dev"]}
pdb = {(x[0], x[1], x[2]): x for x in jl(MP + "m3b_summary.json")["panel_dev"]}
F("MAP-PANEL", "트렌드·선행-후행(패널, 우선권 포함)", r(pd_[("US", "KR", 1)][3]), "β", f"패널 1년 시차: US→KR β {r(pd_[('US','KR',1)][3])}(q {r(pd_[('US','KR',1)][6])}), KR→US {r(pd_[('KR','US',1)][3])}(q {r(pd_[('KR','US',1)][6])}), JP→KR {r(pd_[('JP','KR',1)][3])}(q {r(pd_[('JP','KR',1)][6])})", S(MP + "m3_summary.json"), "우선권 경로를 빼면 사라진다")
qs = [v[6] for k_, v in pdb.items() if k_[2] == 1]
F("MAP-PURGED", "트렌드·선행-후행(우선권 제거)", r(min(qs), 3), "q 최솟값", f"우선권 경로 제거 후 1년 시차 q ≥ {r(min(qs),3)}(KR→US β {r(pdb[('KR','US',1)][3])}, JP→KR {r(pdb[('JP','KR',1)][3])})", S(MP + "m3b_summary.json"), "", "R8 문서 'q ≥ 0.084' = 파일 최솟값 0.0839 반올림")
m3c = jl(MP + "m3c_contemporaneous.json")
F("MAP-SYNC", "트렌드·같은 해 동조", r(m3c["raw_US"]["mean_corr"]), "상관", "같은 해 동조(연도 효과 제거): " + ", ".join(f"{k} {r(m3c['raw_'+k]['mean_corr'])}" for k in ["US", "JP", "CN", "EP"]) + "; 우선권 제거 후 " + ", ".join(f"{k} {r(m3c['origin_purged_'+k]['mean_corr'])}" for k in ["US", "JP", "CN", "EP"]) + "(순열 p ≤ 0.034)", S(MP + "m3c_contemporaneous.json"), "")
F("MAP-PERSIST", "트렌드·[사후] 성장 순위 지속", r(m7["posthoc_persistence_spearman_relgrowth_dev_vs_pub"]), "ρ", f"[사후] 개발 기간 상대 성장 순위 ↔ 공개 2017–19→2022–24 상대 성장 순위 {r(m7['posthoc_persistence_spearman_relgrowth_dev_vs_pub'])}", S(MP + "m7_summary.json"), "\"성장 순위가 지속된다\" 금지")
m4 = jl(MP + "m4_summary.json")
for k, lab in [("rho_patRTA_expRCA_P2", "특허 RTA ↔ 수출 RCA"), ("rho_abroadRTA_expRCA_P2", "해외 출원 RTA ↔ 수출 RCA"), ("rho_patgrowth_empgrowth_0817", "특허 증가 ↔ 종사자 증가(08–17)"), ("rho_patgrowth_shipgrowth_0817", "특허 증가 ↔ 출하 증가"), ("rho_patRTAchange_expRCAchange_0817", "RTA 변화 ↔ RCA 변화")]:
    x = m4[k]
    F(f"MAP-{k}", "트렌드·같은 KSIC로 이은 산업 지표", r(x["rho"]), "ρ", f"{lab} ρ {r(x['rho'])} {ci(*x['ci'])}(n {x['n']})", S(MP + "m4_summary.json"), "개발 2015–17. \"특허가 수출을 예측한다\" 금지" + ("; 0.545는 공개 배정으로도 0.44–0.57" if "abroad" in k else ""))
q = m4["quad_tech_export_P2_counts"]
F("QUAD-COUNTS", "트렌드·기술×수출 사분면", f"{q['기술 강·수출 강(정합)']}/{q['기술 강·수출 약(괴리)']}/{q['기술 약·수출 강(괴리)']}/{q['기술 약·수출 약(정합)']}", "산업", f"기술 강·수출 강 {q['기술 강·수출 강(정합)']}, 기술 강·수출 약 {q['기술 강·수출 약(괴리)']}, 기술 약·수출 강 {q['기술 약·수출 강(괴리)']}, 기술 약·수출 약 {q['기술 약·수출 약(정합)']}, 판정불가 {q['판정불가']}", S(MP + "m4_summary.json"), "48 KSIC. \"괴리 산업은 수출이 늘 것\" 금지")
F("QUAD-PERSIST", "트렌드·사분면 안정성", pc(m4["quad_P2_vs_P4_same_share"]), "%", f"[사후] 다음 기간에도 {pc(m4['quad_P2_vs_P4_same_share'])}%가 같은 사분면", S(MP + "m4_summary.json"), "사후")
F("QUAD-CWMIX", "트렌드·사분면(연계표로 못 그림)", f"{m4['cw_groups_with_mixed_quadrants']}/{m4['cw_multi_groups']}", "그룹", f"연계표가 한 코드로 묶는 {m4['cw_multi_groups']}개 그룹({m4['cw_multi_ksic']} KSIC) 중 {m4['cw_groups_with_mixed_quadrants']}개 그룹({m4['cw_ksic_in_mixed_groups']} KSIC)에서 사분면이 섞임", S(MP + "m4_summary.json"), "연계표 대비 사실")
F("QUAD-LIST-SW", "트렌드·사분면 목록(기술 강·수출 약)", str(len(m4["lists"]["기술 강·수출 약(괴리)"])), "산업", "기술 강·수출 약: " + ", ".join(x.split(" ", 1)[1] for x in m4["lists"]["기술 강·수출 약(괴리)"]), S(MP + "m4_summary.json"), "")
F("QUAD-LIST-WS", "트렌드·사분면 목록(기술 약·수출 강)", str(len(m4["lists"]["기술 약·수출 강(괴리)"])), "산업", "기술 약·수출 강: " + ", ".join(x.split(" ", 1)[1] for x in m4["lists"]["기술 약·수출 강(괴리)"]), S(MP + "m4_summary.json"), "")
m4b = jl(MP + "m4b_summary.json")
F("QUAD-27", "트렌드·불리한 결과(27그룹 해상도)", r(m4b["patRTA_vs_exportRCA"]["rho_label"]), "ρ", f"같은 27그룹 해상도 수출 RCA 상관: 제공 라벨 RTA {r(m4b['patRTA_vs_exportRCA']['rho_label'])} 대 연계표 {r(m4b['patRTA_vs_exportRCA']['rho_crosswalk'])}(차이 CI {ci(*m4b['patRTA_vs_exportRCA']['ci'])}), 해외 출원 {r(m4b['abroadRTA_vs_exportRCA']['rho_label'])} 대 {r(m4b['abroadRTA_vs_exportRCA']['rho_crosswalk'])}", S(MP + "m4b_summary.json"), "n 26, CI 0 포함 → \"유의하게 정합\" 금지")
F("MAP-WITHIN50", "트렌드·그룹 안 분산(50 분모)", pc(m4["within_group_var_share_ship_g_0817"]), "%", f"50 KSIC 분모 그룹 안 분산: 출하 {pc(m4['within_group_var_share_ship_g_0817'])}%, 한국 RTA {pc(m4['within_group_var_share_pat_rta_P2'])}%, 특허 증가율 {pc(m4['within_group_var_share_pat_g_0817'])}%", S(MP + "m4_summary.json"), "본문 표준은 split34 t1(34 KSIC) 값",
  "map5 SUMMARY의 '한국 RTA 차이의 79.4%가 그룹 안'은 JSON에서 확인 불가 → 사용 금지(R8 §6-5)")
m5 = jl(MP + "m5_summary.json")
F("DASH-RANGE", "트렌드·50산업 6지표 행렬", f"{m5['track_pair_spearman_range'][0]}~{m5['track_pair_spearman_range'][1]}", "ρ", f"트랙 간 순위상관 {m5['track_pair_spearman_range'][0]}~+{m5['track_pair_spearman_range'][1]}, 한국 RTA와 |ρ| ≤ {max(abs(v) for v in m5['rta_vs_tracks'].values())}", S(MP + "m5_summary.json"), "합성 지수 만들지 않음. T4 열은 공식 모형 아닌 X26 값(각주)")
F("DASH-HYP", "트렌드·가설 후보", str(len(m5["tech_strong_export_weak_and_T3_top_third"])), "산업", "기술 강·수출 약이면서 T3 상위 1/3: " + ", ".join(x.split(" ", 1)[1] for x in m5["tech_strong_export_weak_and_T3_top_third"]), S(MP + "m5_summary.json"), "검증 안 된 가설 후보")

# ---------------------------------------------------------------- 12. NTIS 표
NT = "_shared_report/"
def pcf(x, nd=1):
    return f"{float(x)*100:.{nd}f}"
def rf(x, nd=1):
    return f"{float(x):.{nd}f}"
t1n = rows(NT + "ntis_t1_ksic_gov_share.csv")
F("NTIS-SRC", "NTIS·출처", "2026-09-29", "수집일", "데이터 출처: NTIS Open API 활용(국가과학기술지식정보서비스 과제검색·성과검색)", S(NT + "ntis_tables.md"), "비상업·제3자 제공 금지. 스냅샷 준-정합, 2006년 이전 비교 안 함, 연관이지 인과 아님")
for x in t1n:
    k = x["ksic"]
    fid = "NTIS-T1-ALL" if k == "전체" else f"NTIS-T1-{k}"
    F(fid, "NTIS 표1·KSIC별 정부 R&D 기원 특허 비중", pcf(x["gov_share_2007_22"]), "%",
      f"{x['name']}{'' if k=='전체' else '('+k+')'} 출원 {int(x['n_2007_22']):,}건, 정부 기원 비중 {pcf(x['gov_share_2007_22'])}%: 2007–10 {pcf(x['gov_2007–10'])}% → 2011–14 {pcf(x['gov_2011–14'])}% → 2015–18 {pcf(x['gov_2015–18'])}% → 2019–22 {pcf(x['gov_2019–22'])}%({sg(float(x['chg_pp']),1)}%p); 수행 대학 {pcf(x['perf_univ'])}%·출연연 {pcf(x['perf_gri'])}%·기업 {pcf(x['perf_firm'])}%",
      S(NT + "ntis_t1_ksic_gov_share.csv"), "KR 특허(실용신안 제외), 출원 2007–2022. 최근 연도는 보고 지연으로 과소 가능")
for x in rows(NT + "ntis_t1b_year_gov_share.csv"):
    F(f"NTIS-T1B-{x['appl_year']}", "NTIS 표1b·출원연도별 정부 기원 비중", pcf(x["gov"]), "%", f"{x['appl_year']}년 출원 {int(x['n']):,}건, 정부 기원 {pcf(x['gov'])}%(대학 {pcf(x['univ'])}%, 출연연 {pcf(x['gri'])}%, 기업 {pcf(x['firm'])}%)", S(NT + "ntis_t1b_year_gov_share.csv"), "2022년 값은 보고 지연으로 더 오를 수 있음")
for x in rows(NT + "ntis_t2_lapse_by_performer.csv"):
    F(f"NTIS-T2-{x['origin']}", "NTIS 표2·수행주체별 3년 내 포기율", pcf(x["rate_gov"]), "%", f"원점 {x['origin']}(기준일 {x['L']}, {int(x['n']):,}건): 전체 {pcf(x['rate_all'])}%, 비기원 {pcf(x['rate_nongov'])}%, 정부 기원 {pcf(x['rate_gov'])}%(비중 {pcf(x['share_gov'])}%); 대학 {pcf(x['rate_univ'])}%, 출연연 {pcf(x['rate_gri'])}%, 대기업·중견 {pcf(x['rate_big'])}%, 중소 {pcf(x['rate_sme'])}%", S(NT + "ntis_t2_lapse_by_performer.csv"), "T4 개발 원점 프레임, 기준일에 알 수 있던 연결만")
for x in rows(NT + "ntis_t3_cohort_lapse_transfer.csv"):
    F(f"NTIS-T3-{x['group']}", "NTIS 표3·등록 코호트 5년 내 포기·이전", pcf(x["lapse5"]), "%", f"{x['group']}({int(x['n']):,}건): 5년 내 포기 {pcf(x['lapse5'])}%, 이전 {pcf(x['transfer5'])}%(포기 2007–11 {pcf(x['lapse5_2007_11'])}%·2012–16 {pcf(x['lapse5_2012_16'])}%)", S(NT + "ntis_t3_cohort_lapse_transfer.csv"), "등록 2007–2016, 수행주체 스냅샷 전체 연결(기술통계)")
for x in rows(NT + "ntis_t4_entry_by_gov.csv"):
    mult = float(x["entry_g_any"]) / float(x["entry_not_g_any"])
    lift = (f", 리프트 상위25% {pcf(x['entry_gtgt_top25'],2)}% 대 하위25% {pcf(x['entry_gtgt_bot25'],2)}%" if x["entry_gtgt_top25"] else "")
    F(f"NTIS-T4-{x['c']}", "NTIS 표4·정부 R&D 이력별 기업×산업 첫 진입률", r(mult, 1), "배", f"원점 {x['c']}(진입 {x['window']}, 위험쌍 {int(x['pairs']):,}): 전체 {pcf(x['entry_all'],2)}%, 정부 기원 특허 보유 {pcf(x['entry_g_any'],2)}% 대 미보유 {pcf(x['entry_not_g_any'],2)}%({r(mult,1)}배){lift}", S(NT + "ntis_t4_entry_by_gov.csv"), "개발 원점 2008–2012, 보조 2015(최종시험 c=2019 제외)")
for x in rows(NT + "ntis_t5_funds_by_ksic.csv"):
    F(f"NTIS-T5-{x['ksic']}", "NTIS 표5·KSIC별 정부연구비 배분", rf(float(x["funds_bil_2021–23"]), 1), "십억원/년", f"{x['name']}({x['ksic']}): 연평균 {rf(float(x['funds_bil_2009–12']),1)}(2009–12) → {rf(float(x['funds_bil_2021–23']),1)}십억원(2021–23), 50개 중 비중 {pcf(x['share50_2009–12'])}% → {pcf(x['share50_2021–23'])}%, 연평균 증가율 {pcf(x['cagr_2009_12_to_2021_23'])}%", S(NT + "ntis_t5_funds_by_ksic.csv"), "과학기술표준분류→KSIC 배분표(성과 Year ≤ t−2). 50개 합계는 연도 전체의 22–25%")
for x in rows(NT + "ntis_t6_funds_year.csv"):
    F(f"NTIS-T6-{x['year']}", "NTIS 표6·연도별 50개 KSIC 배분 정부연구비", rf(float(x["gov_50_bil"]), 1), "십억원", f"{x['year']}년 50개 KSIC 배분 {rf(float(x['gov_50_bil']),1)}십억원 / 수집 전체 {rf(float(x['year_total_bil']),1)}십억원({pcf(x['share50_of_nat'])}%), 기업 {pcf(x['firm_share'])}%·대학 {pcf(x['univ_share'])}%·출연연 {pcf(x['gri_share'])}%", S(NT + "ntis_t6_funds_year.csv"), "수집 과제 합계는 공식 국가R&D 총액보다 작음(보안과제 등 미제공). 2024년 제외")
F("NTIS-T4-MULT", "NTIS 표4·정부 기원 특허 보유 기업 진입률", "1.9–3.0", "배", "정부 기원 특허를 가진 기업의 산업 첫 진입률은 1.9–3.0배; 과제분류가 목표 산업을 가리키는(리프트 상위 25%) 쌍은 하위 25%보다 3.4–4.8배", S(NT + "ntis_t4_entry_by_gov.csv"), "개발 원점 2008–2012·보조 2015. T1 R4E01 채택 근거")
F("NTIS-T3-SUM", "NTIS 표3·5년 내 포기·이전", "14.2", "%", "등록 2007–2016 코호트 5년 내 포기: 정부 R&D 기원 14.2% 대 비기원 21.0%; 이전 10.5% 대 11.5%", S(NT + "ntis_t3_cohort_lapse_transfer.csv"), "수행주체는 스냅샷 전체 연결(기술통계)")

# ---------------------------------------------------------------- 13. 데이터 규모
F("DATA-META", "데이터 규모·제공 데이터", "15,141,159", "행", "제공 융합데이터 5개국 메타 15,141,159행(1행=출원×KSIC 쌍, 원본 TSV 50개 29.5GB)", S("_shared/README.md"), "", f"c1_label_agreement.csv '전체' n {tot:,}로 일치")
F("DATA-KR-APP", "데이터 규모·KR 출원", "1,087,511", "건", "제공 KR 출원 1,087,511건", S("_shared/README.md"), "", f"c1·ntis_link_rates 값 {cnt['KR']:,}과 일치")
nlr = rows("_shared/ext/ntis_link_rates.csv")[0]
F("DATA-KR-REG", "데이터 규모·KR 등록", f"{int(nlr['n_registered']):,}", "건", f"제공 범위 KR 등록특허 {int(nlr['n_registered']):,}건", S("_shared/ext/ntis_link_rates.csv"), "KR 등록 데이터 최종일 2025-08-22")
reg_counts = {}
try:
    con = sqlite3.connect("file:" + os.path.join(BASE, "수집파이프라인", "registry_full.sqlite") + "?mode=ro", uri=True)
    tabs = [t[0] for t in con.execute("select name from sqlite_master where type='table'")]
    for t in tabs:
        reg_counts[t] = con.execute(f'select count(*) from "{t}"').fetchone()[0]
    con.close()
except Exception as ex:
    reg_counts = {"_error": str(ex)}
if "_error" not in reg_counts:
    totr = sum(reg_counts.values())
    F("DATA-REGISTRY", "데이터 규모·등록원부", f"{totr:,}", "행", f"등록원부 완전판 {len(reg_counts)}개 테이블 {totr:,}행(약 {r(totr/1e4,0)}만 행): " + ", ".join(f"{k} {v:,}" for k, v in reg_counts.items()), "수집파이프라인\\registry_full.sqlite (읽기 전용 집계)",
      "자체 수집 DB(data.go.kr·KIPRIS Plus 등록원부). 13개 중 unknown_keys는 0행. 실시권 테이블은 특허당 1행 구조일 수 있음")
    F("DATA-REG-EVENTS", "데이터 규모·등록원부 사건", "122,974", "건", "권리이전 발생 122,974건(18.7%), 소멸등록 349,507, 회복등록 9,523", S("_shared/README.md"), "README 통계")
br = {}
try:
    con = sqlite3.connect("file:" + os.path.join(BASE, "수집파이프라인", "business_rights.sqlite") + "?mode=ro", uri=True)
    for t in ["right_list", "applicant_query"]:
        br[t] = con.execute(f'select count(*) from "{t}"').fetchone()[0]
    con.close()
except Exception as ex:
    br = {"_error": str(ex)}
F("DATA-BIZRIGHTS", "데이터 규모·사업자 권리목록", f"{br.get('right_list', 0):,}", "행", f"법인 포트폴리오 권리목록 right_list {br.get('right_list',0):,}행(특허·실용·디자인·상표 포함), 제공범위 밖 특허 1,494,298건(73.0%)", "수집파이프라인\\business_rights.sqlite (읽기 전용 집계); " + S("_shared/README.md"),
  "2026 스냅샷 → 시점 정렬 필수", f"applicant_query 테이블은 {br.get('applicant_query',0):,}행(조회 기록). README의 '법인 53,967개'와 행 수가 다름 — 법인 수는 README 값")
F("DATA-NTIS-RP", "데이터 규모·NTIS 성과 특허", "1,094,201", "건", "NTIS 성과 특허 1,094,201건 전수(중복 0)", S("_shared/ext/ntis_SOURCES.md"), "NTIS Open API 활용, 2026-09-29 기준")
F("DATA-NTIS-PJ", "데이터 규모·NTIS 과제", "1,145,167", "건", "NTIS 과제 2002–2024 1,145,167건(1,145,177건 중 99.999%)", S("_shared/ext/ntis_SOURCES_B.md"), "", "ntis_B_ksic_report.json n_projects_2002_2024=1145167로 일치")
F("DATA-CORPAPP", "데이터 규모·출원인-법인 연결", "385,256", "행", "출원인코드↔법인번호/사업자번호 385,256행(CORP_APPLICANT 2026-07)", S("_shared/README.md"), "사후 목록(생존 편향 주의)")
F("DATA-DART", "데이터 규모·DART", "12,322", "사", "DART 기업 12,322사(induty_code)", S("_shared/README.md"), "")

# ---------------------------------------------------------------- 14. 연계율
F("LINK-NTIS", "연계율·NTIS 성과 연결", pc(nlr["link_rate"], 2), "%", f"제공 KR 출원의 NTIS 성과 연결 {pc(nlr['link_rate'],2)}%({int(nlr['n_linked']):,}/{int(nlr['n_patents']):,}), 등록 {pc(nlr['link_rate_registered'],2)}%({int(nlr['n_linked_registered']):,}/{int(nlr['n_registered']):,})", S("_shared/ext/ntis_link_rates.csv"), "2007년 이후 출원·등록분부터 연결. KSIC별 0.94–30.45%")
nps = {x["항목"].strip(): x for x in rows("_shared/ext/firm_nps_link_rates.csv")}
F("LINK-NPS", "연계율·국민연금 사업장", pc(nps["NPS 사업장 매칭 출원인의 특허"]["비율"]), "%", f"KR 등록특허 중 NPS 사업장 매칭 출원인 특허 {pc(nps['NPS 사업장 매칭 출원인의 특허']['비율'])}%(사업자번호 확보 특허 대비 {pc(nps['└ 사업자번호 확보 특허 대비']['비율'])}%), 사업자번호 확보 {pc(nps['사업자번호 확보 출원인의 특허']['비율'])}%", S("_shared/ext/firm_nps_link_rates.csv"), "출처: 국민연금공단 가입 사업장(공공데이터포털). NPS 2026 상호 키 오분류는 별칭 보정 필요(N4·N7)")
F("LINK-NPS-KSIC", "연계율·국민연금(KSIC별)", nps["50 KSIC별 NPS 매칭 특허비중 최소/중앙/최대"]["값"], "비율", "50 KSIC별 NPS 매칭 특허 비중 최소/중앙/최대 " + nps["50 KSIC별 NPS 매칭 특허비중 최소/중앙/최대"]["값"], S("_shared/ext/firm_nps_link_rates.csv"), "")
els = rows("T1_산업대기실/R2/ext_link_summary.csv")
for lid, x in zip(["LINK-IO", "LINK-HS6-IMP", "LINK-HS6-EXP", "LINK-KOSIS5", "LINK-5C"], els):
    F(lid, "연계율·KSIC 코드 다리", f"{x['ksic_linked']}/{x['ksic_total']}", "KSIC", f"{x['source']}: {x['link_key']} {x['ksic_linked']}/{x['ksic_total']} KSIC({x['note']})", S("T1_산업대기실/R2/ext_link_summary.csv"), "연결 가능성은 TXT도 같다(고유 장점 아님)" if "HS6" in x["source"] or "KOSIS" in x["source"] else "",
      ("R8·R9 문서는 KOSIS 5자리 50/50으로 표기(T1 R2 파일은 2008년 49/50 — 연도·파일 차이)" if "KOSIS" in x["source"] else ""))
F("LINK-KIND", "연계율·KIND 상장", "76.2", "%", "KIND 국내 주권(스팩 제외) 2,619건 중 법인번호 확보 76.2%(연도별 71–86%)", S("_ideas_R7/IDEAS_R7.md"), "R7 신규 원천")
el = rows("T1_산업대기실/R3/ext_link_stats.csv")
x15 = [y for y in el if y["origin"] == "2015"]
F("LINK-T1-R3", "연계율·T1 외부 원천(원점 2015)", "36.1/31.6/48.1", "법인 연결률", "; ".join(f"{y['source']}/{y['feature']} 법인 연결률 {pc(y['firm_link_rate'])}%" for y in x15 if y["feature"] in ("ds_lsum", "um_lsum", "tm_n", "acq_n", "pr_ppi_g3")), S("T1_산업대기실/R3/ext_link_stats.csv"), "디자인·실용신안·상표(KIPRIS), 권리이전, 물가(ECOS)")
F("LINK-N4", "연계율·N4 원점별", "75.5–80.8", "비율", "; ".join(f"t0 {y['t0']}: n {y['n']}, NPS {pc(y['nps_present'])}%, NTIS 수행 이력 {pc(y['ntis_ever'])}%" for y in rows("N4_특허행태_기업폐업조기경보/link_rates_dev.csv")), S("N4_특허행태_기업폐업조기경보/link_rates_dev.csv"), "개별 기업명 금지(폐업 위험)")

# 기록
with open(os.path.join(OUT, "facts.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "주제", "수치", "단위", "정확한 문장 조각", "출처 파일 경로", "조건·경고", "note"])
    w.writerows(FACTS)
with open(os.path.join(OUT, "facts_mismatch.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "note"]); w.writerows(MISMATCH)
print("rows", len(FACTS), "mismatch", len(MISMATCH))
