# C13-T3 s6: 미반영 ERRATA 통합표. 반영 기록 (1)-(12) 기준 미반영 후보(s2_status 결과) 중 반박·보류·대체·변경불요 제외 후,
# 현재 원고(v14 원고 sha1 13c3d505)에서 옛 문구/새 문구 존재를 문자열로 확인하고 줄 번호·충돌·글자 수·우선순위를 적는다.
import csv, json, sys, hashlib, re
sys.stdout.reconfigure(encoding="utf-8")
MS = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\manuscript_v3.md"
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("13c3d505")
L = open(MS, encoding="utf-8").read().splitlines()
st = {o["id"]: o for o in json.load(open(sys.argv[1], encoding="utf-8")) if not o.get("addendum")}

def find(s):
    return [i + 1 for i, t in enumerate(L) if s in t]

# (id, scope, old_anchor(현재 원고에서 찾을 대상 문구), new_marker(새 문구 고유 조각), 필수/선택, Δ글자(제안 기준 추정), 우선순위, 메모)
# scope: body(제출본 1–441행), appendix(부록 445행~), figure, outside(원고 밖), facts
R = [
 # ---- 본문 ----
 ("E-C12T3-02", "body", "해외 신호와 외부 데이터가 그 위에 더한다", "빼도 같거나 오른다", "필수 권장", 25, 1,
  "주장 지도 K(219행) '과장'의 수정안. 같은 줄 E-C09T3-01과 병합 시 합산. 해외 신호는 최종 SB3 대비 0.182로 받쳐져 원 문구 그대로 타당"),
 ("E-C11T2-02b(168행 동률 단서)", "body", "어느 배정이 맞는지는 판정하지 못했다", "동률", "필수 권장", 20, 1,
  "주장 지도 168행 '단서 누락'. C12-T2 시험(+27자)은 21쪽 → 같은 쪽(8쪽) 축약 동반 필요"),
 ("E-C12T3-04", "body", "물가·산업연관표는 원점 이후 공표된 현행 빈티지", "생산지수", "선택(권장)", 5, 2,
  "143행 단서 누락 해소. 같은 괄호 E-C12T2-02(반영됨)와 정합. 긴 판 +40자"),
 ("E-C07T3-03", "body", "포기 위험 열은 공식 외 시험에서 단순 지속 지수보다 낫지 않았다", "부가가치", "선택(권장)", 110, 2,
  "4.3 분쟁 위험 열 단서. 원안 +110자는 18쪽 연쇄로 21쪽 위험 → 축약 '(포기·분쟁 위험 열은 …)' +8자 대안(E-C13T3-07 참조)"),
 ("E-C07T3-04", "body", "포착한 39건 중 30건은 분쟁 이력이 있는 특허의 재분쟁이다", "4/65", "선택", 60, 3,
  "표 11 387행 E-C13T3-06(0자 교체)과 같은 사실 — 둘 중 하나만"),
 ("E-C02T2-05", "body", "분쟁 90건에서", "2.1–2.9%", "선택", 25, 4, "N8 전향 결손 단서. E-C08T2-04 반영 때 '지면상 생략'"),
 ("E-C11T2-07", "body", "공개번호 연결이면 46.9–53.5%", "22%", "선택", 68, 3, "73행(3쪽, 여유 3.5줄). C12-T2 C1 시험 20쪽"),
 ("E-C11T1-03", "body", "연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%", "2013년 이후", "선택", 113, 4, "410행(19쪽) 21쪽 시험 사례 — 같은 줄 E-C10T1-03·E-C05T3-03과 경합"),
 ("E-C10T1-03", "body", "사업자 권리목록·DART 업종은 2026년 스냅샷이고", "판 권리", "선택", 75, 4, "410행 경합"),
 ("E-C05T3-03", "body", "DART 우위는 2010–14년 문서에서도 남는다", "보고 누락 신호", "선택", 90, 4, "410행 끝 경합(E-C02T1-08 대체)"),
 ("E-C03T1-03", "body", "폐업일 보정(사전 등록 보조 변형)", "이름 매칭으로만 잡힌다", "선택", 70, 4, "242 또는 410행"),
 ("E-C08T2-08", "body", "전체 AUC +0.0004~+0.0011, 국민연금 보정 시", "국민연금 단독은 CI 0 포함", "선택", 15, 4, "242행(11쪽)"),
 ("E-C05T1-04", "body", "상위 100셀에서는 50%가 맞았다(기준선 23–39%)", "신설 분류 셀)", "선택", 20, 4, "209행(10쪽)"),
 ("E-C09T3-01", "body", "부상의 41%는 출원인 한 곳이 증가의 절반 이상을 냈다", "0.423–0.429", "선택", 60, 4, "219행 — E-C12T3-02와 같은 문단"),
 ("E-C11T3-02", "body", "Reliance on Science 과학 의존도를 붙였다", "확장 패밀리", "선택", 104, 4, "205행. C12-T2 C4 시험 21쪽"),
 ("E-C08T1-03", "body", "로 정의에 민감하다.", "0.30 [−0.02, 0.60]", "선택", 30, 4, "172행. 사전 기준 '영향 없음'"),
 ("E-C10T1-05", "body", "| 보수적 skill | 0.395 [0.379, 0.412]", "−0.0007~−0.0046", "선택", 80, 4, "표 7 아래(235–240행)"),
 ("E-C08T2-10", "body", "지연판은 파일럿 점검 뒤 최종시험 전에 정한 주 모형이다", "0.339", "선택", 25, 4, "265행"),
 ("E-C05T2-04", "body", "산업 집계 해석층으로만 쓴다", "약 55%", "선택", 80, 4, "289행"),
 ("E-C06T1-07", "body", "등록업종 참조로 설명될 몫은 많아야 약 1%p다", "147사", "선택", 50, 4, "326행"),
 ("E-C06T1-08", "body", "근거 A를 'DART +7%p, NPS 같음'으로만 쓴다", "225사", "선택", 110, 4, "321·328행"),
 ("E-C08T2-09", "body", "(363사, 5자리 코드 없음)", "DART·KIND는 CI 0 포함", "선택", 18, 4, "323행"),
 ("E-C07T1-07", "body", "| 그룹 안 KOSIS 출하 몫 상승, AUC |", "결과 창이 2016년", "선택", 100, 4, "표 10 아래 주(360행). 원고에 불리한 결과의 민감도"),
 ("E-C08T3-04", "body", "T4는 강화 기준선 대비 CI 하한이 0에 닿고(−0.003)", "0.342", "선택", 60, 4, "409행 경합"),
 ("E-C08T3-05", "body", "T4는 강화 기준선 대비 CI 하한이 0에 닿고(−0.003)", "14,124", "선택", 130, 4, "409행 경합"),
 ("E-C07T1-10", "body", "무역·KOSIS는 해석층으로만 썼으며", "제조업 사업체의 16%", "선택", 70, 4, "411행"),
 ("E-C05T2-03", "body", "KR 등록특허의 44.4%(사업자번호 확보분의 93.1%)", "42.5%", "선택", 45, 4, "표 3 92행. 4.4-4 18%→96%와 내용 중복"),
 ("E-C06T1-06", "body", "12,322사 / 법인번호 확보 76.2%", "상장 9%", "선택", 25, 4, "표 3 93행"),
 ("E-C04T1-03", "body", "| ② | OpenDART 기업개황", "국세청 사업자등록 상태조회 | 사업자번호", "선택", 70, 4, "표 3 새 행 + 부록 OFF-N4-NTS"),
 ("E-C12T1-05", "body", "ETSI는 N5 결과로 썼다", "3GPP", "선택", 24, 4, "99행 각주(5쪽). 짧은 판 +24·긴 판 +81"),
 ("E-C09T1-01", "body", "제공 데이터 연도별 건수는 공식 통계 대비 비율이 안정적이다", "시기 단절이 없다", "선택", 55, 4, "143행 — E-C12T3-04와 같은 글머리"),
 ("E-C09T2-04", "body", "약 46–48%가 원점에 미공개", "약 47–48%", "선택", 0, 4, "0자"),
 ("E-C09T2-07", "body", "~", "", "선택", 0, 4, "범위 기호 통일(전역)"),
 ("E-C06T2-01", "figure", "C2620 디스플레이와 C2610 반도체 사례 — 출처", "연평균 로그 성장률, 2008", "선택", 25, 4, "197행 캡션. v8에서 시도 후 되돌림(효과 없음)"),
 ("E-C06T2-02", "figure", "그림 1. 제공 데이터를 중심에 둔 세 다리 연계 구조와 대조군 — 출처", "공개 최강(최대 58.8%)과의 비교", "선택", 50, 4, "82행 캡션 a안 또는 그림 재렌더 b안"),
 ("E-C06T2-04", "body", "수정 전 값(0.424)도 공개한다.", "(그림 2)", "선택", 18, 4, "109·194·259행 그림 번호 호출"),
 ("E-C05T4-01", "body", "따라서 '이 데이터가 아니면 안 되는 이유'의 중심은 과제 정의(3.6.1)다.", "추가 실험(공식 외)", "선택", 350, 4, "새 문단 — 지면상 사실상 불가(사용자 판단 항목)"),
 ("E-C10T4-03", "body", "Reliance on Science(Marx & Fuegi,", "2022", "선택", 110, 4, "표 13·참고문헌(20쪽) — 마지막 쪽 직접 증가"),
 ("E-C10T4-05", "body", "KOSIS(통계청 광업·제조업조사", "국가데이터처)", "선택", 20, 4, "표 13(20쪽)"),
 ("E-C11T3-01", "body", "Reliance on Science(Marx & Fuegi, CC BY-NC 4.0, 비상업)", "2026-09-25판", "선택", 26, 4, "표 13(20쪽). C12-T2 C3 시험 20쪽"),
 ("E-C12T3-03", "body", "광공업생산지수), ECOS(물가)", "9차 구계열", "선택", 15, 4, "표 13(20쪽)"),
 # ---- 부록(제출본 밖, 지면 0) ----
 ("E-C01-X2", "appendix", "| OFF-DEV2FINAL |", "", "선택", 0, 3, "출처표 '쓰인 절' 정리 — E-C02-T3-12와 병합"),
 ("E-C02-T3-12", "appendix", "| CIRC-AGREE-CC |", "", "선택", 0, 3, "E-C01-X2와 병합"),
 ("E-C02T1-07", "appendix", "| NTIS-T3-SUM |", "1.4–3.5%p", "선택", 0, 3, "출처표 값 보충"),
 ("E-C04T2-02", "appendix", "| OFF-T4-STRONG |", "B=1000", "선택", 0, 3, "T4 강화 기준선 시드 범위"),
 ("E-C07T1-08", "appendix", "| LINK-KOSIS5 |", "5,925/5,925", "선택", 0, 3, ""),
 ("E-C07T1-09", "appendix", "| SP-CASE-26212 |", "sensitivity_313", "선택", 0, 3, ""),
 ("E-C07T2-04", "appendix", "| A-COVER |", "C06-T1\\rescore_B.csv", "선택", 0, 3, "경로 표기"),
 ("E-C07T3-D1", "appendix", "| X-PLACEBO", "PROSP-N8", "선택", 0, 3, "새 행"),
 ("E-C08T1-04", "appendix", "| MAP-KR-TOP |", "1위 상실(C08-T1)", "선택", 0, 3, ""),
 ("E-C08T3-D1", "appendix", "| X-PLACEBO", "PROSP-T4", "선택", 0, 3, "새 행"),
 ("E-C09T1-02", "appendix", "| DATA-BIZRIGHTS |", "DATA-SCOPE-STAB", "선택", 0, 3, "새 행"),
 ("E-C09T3-02", "appendix", "| OFF-T3-NEWIPC |", "T3-APPFIELD", "선택", 0, 3, "새 행"),
 ("E-C12T2-08", "appendix", "| C-T3-AUCS |", "s6_t3_summary.json", "선택", 0, 3, "출처 열 보충"),
 ("E-C05T3-05", "outside", "", "", "선택", 0, 4, "figs\\fig09 CSV 빈칸(그림 픽셀 무관)"),
]
# 대상 소멸·대체 처리(통합표에 표시, 지면 계산 제외)
SUPERSEDED = [
 ("E-C03-T2-07", "대상 소멸", "224행 해당 문장이 v9 지면 조정에서 삭제됨"),
 ("E-C04T4-01", "대체 반영", "E-C05T2-01(4.4-4 연결률 18%→96%)이 같은 내용"),
 ("E-C06T1-02", "대체 처리", "원안 지시대로 E-C06T1-01만 반영하고 407행 '(3.6.2)' 참조 유지"),
 ("E-C07T3-05", "부분 반영", "395행 '과거 분쟁 기준선 대비 우위는 재현되지 않았고'가 같은 내용(4.4-3 위치만 다름)"),
 ("E-C10T4-04", "대체 반영", "E-C11T1-01(표 13 '사업자 권리목록') 반영으로 출처 표기 처리, 입수 경로는 부록 DATA-CORPAPP"),
]
OUTSIDE_FACTS = ["E-C11T2-05", "E-C11T3-03", "E-C12T2-11", "E-C12T3-F1", "E-C12T3-F2", "E-C12T1-F1"]

rows = []
for (i, scope, old, new, need, dch, pri, memo) in R:
    base = i.split("(")[0]
    errl = st.get(base, {}).get("line", "")
    if scope == "outside":
        rows.append(dict(id=i, errata_line=errl, scope=scope, v14_line="", in_manuscript="해당 없음(원고 밖)", conflicts="", need=need, delta_chars=0, priority=pri, note=memo)); continue
    if base == "E-C09T2-07":
        body_tilde = [k + 1 for k in range(441) if "~" in L[k]]
        rows.append(dict(id=i, errata_line=errl, scope=scope, v14_line=",".join(map(str, body_tilde)), in_manuscript=f"미반영('~' 본문 {len(body_tilde)}행)", conflicts="", need=need, delta_chars=0, priority=pri, note=memo)); continue
    lo = find(old) if old else []
    ln = find(new) if new else []
    if new and ln and not (base in ("E-C11T2-02b",) ):
        # 새 문구 조각이 이미 있는지(부록 행 포함) — 같은 줄에서만 반영으로 인정
        same = [x for x in ln if x in lo] if lo else ln
        inms = "이미 반영(기록 누락) 의심: " + ",".join(map(str, same)) if same else "미반영(새 조각은 다른 줄에만: " + ",".join(map(str, ln[:3])) + ")"
    else:
        inms = "미반영" if lo else "대상 문구 없음(재확인)"
    rows.append(dict(id=i, errata_line=errl, scope=scope, v14_line=",".join(map(str, lo[:3])), in_manuscript=inms, conflicts="", need=need, delta_chars=dch, priority=pri, note=memo))
# 같은 줄 충돌
from collections import defaultdict
byline = defaultdict(list)
for r in rows:
    for x in str(r["v14_line"]).split(","):
        if x and r["scope"] != "outside" and r["id"] != "E-C09T2-07": byline[x].append(r["id"])
for r in rows:
    cs = set()
    for x in str(r["v14_line"]).split(","):
        cs |= set(byline.get(x, [])) - {r["id"]}
    r["conflicts"] = ";".join(sorted(cs))
for (i, kind, memo) in SUPERSEDED:
    rows.append(dict(id=i, errata_line=st.get(i, {}).get("line", ""), scope="body", v14_line="", in_manuscript=kind, conflicts="", need="제외", delta_chars=0, priority="", note=memo))
for i in OUTSIDE_FACTS:
    rows.append(dict(id=i, errata_line=st.get(i, {}).get("line", ""), scope="facts", v14_line="", in_manuscript="해당 없음(facts_supplement)", conflicts="", need="선택", delta_chars=0, priority=3, note="facts 읽기 전용 — 다음 facts 정리 때"))
# 원고 밖 D 항목 일괄
listed = {r["id"].split("(")[0] for r in rows}
for k, o in st.items():
    if o["status"] == "미반영후보" and k not in listed:
        rows.append(dict(id=k, errata_line=o["line"], scope="outside", v14_line="", in_manuscript="해당 없음(원고 밖)", conflicts="", need="선택", delta_chars=0, priority=4, note=o["text"][:90]))
TB = {
 "E-C12T3-02": "원안 그대로 B13 포함 → 20쪽(S219 단독 20쪽, 쪽 변화 없음)",
 "E-C11T2-02b(168행 동률 단서)": "원안 +27자 S168 → 21쪽(8쪽부터 연쇄). 같은 문단 압축판 E-C13T3-08(순증 +5자)로 B13 포함 → 20쪽",
 "E-C12T3-04": "짧은 판 +5자 B13 포함 → 20쪽(S143 단독 쪽 변화 없음)",
 "E-C07T3-03": "원안 +110자 미시험(18쪽→19→20 연쇄 위험). 축약판 E-C13T3-07(+3자)로 B13 포함 → 20쪽",
 "E-C11T2-07": "E-C12T2-01 병합판(+36자) B14 포함 → 20쪽(3쪽 여유 흡수)",
 "E-C11T1-03": "원안 +113자 S410 → 21쪽. 짧은 판(+37자)+19쪽 축약 Y409(−28)·Y410(−39) B14 → 20쪽; 짧은 판 단독(B10) 21쪽",
 "E-C07T3-04": "미시험. 표 11 387행 E-C13T3-06(+7자, B13 20쪽)과 같은 사실 — 택1",
}
for r in rows: r["test_build"] = TB.get(r["id"], "")
cols = ["id", "errata_line", "scope", "v14_line", "in_manuscript", "conflicts", "need", "delta_chars", "priority", "test_build", "note"]
with open(sys.argv[2], "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
from collections import Counter
print(len(rows), Counter(r["scope"] for r in rows))
for r in rows:
    if r["scope"] in ("body", "figure", "appendix"): print(r["id"], r["v14_line"], r["in_manuscript"], r["conflicts"], r["delta_chars"], r["priority"])
EOF_MARK = None
