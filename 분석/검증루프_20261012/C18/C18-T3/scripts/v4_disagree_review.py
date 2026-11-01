# C18-T3 verify 4: v1b 전수(294) 불일치 91건 유형 분류 + 수동 검토 결과 기록(PREREG 변경 기록 (2)(c)).
import csv, io, sys, json, collections, hashlib
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T3"
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
MS = open(R + r"\manuscript_v3.md", encoding="utf-8").read(); assert hashlib.sha1(MS.encode()).hexdigest().startswith("b316df8e")
rows = [r for r in csv.DictReader(io.StringIO(open(W + r"\verify\v1b_status_all.csv", encoding="utf-8-sig").read())) if r["일치"] == "0"]
# 수동 검토(원문 블록·반영 기록·v19 문자열 직접 확인). 값: (판정, 확인 문자열 또는 사유)
MANUAL = {
 "E-C15T4-A1": ("본판정 유지(부분)", "값 칸 반영, 출처 칸 '; …v5_frac.json, refute_result.txt' 없음"),
 "E-C17T3-04": ("본판정 유지(미반영)", "v19 143행 '법인 연결표를 빼면'·'국민연금 사업장은' 그대로, 'DART 업종을 빼면' 0건"),
 "E-C01-03": ("본판정 유지(반영됨)", "기록 (1) 반영됨, 168행 '순위상관 0.45, 5개국 중 최저' 축약판"),
 "E-C08T2-09": ("본판정 유지(미반영)", "'DART·KIND는 CI 0' 0건(구절 휴리스틱 오탐)"),
 "E-C11T3-01": ("본판정 유지(미반영)", "'2026-09-25판' 0건"),
 "E-C02-T3-12": ("본판정 유지(부분)", "NTIS-SRC '4.4'·LINK-NPS '2.5' 남음"),
 "E-C13T1-01": ("본판정 유지(미반영, 경계)", "'(직권)' 0건; 병합판의 E-C13T3-09 몫만 있음"),
 "E-C13T2-02": ("본판정 유지(미반영)", "'0 밑' 0건; 독립 판정자가 '현재' 인용을 수정 인용으로 셈"),
 "E-C10T4-03": ("본판정 유지(미반영)", "Marx & Fuegi 2022·JEMS 0건"),
 "E-C10T4-05": ("본판정 유지(미반영), 글자 수 정정", "'특허청, 2013' 0건; 429행 '통계청' 소멸 → +18"),
 "E-C05T3-05": ("본판정 유지(미반영)", "678행은 '중복 표시' 언급이지 반박 아님; fig09 CSV는 원고 밖 그림 자료"),
 "E-C09T1-01": ("본판정 유지(미반영)", "865행은 C13 우선순위 경합 언급이지 반박 아님"),
 "E-C09T3-01": ("본판정 유지(미반영)", "865행 같은 사유"),
 "E-C16T2-06": ("본판정 유지(미반영)", "'광공업지수는' 0건(79행 '광공업지수·'에 구절 오탐)"),
 "E-C16T2-04": ("본판정 유지(미반영)", "'N6 미연결 기업은 결과 0' 0건"),
 "E-C16T3-S": ("본판정 유지(미반영)", "'KR 출원(실용신안 포함)' 0건"),
 "E-C17T1-01b": ("본판정 유지(미반영)", "'T1 법인·위험쌍' 0건; 독립 판정자는 b 접미 ID 블록을 못 찾음"),
 "E-C17T3-01": ("본판정 유지(반영됨)", "기록 (17) 최소판 반영됨, 권장판은 같은 기록의 미반영 목록(하위 판)"),
 "E-C17T1-B1": ("본판정 유지(반영됨)", "기록 (17) '부록 C17-T1-NEWC… 행 추가', 536행 존재"),
 "E-C17T2-B1": ("본판정 유지(반영됨)", "537행 존재"),
 "E-C17T3-B1": ("본판정 유지(반영됨)", "538행 존재"),
 "E-C13T3-10": ("본판정 유지(반영됨)", "기록 (16) 지면 조정으로 해당 구절 삭제 = 제안(삭제)대로 반영"),
 "E-C13T3-11": ("본판정 유지(반영됨)", "같은 사유(삭제 제안이 실행됨)"),
 "E-C14T3-04": ("본판정 유지(폐기·대체)", "기록 (14) '(E-C14T3-01과 중복)' 택1"),
 "E-C03T1-03": ("본판정 유지(폐기·대체)", "기록 (17) '기각 확정'"),
 "E-C01-X2": ("본판정 유지(부분)", "'| C-LAND-prov | 표지·요약, 4.4 |' 1건 남음, 다른 절 열은 정리됨"),
 "E-C03T3-10": ("본판정 유지(미반영)", "'| CIRC-AGREE-CC | 표지·요약, 1.3, 3.6.1, 3.6.4 |' 그대로"),
 "E-C03T3-11": ("본판정 유지(미반영)", "'| C-LAND-prov | 표지·요약, 4.4 |' 그대로"),
 "E-C03T3-12": ("본판정 유지(반영됨)", "'공식 0.135 [0.091, 0.179]' 1건"),
 "E-C03T3-13": ("본판정 유지(반영됨)", "'순열 p 0.08–0.26' 1건"),
 "E-C10T3-06": ("본판정 유지(폐기·대체)", "기록 (10) 보류 → C11 대체판 반영('순위상관은 0.19–0.37')"),
 "E-C13T3-04": ("본판정 유지(반영됨)", "'물가·산업연관표·광공업지수·NTIS' 1건"),
 "E-C13T3-통합보충": ("본판정 유지(미반영)", "480행 그대로; 경합 해소는 E-C03T3-11안"),
 "E-C15T1-02": ("본판정 유지(부분)", "NTIS-T3-SUM 행에 's4_horizon' 출처 0건"),
}
def typ(r):
    a, b = r["본판정"], r["독립판정"]
    if b == "판정불가": return "D 독립 판정자 블록 없음(복합·접미 ID)"
    if a == "반영됨" and b in ("부분", "미반영", "폐기·대체"): return "A 기록 지정 축약·대안판/삭제 반영을 독립 판정자가 인식 못함"
    if a == "폐기·대체": return "B 뒤 기록의 대체·소멸·제안 아님을 독립 판정자가 인식 못함"
    if a in ("미반영", "부분") and b in ("반영됨", "부분"): return "C 구절 휴리스틱 오탐(수정 문자열 직접 검색 0건)"
    return "E 기타"
out = []
for r in rows:
    m = MANUAL.get(r["ID"])
    out.append(dict(ID=r["ID"], 범위=r["범위"], 본판정=r["본판정"], 독립판정=r["독립판정"], 유형=typ(r), 수동검토="Y" if m else "N",
                    수동판정=m[0] if m else "", 확인=m[1] if m else "", 독립근거=r["독립근거"][:100], 본근거=r["본근거"][:120]))
with open(W + r"\verify\v4_disagreement_review.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
# 유형 A·B 전수: 본판정 근거에 적힌 'N행: 문자열'이 v19 그 줄에 실제로 있는지 직접 확인(본판정 증거의 존재 검사)
import re
ML = MS.split("\n")
full = {r["ID"]: r for r in csv.DictReader(io.StringIO(open(W + r"\errata_status_v18.csv", encoding="utf-8-sig").read()))}
for o in out:
    g = full[o["ID"]]["근거"]
    ev = re.findall(r"(\d+)행: (.+?)(?= / \d+행: | ; |$)", g)
    hit = [(int(n), s.strip()) for n, s in ev if 0 < int(n) <= len(ML) and s.strip() in ML[int(n) - 1]]
    miss = [(int(n), s.strip()) for n, s in ev if not (0 < int(n) <= len(ML) and s.strip() in ML[int(n) - 1])]
    absent_ok = g.startswith("없음 확인:") and g.split("없음 확인:")[1].split(";")[0].strip() not in MS
    o["근거문자열_v19확인"] = f"{len(hit)}/{len(ev)}" + (" (없음 확인 성립)" if absent_ok else "") + (f" 불일치:{miss[:1]}" if miss else "")
c = collections.Counter(o["유형"] for o in out); cm = collections.Counter(o["유형"] for o in out if o["수동검토"] == "Y")
s = dict(n_disagree=len(out), by_type=c, manual_by_type=cm, manual_n=sum(1 for o in out if o["수동검토"] == "Y"),
         manual_changed_main=sum(1 for o in out if o["수동검토"] == "Y" and "유지" not in o["수동판정"]))
with open(W + r"\verify\v4_disagreement_review.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
s["근거문자열_확인"] = collections.Counter(("불일치 있음" if "불일치" in o["근거문자열_v19확인"] else ("없음확인" if "없음 확인" in o["근거문자열_v19확인"] else ("근거줄 없음" if o["근거문자열_v19확인"].startswith("0/0") else "전부 확인"))) + "|" + o["유형"][0] for o in out)
s["불일치_목록"] = [(o["ID"], o["근거문자열_v19확인"][:160]) for o in out if "불일치" in o["근거문자열_v19확인"]]
json.dump(s, open(W + r"\verify\v4_disagreement_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(s, ensure_ascii=False, indent=1))
print([k for k in MANUAL if k not in {o["ID"] for o in out}])
