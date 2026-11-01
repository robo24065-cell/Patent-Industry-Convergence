# C12-T2 s8: 미반영 선택 항목·제안의 지면 추정(v13 PDF 실측 줄당 글자 수) + 시험 빌드 실측 결과 병기
import sys, json, re, statistics, fitz, csv
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T2"
d = fitz.open(R + r"\보고서_초안_v13.pdf")
TB = json.load(open(W + r"\build_test\test_builds.json", encoding="utf-8"))
items = [  # id, 원고 줄, PDF 검색 앵커(줄바꿈 없는 짧은 조각), 추가 글자 수, 시험 빌드 키
 ("E-C11T2-07(단독)", 73, "KR 라벨을 상속한다", 68, "C1"),
 ("E-C11T1-03", 410, "이후 96%", 113, "C2"),
 ("E-C11T3-01", 430, "Reliance on Science(Marx", 26, "C3"),
 ("E-C11T3-02", 205, "과학 의존도를 붙였다", 104, "C4"),
 ("168행 최빈 동률 단서(E-C11T2-02 선택)", 168, "0.19–0.37이며", 27, "C5"),
 ("제안 P73(상속 정의·연결 민감도)", 73, "KR 라벨을 상속한다", 42, "F1(묶음)"),
 ("제안 P143('각각')", 143, "0.0022 이하", 3, "F1(묶음)"),
 ("제안 P395('skill')", 395, "0.487 [0.431", 6, "F1(묶음)"),
 ("제안 P410('동명 복원')", 410, "N6(미등록 포함)", 6, "F1(묶음)"),
 ("묶음 F2(P73+E-C11T2-07 병합, P143, P395, P410, E-C11T3-01)", 0, "", 119, "F2"),
]
def para_lines(anchor):
    for pno in range(d.page_count):
        pg = d[pno]
        hits = pg.search_for(anchor)
        if not hits: continue
        r = hits[0]
        L = []
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"])
                if t.strip(): L.append((l["bbox"], t))
        L.sort(key=lambda x: (round(x[0][1], 0), x[0][0]))
        # 앵커 줄
        k = next(i for i, (bb, t) in enumerate(L) if fitz.Rect(bb).intersects(r))
        x0 = L[k][0][0]
        pit = sorted(L[i + 1][0][1] - L[i][0][1] for i in range(len(L) - 1) if 5 < L[i + 1][0][1] - L[i][0][1] < 30)
        base = pit[len(pit) // 4] if pit else 15
        lo = k
        while lo > 0 and 0 < L[lo][0][1] - L[lo - 1][0][1] <= base * 1.12 and abs(L[lo - 1][0][0] - x0) < 30: lo -= 1
        hi = k
        while hi + 1 < len(L) and 0 < L[hi + 1][0][1] - L[hi][0][1] <= base * 1.12 and abs(L[hi + 1][0][0] - x0) < 30: hi += 1
        seg = L[lo:hi + 1]
        return pno + 1, [t for bb, t in seg], [bb[2] - bb[0] for bb, t in seg], None
    return None, [], [], None
out = []
for iid, ln, anc, add, key in items:
    if not anc:
        t = TB.get(key, {})
        out.append(dict(item=iid, line="", page="", lines_in_block="", chars_per_line="", last_line_chars="", added_chars=add, est_added_lines="", build=key,
                        build_pages=t.get("pages"), build_last_bottom=t.get("last_bottom"), build_pages_changed=t.get("pages_differing_from_v13")))
        continue
    p, ls, ws, bb = para_lines(anc)
    if not ls: out.append(dict(item=iid, line=ln, page="앵커 실패")); continue
    full = [len(x) for x, w in zip(ls, ws) if w > 0.9 * max(ws)]
    cpl = statistics.median(full) if full else max(len(x) for x in ls)
    last = len(ls[-1]) if ws[-1] < 0.9 * max(ws) else cpl
    import math
    est = max(0, math.ceil((last + add) / cpl) - 1) if last < cpl else math.ceil(add / cpl)
    t = TB.get(key.split("(")[0], {})
    out.append(dict(item=iid, line=ln, page=p, lines_in_block=len(ls), chars_per_line=cpl, last_line_chars=last, added_chars=add, est_added_lines=est, build=key,
                    build_pages=t.get("pages"), build_last_bottom=t.get("last_bottom"), build_pages_changed=t.get("pages_differing_from_v13")))
with open(W + r"\cache\s8_space.csv", "w", encoding="utf-8-sig", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
for o in out: print(o)
