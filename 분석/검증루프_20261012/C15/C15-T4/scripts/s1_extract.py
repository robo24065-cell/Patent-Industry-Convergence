# C15-T4 s1: v16(현재 manuscript_v3.md)에서 2·3장 본문·박스·캡션·표 행, 부록 출처표 값 셀의 판정 단위를 추출하고
# C13-T3 claim_map(v14 줄)과 겹치는 단위를 제외한다. 원고·claim_map은 읽기만.
import csv, re, json, hashlib, difflib, sys
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MS = B + r"\보고서_20261012\manuscript_v3.md"
V14 = B + r"\보고서_20261012\drafts\manuscript_for_v14.md"
V15 = B + r"\보고서_20261012\drafts\manuscript_for_v15.md"
CM = B + r"\검증루프_20261012\C13\C13-T3\claim_map.csv"
OUT = B + r"\검증루프_20261012\C15\C15-T4\cache"

raw = open(MS, "rb").read(); sha = hashlib.sha1(raw).hexdigest()
L = raw.decode("utf-8").split("\n")
L14 = open(V14, encoding="utf-8").read().split("\n")
L15 = open(V15, encoding="utf-8").read().split("\n")

def linemap(a, b):  # a 줄번호(1기반) -> b 줄번호
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1): m[i1 + k + 1] = j1 + k + 1
    return m
m14 = linemap(L14, L)
changed15 = set()
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, L15, L, autojunk=False).get_opcodes():
    if tag != "equal": changed15.update(range(j1 + 1, j2 + 1))

def find(prefix):
    for i, t in enumerate(L, 1):
        if t.startswith(prefix): return i
    raise KeyError(prefix)
s2, s3, s4 = find("# 2. "), find("# 3. "), find("# 4. ")
ap, memo = find("# [부록"), find("# [편집 메모")

LEX = ["유지", "견고", "영향 없", "영향은 없", "영향이 없", "조건부", "확인", "같다", "같고", "같은 방향", "같거나", "같음", "차이가 없", "차이 없",
       "없다", "없었", "없음", "우위", "앞서", "앞섰", "낫", "높다", "높고", "더 높", "낮다", "낮아", "작다", "작고", "크게", "민감", "경계", "부분",
       "불명", "가능성 낮", "둔감", "재현", "통과", "넘지 못", "이득", "판정", "유효", "약해", "남는다", "남고", "사라", "회복", "줄어", "갈리", "갈린다",
       "반대", "불리", "일치", "보수적", "비슷", "안정", "과대", "미검증", "미재계산", "사후", "인과", "주장하지", "순환", "대부분", "뿐", "유리",
       "의존", "포함된다", "기운다", "치우", "미재현", "→ 같다", "최저", "최고", "가장",
       "빼도", "바꿔도", "바로잡아도", "되살려도", "보정해도", "재채점해도", "골라도", "좁혀도", "재가중해도", "넘", "크다", "하락", "순증분"]

def sents(t):
    t = t.replace("**", "")
    parts = re.split(r"(?<=[가-힣\)\]%”’\"])\.\s+", t.strip())
    return [p.strip() for p in parts if p.strip()]

def sec_of(i):
    cur = ""
    for j in range(i, 0, -1):
        if L[j - 1].startswith("#"): return L[j - 1].lstrip("# ").strip()[:30]
    return cur

units = []
for i in sorted(x for x in changed15 if x < s2):  # 2장 앞 변경 줄(1장)도 감사
    for k, s_ in enumerate(sents(L[i - 1])): units.append(dict(line=i, kind="본문(1장 변경 줄)", text=s_, sub=k + 1))
for i in range(s2, s4):
    t = L[i - 1]
    if not t.strip() or t.startswith("#") or t.startswith("[그림") or t.startswith("---") or re.match(r"^\|[-|]+\|$", t): continue
    if t.startswith("|"):
        cells = [c.strip() for c in t.strip().strip("|").split("|")]
        if L[i] .startswith("|---") if i < len(L) else False: continue  # 표 머리행
        units.append(dict(line=i, kind="표 행", text=" | ".join(cells)))
    elif t.startswith("그림 "):
        units.append(dict(line=i, kind="캡션", text=t.strip()))
    elif t.startswith("**표 "):
        continue
    else:
        kind = "박스" if 271 <= i <= 289 and sec_of(i).startswith("3.5") else "본문"
        for k, s in enumerate(sents(t)): units.append(dict(line=i, kind=kind, text=s, sub=k + 1))
for i in range(ap, memo):
    t = L[i - 1]
    if t.startswith("|") and not t.startswith("|---") and not t.startswith("| id"):
        cells = [c.strip() for c in t.strip().strip("|").split("|")]
        if len(cells) >= 4: units.append(dict(line=i, kind="부록 셀", text=cells[2], rowid=cells[0]))

cm = list(csv.DictReader(open(CM, encoding="utf-8-sig")))
def lcs(a, b):
    a = re.sub(r"\s", "", a); b = re.sub(r"\s", "", b)
    s = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return s.size
def nums(s): return set(re.findall(r"\d+(?:\.\d+)?", s)) - {"1", "2", "3", "4", "5", "0"}
for c in cm:
    c["v16_line"] = m14.get(int(c["line"]))
for u in units:
    u["lex"] = ";".join(w for w in LEX if w in u["text"])
    u["changed_v15_v16"] = u["line"] in changed15
    cov = []
    for c in cm:
        if c["v16_line"] != u["line"]: continue
        a = c["anchor"].strip()
        if a and a in u["text"].replace("**", ""): cov.append(c["claim_id"] + ":anchor"); continue
        if lcs(c["claim"], u["text"]) >= 8: cov.append(c["claim_id"] + ":lcs"); continue
        if len(nums(c["claim"]) & nums(u["text"])) >= 2: cov.append(c["claim_id"] + ":num")
    u["covered_by"] = ",".join(cov)
    u["sec"] = sec_of(u["line"])
    u["region"] = "부록" if u["line"] >= ap else ("1장" if u["line"] < s2 else ("2장" if u["line"] < s3 else "3장"))
cand = [u for u in units if (u["lex"] and not u["covered_by"]) or u["changed_v15_v16"]]
for n, u in enumerate(cand, 1): u["uid"] = f"U{n:03d}"
cols = ["uid", "line", "region", "sec", "kind", "changed_v15_v16", "covered_by", "lex", "text"]
with open(OUT + r"\units_all.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols + ["sub", "rowid"], extrasaction="ignore"); w.writeheader(); [w.writerow(u) for u in units]
with open(OUT + r"\candidates_frozen.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols + ["rowid"], extrasaction="ignore"); w.writeheader(); [w.writerow(u) for u in cand]
unm = [c["claim_id"] for c in cm if c["v16_line"] is None]
st = dict(sha1=sha, lines=len(L), sections=dict(s2=s2, s3=s3, s4=s4, appendix=ap, memo=memo), changed_v15_v16=sorted(changed15),
          units=len(units), units_lex=sum(1 for u in units if u["lex"]), covered=sum(1 for u in units if u["covered_by"]),
          candidates=len(cand), by_region={r: sum(1 for u in cand if u["region"] == r) for r in ("1장", "2장", "3장", "부록")}, changed_units=sum(1 for u in cand if u["changed_v15_v16"]),
          claim_unmapped=unm, claims_in_scope_lines=sum(1 for c in cm if c["v16_line"] and s2 <= c["v16_line"] < s4))
json.dump(st, open(OUT + r"\extract_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False))
