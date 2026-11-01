# C18-T3 s1: ERRATA 항목 파싱 + 반영 기록 전개 + v19 자동 문자열 대조 (읽기 전용)
# 사용: python -I s1_parse.py
import re, json, sys, hashlib
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = AN + r"\검증루프_20261012"; R = AN + r"\보고서_20261012"
W = L + r"\C18\C18-T3"
er = open(L + r"\ERRATA.txt", encoding="utf-8").read().replace("\r\n", "\n").split("\n")
ms_raw = open(R + r"\manuscript_v3.md", "rb").read()
assert hashlib.sha1(ms_raw).hexdigest().startswith("b316df8e")
ms = ms_raw.decode("utf-8").replace("\r\n", "\n").split("\n")
v18 = open(R + r"\drafts\manuscript_for_v18.md", encoding="utf-8").read().replace("\r\n", "\n").split("\n")

# ---- 1. 항목 블록 ----
IDRE = re.compile(r"E-C\d\d[A-Za-z0-9\-~가-힣]*")
items = []          # 블록 단위
section = None
cur = None
def expand(head):
    """머리 ID 문자열 → 개별 ID 목록"""
    out = []
    # 범위 E-C03T3-09~13
    m = re.match(r"(E-C\d\d[A-Za-z0-9\-]*?-)(\d\d)~(\d\d)$", head)
    if m:
        return [m.group(1) + "%02d" % k for k in range(int(m.group(2)), int(m.group(3)) + 1)]
    return [head]
for i, l in enumerate(er, 1):
    if l.startswith("====") or l.startswith("================"):
        mm = re.search(r"C(\d\d)", l)
        if mm: section = "C" + mm.group(1)
        cur = None; continue
    if l.startswith("---- [메인 세션 반영 기록"):
        cur = None; continue
    if l.startswith("--- ") or l.startswith("반영 우선순위") or l.startswith("참고") or l.startswith("(") or l.startswith("[") or l.startswith("공통") or l.startswith("공식 성적") or l.startswith("제출 판정") or l.startswith("반영하지 않는 것") or l.startswith("대체 관계") or l.startswith("지면") or l.startswith("!!") or l.startswith("하지 말 것") or l.startswith("반박으로") or l.startswith("이 묶음") or l.startswith("시험 빌드") or l.startswith("쪽수 주의") or l.startswith("검증자") or l.startswith("해외 FREE") or l.startswith("반영됨") or l.startswith("미반영") or l.startswith("지면 조정") or l.startswith("보류") or l.startswith("시도 후") or l.startswith("facts.csv(읽기"):
        cur = None
        # '(기존) E-…' 줄은 기존 ID 갱신 블록
        m = re.match(r"\(기존\)\s*(E-C\d\d[A-Za-z0-9\-]*)(.*)$", l)
        if m:
            ids = IDRE.findall(l.split("[")[0]) or [m.group(1)]
            cur = dict(ids=ids, head=m.group(1), line=i, section=section, text=l.strip(), kind="기존재기재")
            items.append(cur)
        continue
    m = re.match(r"^(E-C\d\d[^\s\[]*)(.*)$", l)
    if m:
        headtok = m.group(1)
        rest = m.group(2)
        ids = []
        # 'E-C10T3-02 = E-C10T4-01', 'E-C16T1-02 + E-C16T2-01'
        mm = re.match(r"^\s*([=+])\s*(E-C\d\d[A-Za-z0-9\-]*)", rest)
        base = re.sub(r"\((수정|반박 반영)\)$", "", headtok)
        ids += expand(base)
        if mm: ids.append(mm.group(2))
        kind = "갱신" if re.match(r"^\s*→", rest) else "제안"
        cur = dict(ids=ids, head=headtok, line=i, section=section, text=l.strip(), kind=kind)
        items.append(cur); continue
    if cur is not None and l.strip():
        cur["text"] += "\n" + l.strip()
    elif not l.strip():
        cur = None
print("blocks", len(items))

# ---- 2. 반영 기록 전개 ----
recs = []
rec_no = 0
for i, l in enumerate(er, 1):
    if l.startswith("---- [메인 세션 반영 기록"):
        rec_no += 1
        mm = re.search(r"\((\d+)\)", l); n = int(mm.group(1)) if mm else 1
        body = []
        j = i
        while j < len(er) and not er[j].startswith("====") and er[j].strip():
            body.append((j + 1, er[j])); j += 1
        recs.append(dict(no=n, line=i, head=l, body=body))
def expand_list(txt, cyc):
    """'E-C01-02, 04–13, 15' 같은 목록 → ID들"""
    out = []
    prefix = None
    for tok in re.split(r"[,，·、]|\s및\s|\s", txt):
        tok = tok.strip().strip("()")
        m = re.match(r"^(E-C\d\d[A-Za-z0-9\-]*?)(\d\d|X\d|D\d|F\d?|B\d|S)$", tok)
        if m:
            prefix = m.group(1); out.append(tok); continue
        m = re.match(r"^(\d\d|X\d)[–\-~](\d\d|X\d)$", tok)
        if m and prefix:
            a, b = m.group(1), m.group(2)
            if a[0] == "X":
                for k in range(int(a[1:]), int(b[1:]) + 1): out.append(prefix + "X%d" % k)
            else:
                for k in range(int(a), int(b) + 1): out.append(prefix + "%02d" % k)
            continue
        m = re.match(r"^(\d\d|X\d|D\d|F\d?)$", tok)
        if m and prefix:
            out.append(prefix + tok); continue
    return out
REC = {}   # id -> list of (rec_no, category, line, snippet)
for r in recs:
    for ln, l in r["body"]:
        m = re.match(r"^(반영됨[^:：]*|미반영[^:：]*|보류[^:：]*|시도 후 되돌림[^:：]*|지면 조정[^:：]*)[:：](.*)$", l)
        if not m: continue
        cat = m.group(1); txt = m.group(2)
        catn = "반영됨" if cat.startswith("반영됨") else ("미반영" if cat.startswith("미반영") else ("보류" if cat.startswith("보류") else ("되돌림" if cat.startswith("시도") else "지면조정")))
        # 괄호 속 설명 안의 ID도 잡되, 괄호 밖 머리 ID를 우선
        # 단순화: 모든 E-ID 토큰과 그 뒤 '·NN' 연속 번호를 전개
        for seg in re.finditer(r"(E-C\d\d[A-Za-z0-9\-]*?)(\d\d|X\d|D\d|F\d?)((?:\s*[·,]\s*(?:\d\d|X\d)|\s*[–~]\s*(?:\d\d|X\d))*)", txt):
            ids = expand_list(seg.group(0).replace("·", ",").replace("~", "–"), r["no"])
            for x in ids:
                REC.setdefault(x, []).append(dict(rec=r["no"], cat=catn, line=ln, snip=txt[max(0, seg.start() - 0):seg.start() + 120]))
# '반영됨(C01 재검 확인·부분): E-C01-02(…), 04(…), 05(…)' 형식: 쉼표 뒤 맨 번호
for r in recs:
    for ln, l in r["body"]:
        if not l.startswith("반영됨"): continue
        # 괄호 제거 후 토큰화
        depth = 0; flat = []
        for ch in l:
            if ch == "(": depth += 1; continue
            if ch == ")": depth -= 1; continue
            if depth == 0: flat.append(ch)
        flat = "".join(flat)
        prefix = None
        for tok in re.split(r",\s*", flat.split(":", 1)[-1] if ":" in flat else flat):
            tok = tok.strip()
            m = re.match(r"^(E-C\d\d[A-Za-z0-9\-]*?)(\d\d|X\d)$", tok.split()[0] if tok else "")
            if m:
                prefix = m.group(1); x = tok.split()[0]
            else:
                m2 = re.match(r"^(\d\d|X\d|T\d-\d\d)", tok)
                if not (m2 and prefix): continue
                t = m2.group(1)
                x = (prefix + t) if not t.startswith("T") else re.sub(r"T\d-$", "", prefix) + t
            lst = REC.setdefault(x, [])
            if not any(e["rec"] == r["no"] and e["cat"] == "반영됨" for e in lst):
                lst.append(dict(rec=r["no"], cat="반영됨", line=ln, snip=tok[:120]))

# ---- 3. 인용 문자열 + v19 자동 대조 ----
def grams(s, n=6):
    s = re.sub(r"\s+", " ", s)
    return {s[k:k + n] for k in range(max(1, len(s) - n + 1))} if len(s) >= n else {s}
LG = [grams(x) for x in ms]
LG18 = [grams(x) for x in v18]
def best(s, lg=LG):
    g = grams(s)
    if not g: return (0, 0.0)
    bl, bf = 0, 0.0
    for k, lgk in enumerate(lg):
        f = len(g & lgk) / len(g)
        if f > bf: bl, bf = k + 1, f
    return bl, round(bf, 3)
QRE = re.compile(r"'([^']{4,})'")
out = []
for b in items:
    t = b["text"]
    pairs = []
    # "'A' → 'B'" 쌍
    for m in re.finditer(r"'([^']{2,})'\s*→\s*(?:[^']{0,40}?)'([^']{2,})'", t):
        pairs.append((m.group(1), m.group(2)))
    news = [p[1] for p in pairs]
    # '수정:'/'추가:' 뒤 인용
    for m in re.finditer(r"(수정|추가|수정\(최소, 0자\)|수정\(권장\))[^:：\n]*[:：]\s*'([^']{4,})'", t):
        news.append(m.group(2))
    olds = [p[0] for p in pairs]
    for m in re.finditer(r"현재[^:：\n]*[:：]\s*'([^']{4,})'", t):
        olds.append(m.group(1))
    allq = QRE.findall(t)
    res_new = [(q[:60], *best(q)) for q in news[:6]]
    res_old = [(q[:60], *best(q)) for q in olds[:6]]
    res_new18 = [best(q, LG18) for q in news[:6]]
    for x in b["ids"]:
        out.append(dict(id=x, block_line=b["line"], section=b["section"], kind=b["kind"], head=b["head"], text=t,
                        new_q=res_new, old_q=res_old, new_q_v18=res_new18, n_quotes=len(allq),
                        rec=REC.get(x, [])))
json.dump(dict(items=out, recs=[dict(no=r["no"], line=r["line"]) for r in recs]), open(W + r"\cache\s1_items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ids = sorted({o["id"] for o in out})
print("unique ids", len(ids), "blocks->ids", len(out))
print("records", [r["no"] for r in recs])
noreq = sorted(x for x in REC if x not in ids)
print("반영기록에만 있는 ID", noreq)
from collections import Counter
print(Counter(o["section"] for o in out))
