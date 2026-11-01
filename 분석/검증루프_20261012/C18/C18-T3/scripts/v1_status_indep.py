# C18-T3 verify 1: 상태표 독립 판정(PREREG 8절). s1/s2 스크립트와 캐시를 쓰지 않는다.
# 방법: ERRATA.txt를 다시 읽어 ID 블록을 직접 자르고, 반영 기록 줄('반영됨:'/'미반영:'/'보류'/'기각'/'시도 후 되돌림')을 정규식으로 다시 펴고,
#       블록 안 인용문('…')을 공백 정규화 후 v19 전체 문자열에서 부분 문자열로 직접 찾는다(6-그램 포함률 사용 안 함).
# 표본: errata_status_v18.csv에서 범위가 본문·부록·그림인 행의 ID 목록만 읽어 시드 20261017로 30개 무작위 추출(상태 칸은 판정 후에만 읽음).
import re, csv, io, sys, json, random, hashlib, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = AN + r"\검증루프_20261012"; R = AN + r"\보고서_20261012"; W = L + r"\C18\C18-T3"
ms_raw = open(R + r"\manuscript_v3.md", "rb").read(); assert hashlib.sha1(ms_raw).hexdigest().startswith("b316df8e")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()
MSL = ms_raw.decode("utf-8").split("\n"); MS = norm(" ".join(MSL))
ER = open(L + r"\ERRATA.txt", encoding="utf-8").read().split("\n")
# ---- 블록 ----
blocks = {}   # id -> list of (start_line, text)
cur = None
for i, s in enumerate(ER, 1):
    m = re.match(r"^(E-C[0-9A-Za-z\-]+)", s)
    if m:
        cur = m.group(1).rstrip("-"); blocks.setdefault(cur, []).append([i, s])
    elif cur and (s.startswith("  ") or s.startswith("\t")):
        blocks[cur][-1][1] += "\n" + s
    else:
        cur = None
# ---- 반영 기록 ----
def expand(seg):
    """'E-C13T3-10·11', 'E-C01-02, 04–13', 'E-C09T2-04·07' 같은 표기를 ID 목록으로"""
    out = []; base = None
    for tok in re.split(r"[,，]\s*|\s+및\s+", seg):
        tok = tok.strip()
        for m in re.finditer(r"(E-C\d\d(?:-?T\d)?-?)(\d\d|[A-Z]\d+)((?:[·~–]\d\d)*)", tok):
            base = m.group(1); first = m.group(2); out.append(base + first)
            rest = re.findall(r"([·~–])(\d\d)", m.group(3)); prev = first
            for sep, n in rest:
                if sep in "~–" and prev.isdigit():
                    out += [base + "%02d" % k for k in range(int(prev) + 1, int(n) + 1)]
                else: out.append(base + n)
                prev = n
        m2 = re.fullmatch(r"(\d\d)(?:[–~](\d\d))?", tok.split("(")[0].strip())
        if m2 and base:
            a = int(m2.group(1)); b = int(m2.group(2) or a)
            out += [base + "%02d" % k for k in range(a, b + 1)]
    return out
REC = {}   # id -> list of categories in record order
rec_no = 0
for i, s in enumerate(ER, 1):
    if s.startswith("---- [메인 세션 반영 기록"):
        rec_no += 1; continue
    m = re.match(r"^(반영됨[^:：]*|미반영[^:：]*|보류[^:：]*|기각[^:：]*|시도 후 되돌림[^:：]*|지면 조정[^:：]*)[:：](.*)$", s)
    if m and rec_no:
        cat = re.sub(r"[\s(].*", "", m.group(1))
        for x in expand(m.group(2)): REC.setdefault(x, []).append((rec_no, cat, i))
# ---- 인용 ----
Q = re.compile(r"'([^'\n]{6,}?)'")
def quotes(txt, key):
    out = []
    for ln in txt.split("\n"):
        if re.match(r"\s*(" + key + r")\s*[:：]", ln) or (key == "수정|추가|권장|최소" and "→" in ln):
            part = ln.split("→", 1)[1] if ("→" in ln and key != "현재") else ln
            if key == "현재" and "→" in ln: part = ln.split("→", 1)[0]
            out += [norm(q) for q in Q.findall(part)]
    return out
def judge(i):
    bl = blocks.get(i)
    if not bl: return "판정불가", "ERRATA 블록 없음"
    txt = "\n".join(b[1] for b in bl)
    recs = REC.get(i, [])
    later_kill = []
    for j, (st, t) in [(k, b) for k, b in enumerate(ER, 1) if False]: pass
    # 뒤 바퀴에서의 반박·대체(블록 밖 언급 포함)
    first = bl[0][0]
    for k in range(first + 1, len(ER) + 1):
        s = ER[k - 1]
        if i in s and re.search(r"(기각|반박|대체|철회|폐기|넣지 말|쓰지 말|대상 소멸|중복)", s) and not s.startswith(i):
            later_kill.append(k)
    news = quotes(txt, "수정|추가|권장|최소"); olds = quotes(txt, "현재")
    nf = [q for q in news if q in MS]; of = [q for q in olds if q in MS]
    last = recs[-1][1] if recs else ""
    if last.startswith("반영됨"):
        if news and not nf: return "부분", f"기록 반영됨, 수정 인용 0/{len(news)} 발견"
        return "반영됨", f"기록 {recs[-1][0]} 반영됨; 수정 인용 {len(nf)}/{len(news)}"
    if last.startswith("기각") or later_kill and not nf and (not olds or not of):
        return "폐기·대체", f"뒤 언급 {later_kill[:3]}; 현재 인용 {len(of)}/{len(olds)}"
    if news and len(nf) == len(news): return "반영됨", f"수정 인용 전부 발견 {len(nf)}"
    if nf: return "부분", f"수정 인용 {len(nf)}/{len(news)}"
    if olds and not of: return "폐기·대체", "현재 인용 v19에 없음(대상 소멸)"
    if later_kill: return "폐기·대체", f"뒤 언급 {later_kill[:3]}"
    if last.startswith("지면"): return "폐기·대체", "지면 조정 삭제"
    return "미반영", f"기록 {[(r[0], r[1]) for r in recs]}; 수정 인용 0/{len(news)}"
ST = list(csv.DictReader(io.StringIO(open(W + r"\errata_status_v18.csv", encoding="utf-8-sig").read())))
pool = [r["ID"] for r in ST if r["범위"] in ("본문", "부록", "그림")]
rng = random.Random(20261017); sample = rng.sample(pool, 30)
main = {r["ID"]: r for r in ST}
rows = []
for i in sample:
    v, why = judge(i)
    rows.append(dict(ID=i, 범위=main[i]["범위"], 독립판정=v, 독립근거=why, 본판정=main[i]["상태"], 본근거=main[i]["근거"][:160], 일치=int(v == main[i]["상태"])))
cats = ["반영됨", "부분", "미반영", "폐기·대체"]
n = len(rows); po = sum(r["일치"] for r in rows) / n
pa = {c: sum(1 for r in rows if r["독립판정"] == c) / n for c in cats}; pb = {c: sum(1 for r in rows if r["본판정"] == c) / n for c in cats}
pe = sum(pa[c] * pb[c] for c in cats); kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
with open(W + r"\verify\v1_status_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
res = dict(n=n, agree=round(po, 4), kappa=round(kappa, 4), seed=20261017, pool=len(pool), disagreements=[r["ID"] for r in rows if not r["일치"]],
           dist_indep=pa, dist_main=pb, rec_ids=len(REC), blocks=len(blocks))
json.dump(res, open(W + r"\verify\v1_status_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
for r in rows:
    if not r["일치"]: print(r)
