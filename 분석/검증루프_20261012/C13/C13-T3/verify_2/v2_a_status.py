# verify_2 (a): ERRATA 블록 독립 재파싱 + 반영 기록 (1)-(12) 전문에서 ID 자동 추출 -> s2 수기 목록과 대조
import re, json, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012"
T3 = BASE + r"\C13\C13-T3"
L = open(BASE + r"\ERRATA.txt", encoding="utf-8").read().splitlines()
# 1) 헤더: 줄 시작이 E-C 로 시작
hdr = [(i + 1, re.match(r"^(E-C[0-9A-Za-z\-~]+)", l).group(1)) for i, l in enumerate(L) if re.match(r"^E-C\d", l)]
ind = [(i + 1, re.match(r"^\s+(E-C[0-9A-Za-z\-~]+)", l).group(1)) for i, l in enumerate(L) if re.match(r"^\s+E-C\d", l)]
print("줄머리 E-C 헤더:", len(hdr), "고유", len(set(h[1] for h in hdr)), "| 들여쓴 E-C 줄:", len(ind))
dup = [k for k, v in collections.Counter(h[1] for h in hdr).items() if v > 1]
print("중복 헤더 ID:", dup)
# 2) 반영 기록 블록 전문
recs = [i for i, l in enumerate(L) if "메인 세션 반영 기록" in l and l.startswith("----")]
reflected, notref = set(), set()
idre = re.compile(r"E-C\d\d(?:-T\d|T\d|-)?-?[A-Z]?\d*[0-9A-Za-z]*")
def expand(text):
    # "E-C01-02(...), 04(...), 05" 같은 이어쓰기 확장
    out = set()
    toks = re.split(r"[,，、]\s*|\s+|/", text)
    return out
for k, s in enumerate(recs):
    e = recs[k + 1] if k + 1 < len(recs) else len(L)
    # 다음 섹션 헤더(====)에서 끊기
    for j in range(s + 1, e):
        if L[j].startswith("===="): e = j; break
    blk = L[s:e]
    for l in blk:
        tag = None
        if l.startswith("반영됨"): tag = "R"
        elif l.startswith("미반영") or l.startswith("보류"): tag = "N"
        if not tag: continue
        # 괄호 안 제거 후 ID와 이어쓰기 번호 추출
        x = l
        while True:
            y = re.sub(r"\([^()]*\)", "", x)
            if y == x: break
            x = y
        last = None
        for m in re.finditer(r"(E-C\d\d(?:-T\d-|T\d-|-))([0-9A-Z][0-9A-Za-z~]*)|(?<=[,·、 ])(\d\d|X\d|D\d)(?=[,·、 .]|$)", x):
            if m.group(1):
                last = m.group(1); idv = last + m.group(2)
            elif last:
                idv = last + m.group(3)
            else:
                continue
            (reflected if tag == "R" else notref).add(idv)
print("반영 기록 블록 수:", len(recs))
# s2 수기 목록 재구성
import importlib.util, io, contextlib
src = open(T3 + r"\scripts\s2_status.py", encoding="utf-8").read()
src_R = src.split("EXCL = {")[0].replace("d = json.load(open(sys.argv[1], encoding=\"utf-8\"))", "")
ns = {}
exec(src_R, ns)
R2 = ns["R"]
print("s2 수기 반영 집합:", len(R2), "| 자동 추출 반영 집합:", len(reflected))
print("자동엔 있고 s2엔 없음:", sorted(reflected - R2))
print("s2엔 있고 자동엔 없음:", sorted(R2 - reflected))
ids = set(h[1] for h in hdr)
print("자동 반영 ID 중 ERRATA 헤더에 없는 것:", sorted(r for r in reflected if r not in ids))
print("자동 '미반영/보류'이면서 '반영됨'에도:", sorted(reflected & notref))
json.dump({"hdr": hdr, "ind": ind, "reflected_auto": sorted(reflected), "notref_auto": sorted(notref)}, open(T3 + r"\verify_2\v2_a_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
