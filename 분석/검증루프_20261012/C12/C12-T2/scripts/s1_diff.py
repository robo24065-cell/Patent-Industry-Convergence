# C12-T2 s1: v11→v12, v12→v13 변경 블록 추출(행 단위) + 바뀐 숫자 토큰
import sys, json, difflib, re, hashlib
sys.stdout.reconfigure(encoding="utf-8")
D = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\drafts"
def rd(v):
    b = open(D + "\manuscript_for_%s.md" % v, "rb").read()
    return hashlib.sha1(b).hexdigest(), b.decode("utf-8").replace("\r\n", "\n").split("\n")
NUM = re.compile(r"[+−\-]?\d[\d,]*(?:\.\d+)?%?p?")
out = []
for a, b in [("v11", "v12"), ("v12", "v13")]:
    ha, A = rd(a); hb, B = rd(b)
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal": continue
        for k in range(max(i2 - i1, j2 - j1)):
            la = A[i1 + k] if i1 + k < i2 else ""
            lb = B[j1 + k] if j1 + k < j2 else ""
            # 행 안 차이 조각
            cs = difflib.SequenceMatcher(None, la, lb, autojunk=False)
            frags = [(la[x1:x2], lb[y1:y2]) for o, x1, x2, y1, y2 in cs.get_opcodes() if o != "equal"]
            na = set(NUM.findall(la)); nb = set(NUM.findall(lb))
            out.append(dict(step=f"{a}->{b}", line_old=i1 + k + 1, line_new=j1 + k + 1, op=op,
                            chars_old=len(la), chars_new=len(lb), d_chars=len(lb) - len(la),
                            nums_removed=sorted(na - nb), nums_added=sorted(nb - na),
                            frags=[(x[:120], y[:120]) for x, y in frags][:12]))
    print(a, b, ha[:8], hb[:8], "changed lines:", sum(1 for o in out if o["step"] == f"{a}->{b}"))
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for o in out: print(o["step"], o["line_new"], o["d_chars"], o["nums_removed"], "->", o["nums_added"])
