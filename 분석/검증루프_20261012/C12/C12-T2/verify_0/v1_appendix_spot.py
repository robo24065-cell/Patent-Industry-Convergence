# C12-T2 verify_0 v1: 부록 토큰 '일치'(자동 존재 대조) 무작위 15개의 의미 확인 — 원천 파일에서 그 값 주변 문맥(키·열 이름)을 출력해 같은 양인지 사람이 판정
import sys, os, re, random, csv
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = ROOT + r"\분석"; MC = AN + r"\모델경쟁_20260927"
W = AN + r"\검증루프_20261012\C12\C12-T2"
v13 = open(AN + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().split("\n")
def resolve(cell):
    out = []
    for part in re.split(r";\s*", cell):
        p = part.strip().replace("(MC)", MC).replace("(분석)", AN)
        if p.startswith("분석\\") or p.startswith("수집파이프라인\\"): p = ROOT + "\\" + p
        elif p.startswith("보고서_20261012\\"): p = AN + "\\" + p
        elif not re.match(r"^[A-Za-z]:", p) and out: p = os.path.join(os.path.dirname(out[-1]), p)
        p = re.split(r"\s+(?:\(|§|표|fig|합계)", p)[0].strip(); p = re.sub(r"\s*\(.*$", "", p)
        out.append(p)
    return [x for x in out if os.path.isfile(x)]
rows = list(csv.DictReader(open(W + r"\appendix_tokens.csv", encoding="utf-8-sig")))
cand = [r for r in rows if r["final_verdict"] == "일치" and r["src_value"] and r["v13_line"]]
random.seed(20261013)
pick = random.sample(cand, 15)
for r in pick:
    ln = int(r["v13_line"]); cell = [c.strip() for c in v13[ln - 1].strip().strip("|").split("|")][-1]
    ps = resolve(cell); sv = float(r["src_value"])
    ctx = "n/a"
    for p in ps:
        t = open(p, encoding="utf-8-sig", errors="replace").read()
        for cand_s in sorted({repr(sv), ("%g" % sv), str(int(sv)) if sv == int(sv) else repr(sv)}, key=len, reverse=True):
            k = t.find(cand_s)
            if k >= 0:
                ctx = os.path.basename(p) + ": …" + t[max(0, k - 90):k + len(cand_s) + 15].replace("\n", " ") + "…"; break
        if ctx != "n/a": break
    print("## %s %s 토큰 %s (원값 %s) | 원고: %s" % (ln, r["row_id"], r["token"], r["src_value"], r["context"][:80]))
    print("   원천:", ctx[:260])
