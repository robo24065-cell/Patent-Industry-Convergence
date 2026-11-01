import csv, os, re, sys, glob
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
LOOP = B + r"\검증루프_20261012"; MC = B + r"\모델경쟁_20260927"
bad = []; n = 0
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")):
    for p in [x.strip() for x in r["result_files"].split(";") if x.strip()]:
        if p.startswith("원고"): continue
        p0 = re.split(r"\s(?=[§(A-Z0-9가-힣])", p)[0] if " " in p else p
        p0 = p0.split(" ")[0]
        full = p0.replace("(MC)", MC) if p0.startswith("(MC)") else os.path.join(LOOP, p0)
        n += 1
        if not glob.glob(full): bad.append((r["claim_id"], p0))
print("checked", n, "missing", len(bad)); [print(b) for b in bad]
