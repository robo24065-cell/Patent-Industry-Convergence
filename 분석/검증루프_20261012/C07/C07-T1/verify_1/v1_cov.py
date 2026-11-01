# -*- coding: utf-8 -*-
import re, csv, io, os, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1"
rows = list(csv.reader(io.StringIO(open(os.path.join(H, r"download\ksic_link\ksic_ksco_link_2025Q2.csv"), "rb").read().decode("cp949"))))[1:]
new10 = {r[3] for r in rows if r[0]=="KSIC" and r[1]=="10" and r[2]=="5"}; old9 = {r[5] for r in rows if r[0]=="KSIC" and r[1]=="10" and r[2]=="5"}
new11 = {r[3] for r in rows if r[0]=="KSIC" and r[1]=="11" and r[2]=="5"}
for lab, f, S in (("10th DT_1FS1001", "meta_DT_1FS1001_ITM.json", new10), ("9th DT_1F160622", "meta_DT_1F160622_ITM.json", old9)):
    t = open(os.path.join(H, "download", "meta", f), encoding="utf-8").read()
    objs = re.findall(r'\{([^{}]*)\}', t)
    cnt = collections.Counter(); cs = collections.defaultdict(set)
    for o in objs:
        oid = re.search(r'OBJ_ID:"([^"]*)"', o).group(1); iid = re.search(r'(?<![A-Z_])ITM_ID:"([^"]*)"', o).group(1)
        cnt[oid] += 1
        m = re.fullmatch(r'C?(\d{5})', iid)
        if m: cs[oid].add(m.group(1))
    print(lab, dict(cnt), {k: len(v) for k, v in cs.items()})
    for k, v in cs.items():
        man = {c for c in v if "10" <= c[:2] <= "34"}
        print("  ", k, "manuf 5-digit", len(man), "in link", len(man & S), "missing", sorted(man - S)[:20])
# check 11th codes count vs official KSIC11 (1,205)
print("11th new codes", len(new11))
