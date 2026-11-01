# -*- coding: utf-8 -*-
# verify_1: 연계표 독립 재계산 (csv 모듈만, 기존 스크립트 미사용)
import csv, json, sys, io, hashlib, collections, os
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
p = os.path.join(H, r"download\ksic_link\ksic_ksco_link_2025Q2.csv")
raw = open(p, "rb").read(); print("sha256", hashlib.sha256(raw).hexdigest(), len(raw))
rows = list(csv.reader(io.StringIO(raw.decode("cp949"))))
hdr, rows = rows[0], rows[1:]
print(hdr, len(rows))
print("kinds/rev/lvl", collections.Counter((r[0], r[1], r[2]) for r in rows).most_common(30))
out = {}
def pairs(rev, lvl="5"):
    return [(r[3].strip(), r[5].strip(), r[4], r[6]) for r in rows if r[0] == "KSIC" and r[1] == rev and r[2] == lvl]
P10 = pairs("10"); P11 = pairs("11")
dup10 = len(P10) - len(set((a, b) for a, b, _, _ in P10)); dup11 = len(P11) - len(set((a, b) for a, b, _, _ in P11))
new10 = {a for a, *_ in P10 if a}; old10 = {b for _, b, _, _ in P10 if b}
new11 = {a for a, *_ in P11 if a}; old11 = {b for _, b, _, _ in P11 if b}
print("10차 rows", len(P10), "new", len(new10), "old(9)", len(old10), "dup", dup10, "blank new/old", sum(1 for a,b,_,_ in P10 if not a), sum(1 for a,b,_,_ in P10 if not b))
print("11차 rows", len(P11), "new", len(new11), "old(10)", len(old11), "dup", dup11)
print("old11(10th codes) == new10 ?", old11 == new10, len(old11 ^ new10))
# 50 codes
W = list(csv.DictReader(open(os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), encoding="utf-8-sig")))
print("W cols", list(W[0].keys())[:12])
K50 = sorted({r["세세분류코드"] for r in W}); print("n K50", len(K50))
o_of = collections.defaultdict(set); n_of = collections.defaultdict(set)
for a, b, _, _ in P10:
    o_of[a].add(b); n_of[b].add(a)
def t9(k):
    olds = o_of.get(k, set())
    if not olds: return "없음"
    split = any(len(n_of[o]) > 1 for o in olds); merge = len(olds) > 1
    if not split and not merge: return "1:1" if list(olds)[0] == k else "1:1코드변경"
    return "분할" if split and not merge else ("통합" if merge and not split else "복합")
T = {k: t9(k) for k in K50}
print("types50", collections.Counter(T.values()))
# 10->11
n11 = collections.defaultdict(set); o11 = collections.defaultdict(set)
for a, b, _, _ in P11:
    n11[b].add(a); o11[a].add(b)
def t11(k):
    news = n11.get(k, set())
    if not news: return "없음"
    merge = any(len(o11[n]) > 1 for n in news); split = len(news) > 1
    if not split and not merge: return "1:1" if list(news)[0] == k else "1:1코드변경"
    return "split" if split and not merge else ("merge" if merge and not split else "NM")
print("types 10->11", collections.Counter(t11(k) for k in K50))
# K34 via group mapping from the existing 3.1.3 definition
sys.path.insert(0, os.path.join(B, r"분석\모델경쟁_20260927\_shared_report\label_validity"))
from lv_common import ksic_to_group
G = {k: ksic_to_group(k) for k in K50}; gc = collections.Counter(G.values())
K34 = [k for k in K50 if gc[G[k]] > 1]; print("K34 n", len(K34))
c34 = collections.Counter(T[k] for k in K34); print("types K34", c34)
non = [k for k in K34 if T[k] in ("분할", "복합", "통합")]; print("K34 split/merge/complex", len(non), non)
noncc = [k for k in K34 if T[k] != "1:1"]; print("K34 any non-identity (incl code change)", len(noncc))
# year counts per code in W to check "20관측"
yrs = collections.Counter(r["세세분류코드"] for r in W); print("rows per code in W (set)", set(yrs.values()), "W cols all", list(W[0].keys()))
# same-number different industry: 10th code k exists as a 9th code with different name
nm9 = {}
for a, b, an, bn in P10: nm9[b] = bn
nm10 = {a: an for a, b, an, bn in P10}
norm = lambda s: s.replace(" ", "").replace(",", "")
coll = [(k, nm10.get(k), nm9[k]) for k in K50 if k in nm9 and norm(nm9[k]) != norm(nm10.get(k, ""))]
print("same code diff name (linktable names):", len(coll)); [print(" ", c) for c in coll]
# also: K50 codes that exist in 9th as a code but mapping o_of[k] doesn't contain k
coll2 = [k for k in K50 if k in old10 and k not in o_of[k]]; print("code exists in 9th but not own predecessor:", coll2)
# KOSIS code coverage
def codes(fp):
    d = json.load(open(fp, encoding="utf-8")); return d
for lab, fp, S in (("10th meta", r"download\meta\meta_DT_1FS1001_ITM.json", new10), ("9th meta", r"download\meta\meta_DT_1F160622_ITM.json", old10)):
    d = json.load(open(os.path.join(H, fp), encoding="utf-8"))
    cs = set()
    for x in d:
        c = str(x.get("ITM_ID", "")).lstrip("C")
        if x.get("OBJ_ID") in ("C2", "B", "ITEM2") or True:
            if len(c) == 5 and c.isdigit() and str(x.get("ITM_ID","")).startswith("C"): cs.add(c)
    print(lab, "5-digit C-codes", len(cs), "in link", len(cs & S), "missing", sorted(cs - S)[:10])
