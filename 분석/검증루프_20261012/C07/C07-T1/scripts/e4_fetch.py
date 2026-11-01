import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
B = klib.BASE
K50 = sorted({r["세세분류코드"] for r in csv.DictReader(open(B + r"\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv", encoding="utf-8-sig"))})
# 9차 대응 구코드(연계표에서): 50개 신코드의 9차 코드
link = [r for r in csv.reader(open(klib.HERE + r"\download\ksic_link\ksic_ksco_link_2025Q2.csv", encoding="cp949"))]
old9 = sorted({r[5] for r in link if r[0] == "KSIC" and r[1] == "10" and r[2] == "5" and r[3] in K50})
print(len(K50), "old9", len(old9), old9)
objs10 = "+".join(["0", "C"] + ["C" + k for k in K50])
objs9 = "+".join(["0", "C"] + ["C" + k for k in sorted(set(old9) | set(K50))])
objs11 = objs10
jobs = [("DT_1FS1001", "kosis10", "DT_1FS1001_2007_2020", objs10, "2007", "2020"),
        ("DT_1F160622", "kosis9", "DT_1F160622_2005_2014", objs9, "2005", "2014"),
        ("DT_1FS1101", "kosis11", "DT_1FS1101_2020_2024", objs11, "2020", "2024")]
for tbl, sub, nm, objs, s, e in jobs:
    j = klib.data(tbl, sub, nm, itmId="T01+T02+T03+T04+T05+T06+T07", objL1="00", objL2=objs, prdSe="Y", startPrdDe=s, endPrdDe=e)
    if isinstance(j, dict):
        print(tbl, "ERR", j); continue
    print(tbl, len(j), "rows; codes", len({r["C2"] for r in j}), "years", sorted({r["PRD_DE"] for r in j}))
