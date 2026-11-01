import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
K50 = sorted({r["세세분류코드"] for r in csv.DictReader(open(klib.BASE + r"\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv", encoding="utf-8-sig"))})
r = klib.data("DT_1K52C03", "census", "DT_1K52C03_2016_2020", itmId="T1+T2", objL1="00", objL2="+".join(["C"] + ["C" + k for k in K50]), objL3="ALL", prdSe="Y", startPrdDe="2016", endPrdDe="2020")
print(len(r) if isinstance(r, list) else r)
if isinstance(r, list):
    print(r[0]); print(sorted({x["C3"] + ":" + x["C3_NM"] for x in r}))
