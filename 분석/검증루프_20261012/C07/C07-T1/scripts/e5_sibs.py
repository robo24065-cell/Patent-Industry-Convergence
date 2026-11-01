import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
sib = ["26219", "26221", "26222", "25913", "28121", "29171", "27193", "31311", "31322", "31920"]
j = klib.data("DT_1FS1001", "kosis10", "DT_1FS1001_siblings_2007_2020", itmId="T01+T02+T04", objL1="00", objL2="+".join("C" + s for s in sib), prdSe="Y", startPrdDe="2007", endPrdDe="2020")
print(len(j) if isinstance(j, list) else j)
