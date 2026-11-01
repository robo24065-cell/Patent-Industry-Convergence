# -*- coding: utf-8 -*-
# 각 연보의 심판 통계 파일에서 시트 이름을 나열하고, '심판종류별' 시트를 텍스트로 덤프한다(탐색용).
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
root = sys.argv[1]; out = sys.argv[2]
os.makedirs(out, exist_ok=True)
for d in sorted(glob.glob(os.path.join(root, "*"))):
    fs = [f for f in glob.glob(os.path.join(d, "*")) if "심판" in os.path.basename(f)]
    for f in fs:
        try:
            xl = pd.ExcelFile(f)
        except Exception as e:
            print("ERR", f, e); continue
        print("==", os.path.basename(d), os.path.basename(f), xl.sheet_names)
        for sh in xl.sheet_names:
            df = xl.parse(sh, header=None)
            txt = df.to_csv(index=False, header=False)
            tag = os.path.basename(d) + "__" + "".join(ch if ch.isalnum() else "_" for ch in sh)
            with open(os.path.join(out, tag + ".csv"), "w", encoding="utf-8") as g:
                g.write(f"# {f}\n# sheet {sh}\n" + txt)
