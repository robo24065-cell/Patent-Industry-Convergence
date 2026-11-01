# -*- coding: utf-8 -*-
"""C15-T1 p0: 연보 원표 전체에서 '소멸/불납/포기/존속기간' 문구가 있는 시트·셀 찾기 (읽기만). 사용: python -I p0 <moip_unzip> <out>"""
import sys, glob, os, re
import pandas as pd
root, outp = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(root, "*", "*.xls*")))
pat = re.compile("소멸|불납|포기|존속기간|연차료")
with open(outp, "w", encoding="utf-8") as o:
    for f in fs:
        try:
            x = pd.read_excel(f, sheet_name=None, header=None)
        except Exception as e:
            o.write(f"ERR {f} {e}\n"); continue
        for sn, df in x.items():
            hits = []
            for v in df.values.ravel():
                if isinstance(v, str) and pat.search(v):
                    hits.append(v.strip().replace("\n", " ")[:80])
            if hits:
                o.write(f"{os.path.relpath(f, root)} | {sn} | {df.shape} | {' || '.join(sorted(set(hits))[:8])}\n")
        o.flush()
print("done")
