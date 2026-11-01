# -*- coding: utf-8 -*-
import json, openpyxl, hashlib, sys
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
X = BASE + r"\공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx"
J = BASE + r"\공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"
for p in (X, J):
    print(p[-30:], hashlib.sha256(open(p,'rb').read()).hexdigest())
wb = openpyxl.load_workbook(X, read_only=True)
for ws in wb.worksheets:
    print("SHEET", ws.title, ws.max_row, ws.max_column)
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 6: print("  ", r)
        else: break
d = json.load(open(J, encoding="utf-8"))
print(type(d), len(d)); print(json.dumps(d[0], ensure_ascii=False)[:1500]); print(sorted(set(k for x in d for k in x)))
