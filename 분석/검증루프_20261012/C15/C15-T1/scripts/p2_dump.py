# -*- coding: utf-8 -*-
"""C15-T1 p2: 지정 원표 시트 덤프(읽기만). 사용: python -I p2_dump.py <xlsx> <시트명 부분문자열> <out.txt>"""
import sys, openpyxl
f, key, outp = sys.argv[1], sys.argv[2], sys.argv[3]
wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
with open(outp, "a", encoding="utf-8") as o:
    for sn in wb.sheetnames:
        if key in sn:
            o.write(f"##### {f.split(chr(92))[-1]} | {sn}\n")
            for row in wb[sn].iter_rows(values_only=True):
                if any(v is not None for v in row):
                    o.write(" | ".join("" if v is None else str(v).replace("\n", " ") for v in row) + "\n")
print("ok")
