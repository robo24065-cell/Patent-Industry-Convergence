# -*- coding: utf-8 -*-
import sys, openpyxl, pathlib
sys.dont_write_bytecode=True
B=pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C13/C13-T1/download/moip_unzip")
for f in [B/'f_400_2/013_7.기타 출원등록.xlsx', B/'f_416_2/013_7.기타 출원등록.xlsx']:
    wb=openpyxl.load_workbook(f, read_only=True, data_only=True)
    print('=====',f.parent.name, wb.sheetnames)
    for sn in wb.sheetnames:
        ws=wb[sn]
        rows=list(ws.iter_rows(values_only=True))
        txt=' '.join(str(c) for r in rows[:6] for c in r if c is not None)
        if '월' in txt and '등록' in txt:
            print('---',sn)
            for r in rows[:40]:
                if any(c is not None for c in r): print([c for c in r if c is not None][:16])
