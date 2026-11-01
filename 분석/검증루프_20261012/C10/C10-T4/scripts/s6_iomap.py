import sys, openpyxl
sys.stdout.reconfigure(encoding="utf-8")
p = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\산업연관표\IO코드와 KSIC 및 HS코드 대조표.xlsx"
wb = openpyxl.load_workbook(p, read_only=True)
for ws in wb.worksheets:
    print("==", ws.title, ws.max_row, ws.max_column)
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 6: print([str(x)[:40] for x in r if x is not None])
        else: break
