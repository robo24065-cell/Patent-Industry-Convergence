# -*- coding: utf-8 -*-
"""C18-T1 s0: C13-T1이 내려받은 지식재산통계연보(읽기 전용)에서 '월별 등록' 시트를 찾아 텍스트로 덤프."""
import sys, pathlib, re, openpyxl, hashlib
sys.dont_write_bytecode = True
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
U = B/'분석/검증루프_20261012/C13/C13-T1/download/moip_unzip'
OUT = B/'분석/검증루프_20261012/C18/C18-T1/cache'
o = open(OUT/'s0_monthly_sheets_dump.txt', 'w', encoding='utf-8')
import time; t0=time.time()
for d in sorted(U.iterdir()):
    for f in sorted(d.rglob('*.xls*')):
        if f.suffix.lower() != '.xlsx' or not re.search(r'기타|참고', f.name):
            continue
        try:
            wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        except Exception as e:
            o.write(f'!! {f} {e}\n'); continue
        for ws in wb.worksheets:
            if re.search(r'월\s*별\s*등\s*록', ws.title):
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                o.write(f'\n######## {d.name} | {f.name} | sheet={ws.title} | sha256={h}\n')
                for r in ws.iter_rows(values_only=True):
                    if any(v is not None for v in r):
                        o.write(' | '.join('' if v is None else str(v) for v in r) + '\n')
        wb.close(); o.flush(); print(d.name, f.name[-25:], round(time.time()-t0), flush=True)
o.close()
