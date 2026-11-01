# -*- coding: utf-8 -*-
import csv, sys, os
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012'
r = list(csv.DictReader(open(os.path.join(B, 'C18', 'C18-T3', 'errata_status_v18.csv'), encoding='utf-8-sig')))
delta = [x['ID'] for x in csv.DictReader(open(os.path.join(B, 'C19', 'C19-T4', 'status_v20_delta.csv'), encoding='utf-8-sig'))]
for x in r:
    if x['ID'] in delta: continue
    hit = [d for d in delta + ['E-C12T2-08'] if d in x['대체ID'] or d in x['근거']]
    if hit or x['ID'] == 'E-C12T2-08':
        print(x['ID'], x['상태'], x['분류'], x['범위'], x['v19줄'], x['대체ID'], '|', x['근거'][:250], '| hit', hit)
