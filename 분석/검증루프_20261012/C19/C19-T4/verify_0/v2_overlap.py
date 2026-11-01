# -*- coding: utf-8 -*-
# 바뀐 줄과 겹치는 미반영·부분 행 전부 나열(22행 밖 누락 탐색)
import csv, sys, os, re
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012'
r = list(csv.DictReader(open(os.path.join(B, 'C18', 'C18-T3', 'errata_status_v18.csv'), encoding='utf-8-sig')))
changed = {96,130,142,143,286,409,410,480,482,496,498,499,501,503,509,513,515,518,519,534,642,652,655,658}
delta = set(x['ID'] for x in csv.DictReader(open(os.path.join(B, 'C19', 'C19-T4', 'status_v20_delta.csv'), encoding='utf-8-sig')))
for x in r:
    if x['상태'] in ('미반영', '부분'):
        ls = set(int(t) for t in re.findall(r'\d+', x['v19줄']))
        if ls & changed and x['ID'] not in delta:
            print('OVERLAP-not-in-delta', x['ID'], x['상태'], x['분류'], x['v19줄'], '|', x['근거'][:300])
# 미반영·부분 중 근거에 주요 키워드
for x in r:
    if x['상태'] in ('미반영', '부분') and x['ID'] not in delta:
        if re.search(r'2025|47%|CIRC-BOUND|직권|광공업|SE=|4\.4', x['근거']):
            print('KW', x['ID'], x['상태'], x['분류'], x['범위'], x['v19줄'], '|', x['근거'][:300])
