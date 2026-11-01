# -*- coding: utf-8 -*-
# 내려받은 KOSIS json(download/kosis)을 python -I 로 읽어 연도별 특허 등록·존속권리 csv로 정리
import json, os, sys, csv
W = sys.argv[1]
K = os.path.join(W, 'download', 'kosis')
reg = {r['PRD_DE']: int(r['DT']) for r in json.load(open(os.path.join(K, 'data_TX_13801_A006.json'), encoding='utf-8')) if r.get('C1_NM') == '특허'}
inf = {r['PRD_DE']: int(r['DT']) for r in json.load(open(os.path.join(K, 'data_TX_13801_A052.json'), encoding='utf-8')) if r.get('C1_NM') == '특허' and r.get('C2_NM') == '계'}
with open(os.path.join(W, 'cache', 's05_kosis.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['year', 'kipo_reg_patent', 'kipo_inforce_patent'])
    for y in sorted(set(reg) | set(inf)): w.writerow([y, reg.get(y), inf.get(y)])
print(len(reg), len(inf))
