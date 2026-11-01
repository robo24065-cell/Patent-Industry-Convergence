# -*- coding: utf-8 -*-
"""(B1) 구표 DT_1F01001·신표 DT_1F02001 ITM 메타 요약."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
from kjson import loadj
sys.stdout.reconfigure(encoding='utf-8')
D = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T3\download\meta_old"
for t in ['DT_1F01001', 'DT_1F02001']:
    j = loadj(os.path.join(D, f'meta_{t}_ITM.json'))
    print('==', t, len(j), list(j[0].keys()))
    print(collections.Counter((e.get('OBJ_ID'), e.get('OBJ_NM')) for e in j))
    by = collections.defaultdict(list)
    for e in j: by[e.get('OBJ_ID')].append((e.get('ITM_ID'), e.get('ITM_NM'), e.get('UP_ITM_ID')))
    for k, v in by.items():
        print(k, len(v), v[:12] if len(v) > 30 else v)
