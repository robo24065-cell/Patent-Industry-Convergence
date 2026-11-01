# -*- coding: utf-8 -*-
"""(B1) 내려받기: 구표 DT_1F01001(2010=100, 9차, 1975.01–2017.12)과 신표 DT_1F02001(2020=100) 전국·전 산업·T10/T11 월별.
KOSIS OpenAPI(PROTOCOL_R2 5.1b 승인 키, 값 기록 금지), 요청 간 2.2초, 4만 셀 한도 때문에 기간 분할."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
import klib
sys.stdout.reconfigure(encoding='utf-8')
jobs = [('DT_1F01001', 'old9', '200501', '201012'), ('DT_1F01001', 'old9', '201101', '201712'),
        ('DT_1F02001', 'new10', '200501', '201212'), ('DT_1F02001', 'new10', '201301', '202112')]
for tbl, sub, a, b in jobs:
    j = klib.data(tbl, sub, f'{tbl}_00_ALL_T10T11_{a}_{b}', objL1='00', objL2='ALL', itmId='T10+T11', prdSe='M', startPrdDe=a, endPrdDe=b)
    n = len(j) if isinstance(j, list) else j
    print(tbl, a, b, n if not isinstance(j, list) else len(j), (j[0] if isinstance(j, list) and j else ''))
