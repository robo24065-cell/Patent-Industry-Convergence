# -*- coding: utf-8 -*-
"""(B1) 구 기준 광공업생산지수 표 ID 탐침: 메타(TBL·PRD·CMMT)만 요청. 사용: python -B b1_probe.py DT_1F01501,DT_1F01001"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
import klib
sys.stdout.reconfigure(encoding='utf-8')
for tbl in sys.argv[1].split(','):
    for typ in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['TBL', 'PRD', 'CMMT']):
        try:
            j = klib.meta(tbl, typ, sub='meta_old')
        except Exception as e:
            print(tbl, typ, 'EXC', e); break
        s = json.dumps(j, ensure_ascii=False)
        print(tbl, typ, s[:1500])
        if isinstance(j, dict) and 'err' in j: break
