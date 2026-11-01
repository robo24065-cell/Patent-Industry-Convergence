# -*- coding: utf-8 -*-
"""(B1) KOSIS 목록 탐색: 광업제조업동향조사(MT_CTITLE_O) 아래 표 목록을 깊이 4까지 내려가며 광공업생산지수 표(구 기준 포함)를 찾는다."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
import klib
sys.stdout.reconfigure(encoding='utf-8')
seen = []; tables = []
def walk(pid, depth, path):
    if depth > 4: return
    j = klib.lst(pid, sub='list')
    if not isinstance(j, list): print('ERR', pid, j); return
    for e in j:
        if 'TBL_ID' in e:
            tables.append(dict(path=path, tbl=e['TBL_ID'], nm=e.get('TBL_NM'), send=e.get('SEND_DE'), stat=e.get('STAT_ID')))
        elif 'LIST_ID' in e:
            walk(e['LIST_ID'], depth + 1, path + ' > ' + e['LIST_NM'])
walk('MT_CTITLE_O', 1, '광업제조업동향조사')
json.dump(tables, open(os.path.join(klib.DL, 'list', 'tables_MT_CTITLE_O.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for t in tables: print(t['tbl'], t['send'], t['path'], '|', t['nm'])
