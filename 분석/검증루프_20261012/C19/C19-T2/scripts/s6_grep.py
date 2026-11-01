# -*- coding: utf-8 -*-
r"""C19-T2 s6: 9개 공식 트랙 동결 스펙에서 family·해외 관련 피처 이름을 뽑고, 트랙 코드에서 family 토큰을 나누거나 세는 줄을 grep(읽기만).
출력: cache/s6_spec_feats.json, cache/s6_grep_hits.csv"""
import os, sys, re, json, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd

SPECS = {'T1': J('T1_산업대기실', 'R3', 'final_spec_r3.json'), 'T3': J('T3_기술셀부상예측', 'R3', 'final_spec_r3.json'), 'T4': J('T4_특허포기예측', 'R2', 'final_spec_r2.json'),
         'N4': J('N4_특허행태_기업폐업조기경보', 'final_spec_r4.json'), 'N5': J('N5_표준특허선언레이더', 'final_spec_r4.json'), 'N6': J('N6_정부RnD기업_첫특허전환', 'final_spec_r5.json'),
         'N7': J('N7_특허기업_정부RnD첫수행', 'final_spec_r5.json'), 'N8': J('N8_특허분쟁표적레이더', 'final_spec_r6.json'), 'N9': J('N9_정부RnD과제_특허성과예측', 'final_spec_r6.json')}
PAT = re.compile(r'(fam|family|foreign|prio|ros_|pct|_wo|wo_|ep_|us_|jp_|cn_|glob|tm_|ext|offices)', re.I)


def strings(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v, path + '/' + str(k))
    elif isinstance(o, list):
        for v in o:
            yield from strings(v, path)
    elif isinstance(o, str):
        yield path, o


R = {}
for t, p in SPECS.items():
    s = json.load(open(J(MC, p), encoding='utf-8'))
    ids = sorted({(pa, v) for pa, v in strings(s) if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{1,40}', v) and PAT.search(v)})
    R[t] = {'spec': p, 'family_like_ids': [f'{v}  @{pa}' for pa, v in ids]}
jdump(R, J(C, 's6_spec_feats.json'))
# 코드 grep
DIRS = {'T1': 'T1_산업대기실', 'T3': 'T3_기술셀부상예측', 'T4': 'T4_특허포기예측', 'N4': 'N4_특허행태_기업폐업조기경보', 'N5': 'N5_표준특허선언레이더',
        'N6': 'N6_정부RnD기업_첫특허전환', 'N7': 'N7_특허기업_정부RnD첫수행', 'N8': 'N8_특허분쟁표적레이더', 'N9': 'N9_정부RnD과제_특허성과예측', 'shared_ext': J('_shared', 'ext')}
G = re.compile(r"(family|fam_|famstr|\.split\(['\"]\|['\"]\)|string_split\(fam)", re.I)
rows = []
for t, d in DIRS.items():
    for f in glob.glob(J(MC, d, '**', '*.py'), recursive=True):
        if '__pycache__' in f:
            continue
        try:
            L = open(f, encoding='utf-8', errors='replace').read().splitlines()
        except Exception:
            continue
        for i, line in enumerate(L, 1):
            if G.search(line):
                cnt = bool(re.search(r'(size\(\)|nunique|len\(|count\(|sum\(|bincount)', line))
                rows.append(dict(track=t, file=os.path.relpath(f, MC), line=i, counts=cnt, text=line.strip()[:220]))
pd.DataFrame(rows).to_csv(J(C, 's6_grep_hits.csv'), index=False, encoding='utf-8-sig')
out('hits', len(rows), pd.DataFrame(rows).groupby('track').size().to_dict())
for t in R:
    print(t, R[t]['family_like_ids'][:40])
