# -*- coding: utf-8 -*-
"""탐색 보조: JSON 평탄화 후 키 정규식 필터 출력. 사용: python jflat.py <json> [regex] [maxlen]"""
import json, sys, re
p = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
ml = int(sys.argv[3]) if len(sys.argv) > 3 else 120
d = json.load(open(p, encoding='utf-8'))
def walk(o, k=''):
    if isinstance(o, dict):
        for kk, v in o.items():
            yield from walk(v, f'{k}/{kk}')
    elif isinstance(o, list) and o and all(not isinstance(x, (dict, list)) for x in o) and len(o) <= 6:
        yield k, o
    elif isinstance(o, list):
        for i, v in enumerate(o[:200]):
            yield from walk(v, f'{k}[{i}]')
    else:
        yield k, o
for k, v in walk(d):
    if pat is None or pat.search(k):
        print(k, '=', str(v)[:ml])
