# -*- coding: utf-8 -*-
r"""C19-T2 verify v1: 정규화 독립 구현(정규식·str.lstrip·pandas groupby 미사용).
 토큰 문자열을 문자 단위로 파싱: 국가 2자(알파벳) → 연속 숫자 구간 → 나머지(종류 코드 접미) 버림, 일련번호는 int()로 앞 0 제거.
 F1 키 = (국가, int(숫자)). F1x 키: US 10·11자리이고 앞 4자리가 1985–2030 → (US,'P',연도,int(뒤)), JP 10자리 동일, JP 8자리(0 아님) → (JP,'P',1988+앞2,int(뒤6)), 그 밖 F1 키.
 pid별 키→최소 yr 사전을 만들고, 각 프레임 행(pid,L)마다 정렬된 연도 목록에서 bisect로 yr<=L연도-1 개수를 센다.
 입력: T4 cache/patent.parquet·s2_fam_tokens.parquet(원본 다시 읽음), 공식 F22 프레임. 비교 대상: cache/famn/F1__*.npy·F1x__*.npy
 출력: verify/v1_parse.json"""
import os, sys, bisect
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from lib19 import *
import numpy as np, pandas as pd
import pyarrow.parquet as pq

pt = pq.read_table(J(T4, 'cache', 'patent.parquet'), columns=['rgst_no', 'rgstDate', 'applNo']).to_pydict()
amap = {}
i = 0
for rn, rd, a in zip(pt['rgst_no'], pt['rgstDate'], pt['applNo']):
    if rd is None or (isinstance(rd, float) and rd != rd) or str(rd).strip() == '' or rn is None or str(rn)[:2] != '10':
        continue
    if a not in amap:
        amap[a] = i
    i += 1
tk = pq.read_table(J(T4, 'cache', 's2_fam_tokens.parquet'), columns=['applNo', 'yr', 'tok']).to_pydict()


def parse(tok):
    cc = tok[:2]
    j = 2
    while j < len(tok) and tok[j].isdigit():
        j += 1
    dg = tok[2:j]                       # 뒤 접미는 버림
    return cc, dg


def key1(cc, dg):
    return (cc, int(dg)) if dg else (cc, -1)


def keyx(cc, dg):
    n = len(dg)
    if cc in ('US', 'JP') and n in (10, 11) and 1985 <= int(dg[:4]) <= 2030:
        return (cc, 'P', int(dg[:4]), int(dg[4:]))
    if cc == 'JP' and n == 8 and dg[0] != '0':
        return (cc, 'P', 1988 + int(dg[:2]), int(dg[2:]))
    return key1(cc, dg)


D1, DX = {}, {}
nm = 0
for a, y, t in zip(tk['applNo'], tk['yr'], tk['tok']):
    p = amap.get(a)
    if p is None:
        continue
    nm += 1
    cc, dg = parse(t)
    for D, k in ((D1, key1(cc, dg)), (DX, keyx(cc, dg))):
        d = D.setdefault(p, {})
        if k not in d or y < d[k]:
            d[k] = y
R = {'tokens_mapped': nm, 'pids_with_tokens': len(D1)}
S1 = {p: sorted(d.values()) for p, d in D1.items()}
SX = {p: sorted(d.values()) for p, d in DX.items()}
R['unique_keys_F1'] = int(sum(len(v) for v in S1.values())); R['unique_keys_F1x'] = int(sum(len(v) for v in SX.values()))
out('built', R)
for fr in FRAMES:
    d = pq.read_table(J(F22, fr + '.parquet'), columns=['pid', 'L', 'fam_n']).to_pydict()
    v1 = np.empty(len(d['pid'])); vx = np.empty(len(d['pid']))
    for i_, (p, L) in enumerate(zip(d['pid'], d['L'])):
        c = int(str(L)[:4]) - 1
        s = S1.get(int(p)); v1[i_] = bisect.bisect_right(s, c) if s else 0
        s = SX.get(int(p)); vx[i_] = bisect.bisect_right(s, c) if s else 0
    a1 = np.load(J(C, 'famn', f'F1__{fr}.npy')); ax = np.load(J(C, 'famn', f'F1x__{fr}.npy'))
    st = np.asarray(d['fam_n'], dtype=float)
    R[fr] = dict(rows=len(v1), max_abs_vs_s1_F1=float(np.abs(v1 - a1).max()), max_abs_vs_s1_F1x=float(np.abs(vx - ax).max()),
                 n_diff_F1=int((v1 != a1).sum()), n_diff_F1x=int((vx != ax).sum()),
                 changed_vs_stored_F1=int((v1 != st).sum()), changed_vs_stored_F1x=int((vx != st).sum()))
    out(fr, R[fr])
jdump(R, J(W, 'verify', 'v1_parse.json'))
