# C19-T3 s6: (1) 사본 P0 산출 ↔ 원본 저장값 대조(재현 관문) (2) P0 ↔ V1E 차이 전수 목록
# 사용: python s6_compare.py map|lo
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd, numpy as np
grp = sys.argv[1]
FA = MC + r'\_shared_report\fusion_advantage'
LO = MC + r'\_shared_report\label_origin_R10'
if grp == 'map':
    P0 = C + r'\map_P0'; V = C + r'\map_V1E'
    def orig(fn):
        for d in (FA + r'\map5', FA + r'\split34'):
            p = os.path.join(d, fn)
            if os.path.exists(p): return p
else:
    P0 = C + r'\lo_P0'; V = C + r'\lo_V1E'
    def orig(fn):
        rel = os.path.relpath(fn, P0) if os.path.isabs(fn) else fn
        if rel.startswith('fp' + os.sep):
            p = os.path.join(LO, 'foreign_public_R11', rel[3:])
        else:
            p = os.path.join(LO, rel)
        return p if os.path.exists(p) else None
SKIPKEY = {'elapsed', 'sec', 'seconds', 'time', 'runtime', 'elapsed_s', 'run_sec'}
def flat(o, pre=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(flat(v, f'{pre}/{k}'))
    elif isinstance(o, list):
        for i, v in enumerate(o): out.update(flat(v, f'{pre}[{i}]'))
    else:
        out[pre] = o
    return out
def load(p):
    if p.endswith('.json'):
        return ('json', flat(json.load(open(p, encoding='utf-8'))))
    if p.endswith('.csv'):
        try:
            d = pd.read_csv(p, dtype=str, encoding='utf-8-sig', keep_default_na=False)
        except Exception:
            d = pd.read_csv(p, dtype=str, encoding='cp949', keep_default_na=False)
        return ('csv', d)
    if p.endswith('.parquet'):
        return ('pq', pd.read_parquet(p))
    return (None, None)
def num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None
def diff(a, b, cap=100000):
    ka, da = a; kb, db = b
    rows = []
    if ka == 'json':
        for k in sorted(set(da) | set(db)):
            if any(s in k.lower().split('/')[-1] for s in SKIPKEY): continue
            x, y = da.get(k), db.get(k)
            if x != y: rows.append((k, x, y))
    elif ka == 'csv':
        if da.shape != db.shape or list(da.columns) != list(db.columns):
            rows.append(('<shape>', str(da.shape) + str(list(da.columns))[:200], str(db.shape) + str(list(db.columns))[:200]))
            n = min(len(da), len(db)); cols = [c for c in da.columns if c in db.columns]
            da = da.iloc[:n][cols]; db = db.iloc[:n][cols]
        ne = (da.values != db.values)
        for i, j in zip(*np.where(ne)):
            rows.append((f'r{i}:{da.columns[j]}|{da.iloc[i, 0]}', da.iat[i, j], db.iat[i, j]))
            if len(rows) > cap: break
    elif ka == 'pq':
        try:
            eq = da.shape == db.shape and da.reset_index(drop=True).equals(db.reset_index(drop=True))
        except Exception:
            eq = False
        if not eq:
            rows.append(('<parquet>', str(da.shape), str(db.shape)))
    return rows
files = sorted(p for p in glob.glob(P0 + r'\**\*', recursive=True) if os.path.isfile(p) and p.split('.')[-1] in ('json', 'csv', 'parquet') and r'\cache\_tmp' not in p and 'duck_tmp' not in p)
R = {'repro': [], 'v1e': []}
for p in files:
    rel = os.path.relpath(p, P0)
    o = orig(rel if grp == 'map' else p)
    if grp == 'map': o = orig(os.path.basename(p))
    a = load(p)
    if a[0] is None: continue
    if o:
        try:
            d0 = diff(load(o), a)
        except Exception as e:
            d0 = [('<err>', str(e), '')]
        R['repro'].append(dict(file=rel, orig=o, n_diff=len(d0), sample=[list(map(str, r)) for r in d0[:8]]))
    pv = os.path.join(V, rel)
    if os.path.exists(pv):
        d1 = diff(a, load(pv))
        R['v1e'].append(dict(file=rel, n_diff=len(d1), diffs=[list(map(str, r)) for r in d1[:3000]]))
    else:
        R['v1e'].append(dict(file=rel, n_diff=-1, diffs=[]))
jdump(R, f's6_compare_{grp}.json')
for r in R['repro']: out('REPRO', r['file'], r['n_diff'], r['sample'][:2])
for r in R['v1e']: out('V1E', r['file'], r['n_diff'])
