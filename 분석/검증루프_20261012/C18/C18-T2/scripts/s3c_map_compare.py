# C18-T2 s3c: 원고 3.1 생성물 P0 재현(원본 저장값 대조) + V1·V1E 대 P0 차이 → cache/s3c_map_compare.json, cache/s3c_map_diffs.csv
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
FA = MC + r'\_shared_report\fusion_advantage'
ORIG = {'map5': FA + r'\map5', 'split34': FA + r'\split34'}
FILES = [('map5', 'm2_checks.json'), ('map5', 'm3c_contemporaneous.json'), ('map5', 'm4_summary.json'), ('split34', 't1_meta.json'),
         ('map5', 'm2_ksic_profile.csv'), ('map5', 'm2_country_summary.csv'), ('map5', 'm2_coverage_factors_pub_total.csv'), ('map5', 'm4_quadrant_table.csv'),
         ('split34', 't1_variance_decomposition.csv'), ('split34', 't2_within_group_external.csv'), ('map5', 'm2_panel_appl.csv'), ('map5', 'm5_summary.json'), ('map5', 'm5_indicator_spearman.csv'), ('map5', 'm1_counts_appl.parquet')]


def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f'{p}[{i}]')
    else:
        yield p, o


def load(path):
    if path.endswith('.json'):
        return dict(flat(json.load(open(path, encoding='utf-8'))))
    d = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    d = d.reset_index(drop=True)
    return {f'r{i}.{c}': v for c in d.columns for i, v in enumerate(d[c].tolist())}


def num(x):
    try:
        f = float(x); return None if np.isnan(f) else f
    except Exception:
        return None


def cmp(a, b):
    keys = sorted(set(a) | set(b)); diffs = []; mx = 0.0; nd = 0
    for k in keys:
        x, y = a.get(k), b.get(k); fx, fy = num(x), num(y)
        if fx is not None and fy is not None:
            d = abs(fx - fy)
            if d > 1e-9: nd += 1; diffs.append((k, x, y, d)); mx = max(mx, d)
        elif str(x) != str(y) and not (num(x) is None and num(y) is None and str(x) in ('nan', 'None') and str(y) in ('nan', 'None')):
            nd += 1; diffs.append((k, x, y, None))
    return dict(n_keys=len(keys), n_diff=nd, max_abs=mx), diffs


res, rows = {}, []
for grp, fn in FILES:
    o = os.path.join(ORIG[grp], fn); p0 = os.path.join(C, 'map_P0', fn)
    if not (os.path.exists(o) and os.path.exists(p0)):
        res[f'{fn}|P0_vs_orig'] = 'missing'; continue
    A0 = load(o); P0 = load(p0)
    s, d = cmp(A0, P0); res[f'{fn}|P0_vs_orig'] = s
    for k, x, y, dd in d[:30]: rows.append(('P0_vs_orig', fn, k, x, y, dd))
    for V in ('V1', 'V1E'):
        pv = os.path.join(C, f'map_{V}', fn)
        if not os.path.exists(pv): res[f'{fn}|{V}_vs_P0'] = 'missing'; continue
        s, d = cmp(P0, load(pv)); res[f'{fn}|{V}_vs_P0'] = s
        for k, x, y, dd in d[:400]: rows.append((f'{V}_vs_P0', fn, k, x, y, dd))
pd.DataFrame(rows, columns=['cmp', 'file', 'key', 'a', 'b', 'absdiff']).to_csv(os.path.join(C, 's3c_map_diffs.csv'), index=False, encoding='utf-8-sig')
jdump(res, 's3c_map_compare.json')
for k, v in res.items(): out(k, v)
