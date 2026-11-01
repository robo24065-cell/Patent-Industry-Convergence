# C18-T2 verify_0 r2: map 산출물 원본/P0/V1E 직접 대조 (m1 정렬 후 동일성, t1 분산분해, m4 CI)
import os, json, pandas as pd, numpy as np
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T2\cache'
FA = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage'
o = pd.read_parquet(FA + r'\map5\m1_counts_appl.parquet'); p = pd.read_parquet(W + r'\map_P0\m1_counts_appl.parquet')
print('m1 cols', list(o.columns), len(o), len(p))
keys = [c for c in o.columns if o[c].dtype == object or 'year' in c or c in ('cc', 'ksic', 'ipc4', 'fld')]
os_ = o.sort_values(list(o.columns)).reset_index(drop=True); ps = p[o.columns].sort_values(list(o.columns)).reset_index(drop=True)
print('m1 sorted equal:', os_.astype(str).equals(ps.astype(str)))
for V in ('orig', 'P0', 'V1', 'V1E'):
    f = FA + r'\split34\t1_variance_decomposition.csv' if V == 'orig' else W + rf'\map_{V}\t1_variance_decomposition.csv'
    t = pd.read_csv(f, encoding='utf-8-sig')
    print('== t1', V); print(t.to_string(max_cols=20, max_colwidth=30))
for V in ('orig', 'P0', 'V1E'):
    f = FA + r'\map5\m4_summary.json' if V == 'orig' else W + rf'\map_{V}\m4_summary.json'
    j = json.load(open(f, encoding='utf-8'))
    s = json.dumps(j, ensure_ascii=False)
    i = s.find('0.048'); print('== m4', V, s[max(0, i - 300):i + 200])
