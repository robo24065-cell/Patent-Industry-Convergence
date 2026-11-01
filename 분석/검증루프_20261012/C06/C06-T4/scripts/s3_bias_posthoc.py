# s3: 편향 점검(C06-A 연결된 벤처 vs 안 된 벤처, 특허기업 중 벤처 연결 vs 비연결) + 사후 진단(판정 밖).
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib6 import *
T0 = time.time()
ca = load_ca(); nm2b6, _ = name_map(ca)
docs = load_docs().drop_duplicates('appl_no')
docs = docs[docs.vis.notna() & docs.nname.isin(nm2b6.keys())]
v = pd.read_parquet(os.path.join(C, 'venture_clean.parquet'))
v['s8'] = v.s.dt.strftime('%Y%m%d')
A = pd.read_parquet(os.path.join(C, 'A_units.parquet'))
unit_names = set(A.nname)
v['linked'] = v.nname.isin(unit_names)
ROWS = []


def tvd(a, b):
    pa = a.value_counts(normalize=True); pb = b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return round(float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum()), 4)


def smd(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    s = np.sqrt((x1.var() + x0.var()) / 2)
    return round(float((x1.mean() - x0.mean()) / s), 4) if s > 0 else 0.0


L, U = v[v.linked], v[~v.linked]
ROWS += [('C06-A', '벤처(연결) vs 벤처(비연결)', '건수', len(L), len(U), ''),
         ('C06-A', '벤처(연결) vs 벤처(비연결)', '지역 TVD', tvd(L.region, U.region), '', ''),
         ('C06-A', '벤처(연결) vs 벤처(비연결)', '확인유형 TVD', tvd(L.vtype, U.vtype), '', ''),
         ('C06-A', '벤처(연결) vs 벤처(비연결)', '업종명(11차) TVD', tvd(L.ind11, U.ind11), '', ''),
         ('C06-A', '벤처(연결) vs 벤처(비연결)', '신규 비율', round(float((L.newre == '신규').mean()), 4), round(float((U.newre == '신규').mean()), 4), ''),
         ('C06-A', '벤처(연결) vs 벤처(비연결)', '시작연도 TVD', tvd(L.s.dt.year, U.s.dt.year), '', '')]
# 특허기업(개발 첫 원점 위험집합 이전 전체 상호): 벤처 명단 등장 vs 미등장의 as-of 속성(기준일 2025-10-31까지 보인 문서)
ref = ymd(LOCK_A[0])
dv = docs[docs.vis <= ref]
g = dv.groupby('nname').agg(n=('appl_no', 'size'), fy=('ay', 'min'), reg=('rg', lambda s: float(s.notna().mean())), dom=('dom_prio', 'mean'))
g['inlist'] = g.index.isin(set(v.nname))
a1, a0 = g[g.inlist], g[~g.inlist]
for c, lab in [('n', 'log1p 문서 수'), ('fy', '첫 출원연도'), ('reg', '등록 비율'), ('dom', '국내 우선 비율')]:
    x1 = np.log1p(a1[c]) if c == 'n' else a1[c]; x0 = np.log1p(a0[c]) if c == 'n' else a0[c]
    ROWS.append(('C06-A', '특허기업: 벤처명단 등장 vs 미등장(기준 2025-10-31)', f'SMD {lab}', smd(x1, x0), f'n1={len(a1)}', f'n0={len(a0)}'))
km = dv.groupby(['nname', 'ksic']).size().reset_index(name='c').sort_values(['nname', 'c'], ascending=[True, False]).drop_duplicates('nname').set_index('nname').ksic
ROWS.append(('C06-A', '특허기업: 벤처명단 등장 vs 미등장', 'KSIC 최빈 TVD', tvd(km[km.index.isin(a1.index)], km[km.index.isin(a0.index)]), '', ''))
POST = {}
# 사후 1: C06-A 결과를 '신규+재확인'으로 넓히면(판정 밖)
for t0, t1 in DEV_A:
    a, b = ymd(t0), ymd(t1)
    u = A[A.t0 == t0]
    ev_all = set(v[(v.s8 > a) & (v.s8 <= b)].nname)
    POST[f'A_any_start_{t0}'] = dict(n=len(u), pos_new=int(u.y.sum()), pos_any=int(u.nname.isin(ev_all).sum()))
# 사후 2: C06-B를 '상호만 같고 biz6가 다른 새 사업장'까지 넓히면(판정 밖, 동명이인 위험) — 개발 2개 원점만
B_ = pd.read_parquet(os.path.join(C, 'B_units.parquet'))
for s0, s1, r0, r1 in [('202112', '202212', '20211231', '20221231'), ('202212', '202312', '20221231', '20231231')]:
    d1 = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s1}.parquet', columns=['name', 'biz6', 'stat', 'adpt_dt'])
    d1 = d1[(d1.stat.astype(str) == '1') & d1.biz6.astype(str).str[3:5].isin(CORP_TC)]
    d1['nname'] = d1.name.map(norm); d1['ad'] = d1.adpt_dt.fillna('').astype(str).str.replace('-', '').str[:8]
    u = B_[B_.s0 == s0].copy(); u['nname'] = u.key.str.split('|', n=1).str[1]
    newn = set(d1[(d1.ad > r0) & (d1.ad <= r1)].nname)
    POST[f'B_name_only_{s0}'] = dict(n=len(u), pos=int(u.nname.isin(newn).sum()))
pd.DataFrame(ROWS, columns=['후보', '비교', '지표', '값1', '값2', '비고']).to_csv(os.path.join(C, 's3_bias_rows.csv'), index=False, encoding='utf-8-sig')
json.dump(POST, open(os.path.join(C, 's3_posthoc.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for r in ROWS: print(r)
print(json.dumps(POST, ensure_ascii=False, indent=1)); print('done', round(time.time() - T0))
