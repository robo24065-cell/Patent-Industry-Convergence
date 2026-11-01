# -*- coding: utf-8 -*-
r"""C18-T2 verify_1: '영향 없음' 결론 반박 시도.
 R1 V1A(쌍에서 7자리 유지) · R2 VNUS(US 행 전체 제거, 상한) — t3res 결과 요약
 R3 결함 노출이 큰 셀(최종 t0=2021 시험 셀 중 US 비중 상위 10%, D1 쌍 행 비중 상위, 공개일 없는 US 행(=V2 대상) 비중 상위 10%)에서
    P0 대비 변형의 점수 순위 변화(순위 백분위 중앙 |Δ|)와 전체 상위 10% 안 적중 수 변화
 → verify_1/v1_refute.json"""
import os, json, glob
import numpy as np, pandas as pd, pyarrow.dataset as ds
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T2'
C = W + r'\cache'; T = C + r'\t3res'; R = {}
for V in ('V1A', 'VNUS', 'V1', 'V2', 'V12', 'V1E', 'V2r'):
    f = T + rf'\res_{V}_final.json'
    if os.path.exists(f):
        R[f'{V}_final_skill'] = json.load(open(f, encoding='utf-8'))['mean_skill']
p0 = json.load(open(T + r'\res_P0_final.json', encoding='utf-8'))['mean_skill']
R['P0_final_skill'] = p0
R['delta'] = {k.replace('_final_skill', ''): v - p0 for k, v in R.items() if k.endswith('_final_skill')}
# R3: 셀 노출 (최종 캐시 base_rows: P0 공식 vs V2·V1 변형 행 수 차이, 창 = 공개연도 2019–2021)
def cell_counts(f, us_only=True):
    t = ds.dataset(f).to_table(columns=['ksic', 'cc', 'year', 'ipc_raw']).to_pandas()
    t = t[(t.year >= 2019) & (t.year <= 2021)]
    import re
    m = t.ipc_raw.str.replace(' ', '', regex=False).str.extract(r'^([A-H][0-9]{2}[A-Z])([0-9]{1,4})/')
    t['cell'] = t.ksic.astype(str) + '|' + m[0] + ' ' + m[1].astype(float).astype('Int64').astype(str)
    t = t[m[0].notna()]
    return t.groupby(['cell', 'cc']).size().unstack(fill_value=0)
off = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet'
A0 = cell_counts(off); A2 = cell_counts(C + r'\t3_V2_fin\base_rows.parquet'); A1 = cell_counts(C + r'\t3_V1_fin\base_rows.parquet')
P = pd.read_parquet(T + r'\preds_P0_final.parquet')
ex = pd.DataFrame(index=P.cell.values)
tot = A0.sum(1).reindex(ex.index).fillna(0)
ex['us_share'] = (A0['US'].reindex(ex.index).fillna(0) / tot.clip(lower=1)).values
ex['d2_share'] = ((A0['US'] - A2['US'].reindex(A0.index).fillna(0)).reindex(ex.index).fillna(0) / tot.clip(lower=1)).values
ex['d1_rows'] = ((A0['US'] - A1['US'].reindex(A0.index).fillna(0)).reindex(ex.index).fillna(0)).values
R['test_cells'] = int(len(ex)); R['cells_with_d1_rows_in_window'] = int((ex.d1_rows > 0).sum())
R['d2_share_dist'] = [float(x) for x in np.percentile(ex.d2_share, [50, 90, 99, 100])]
P['rk'] = P.p.rank(pct=True); P = P.set_index('cell')
k10 = P.rk >= 0.9
out = {}
for V in ('V1', 'V2', 'V12', 'V2r', 'V1A', 'VNUS'):
    f = T + rf'\preds_{V}_final.parquet'
    if not os.path.exists(f): continue
    Q = pd.read_parquet(f); Q['rk'] = Q.p.rank(pct=True); Q = Q.set_index('cell')
    common = P.index.intersection(Q.index)
    res = {'n_common': int(len(common)), 'n_only_P0': int(len(P.index.difference(Q.index))), 'n_only_V': int(len(Q.index.difference(P.index)))}
    for nm, sel in (('all', np.ones(len(ex), bool)), ('us_share_top10', ex.us_share >= np.percentile(ex.us_share, 90)), ('d2_share_top10', ex.d2_share >= np.percentile(ex.d2_share, 90))):
        cells = [c for c in ex.index[sel] if c in common]
        d = (Q.loc[cells, 'rk'] - P.loc[cells, 'rk']).abs()
        hitP = int(((P.loc[cells, 'rk'] >= 0.9) & (P.loc[cells, 'yb'] == 1)).sum()); hitV = int(((Q.loc[cells, 'rk'] >= 0.9) & (Q.loc[cells, 'yb'] == 1)).sum())
        res[nm] = dict(n=len(cells), median_abs_rank_change=float(d.median()), p90_abs_rank_change=float(d.quantile(0.9)), top10_hits_P0=hitP, top10_hits_V=hitV,
                       pos=int(P.loc[cells, 'yb'].sum()))
    out[V] = res
R['R3_cells'] = out
json.dump(R, open(W + r'\verify_1\v1_refute.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False)[:4000])
