# -*- coding: utf-8 -*-
"""
적대적 검증 3차 — "특정 소수 산업이 아니라 전반적 약화" 주장을 유효단위(결과셀)에서 재검정.
저자의 LOO 는 산업 단위였으나, 같은 결과셀 산업들은 dLOCO 가 동일해 서로를 가린다.
셀 단위로 제거해야 영향점 집중도가 드러난다.
"""
import os
import numpy as np
import pandas as pd

pd.set_option('display.width', 230); pd.set_option('display.unicode.east_asian_width', True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel.csv')
PARQ = os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_registered.parquet')
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
FEATS = ['dd_prod_early', 'over1']; YCOL = 'dd_ship_late'


def loco(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        b = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), X[tr]]), y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok]-pr[ok])**2).sum(); sst = ((y[ok]-y[ok].mean())**2).sum()
    return float(1 - sse/sst) if sst > 0 else np.nan


def eval_spec(d, feats=FEATS):
    r2s, betas, n = [], [], None
    for s in SPLITS:
        sub = d[d['split'] == s].dropna(subset=feats + [YCOL, 'mid'])
        if len(sub) < 10: r2s.append(np.nan); continue
        n = len(sub)
        X = sub[feats].to_numpy(float); y = sub[YCOL].to_numpy(float)
        r2s.append(loco(X, y, sub['mid'].to_numpy()))
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        betas.append(b[2])
    return float(np.nanmean(r2s)), (float(np.mean(betas)) if betas else np.nan), n


d = pd.read_csv(PANEL, encoding='utf-8-sig')
d.columns = [c.strip().lstrip('﻿') for c in d.columns]
d['over1'] = (d['padi_e'] > 1.0).astype(float)
d['cell'] = d['dd_prod_early'].round(6).astype(str) + '|' + d['dd_ship_late'].round(6).astype(str)
try:
    k = pd.read_parquet(PARQ, columns=['세세분류코드', '세세분류명'])
    k['세세분류코드'] = pd.to_numeric(k['세세분류코드'], errors='coerce')
    NAMES = k.dropna(subset=['세세분류코드']).astype({'세세분류코드': int}) \
             .drop_duplicates('세세분류코드').set_index('세세분류코드')['세세분류명'].to_dict()
except Exception:
    NAMES = {}

base, bbeta, bn = eval_spec(d)
print('=' * 110)
print('(M) 결과셀 단위 leave-one-CELL-out — 유효 단위에서의 영향점 집중도')
print('=' * 110)
print(f'  기준 47산업 LOCO평균 = {base:+.4f}  (유효 결과셀 {d[d.split==2017]["cell"].nunique()}개)')
s17 = d[d['split'] == 2017]
cells = sorted(s17['cell'].unique())
rec = []
for c in cells:
    ks = sorted(int(x) for x in s17.loc[s17['cell'] == c, 'ksic'])
    m, bb, n = eval_spec(d[~d['ksic'].isin(ks)])
    rec.append(dict(셀=c[:22], 산업수=len(ks),
                    산업=','.join(map(str, ks)),
                    신규수=sum(x in NEW8 for x in ks),
                    LOCO=m, dLOCO=m - base, beta_over1=bb, 잔여n=n))
M = pd.DataFrame(rec).sort_values('dLOCO', ascending=False).reset_index(drop=True)
print()
print('  ▶ 셀 제거 시 LOCO 개선 상위 8개 (각 행의 잔여 n 명기)')
print(M.head(8)[['산업', '산업수', '신규수', 'LOCO', 'dLOCO', 'beta_over1', '잔여n']]
      .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print()
pos = M[M['dLOCO'] > 0]
print(f'  · dLOCO>0 셀 = {len(pos)}/{len(M)}  |  개선분 총합 대비 상위1 비중 = '
      f'{M["dLOCO"].max()/pos["dLOCO"].sum()*100:.1f}%, 상위3 비중 = '
      f'{M.nlargest(3,"dLOCO")["dLOCO"].sum()/pos["dLOCO"].sum()*100:.1f}%')
print(f'  · 최대 단일 셀 개선폭 = {M["dLOCO"].max():+.4f}  (산업 단위 최대 +0.0397 의 '
      f'{M["dLOCO"].max()/0.0397:.1f}배)')

print()
print('  ▶ 누적 제거: dLOCO 상위 셀부터 k개씩 빼면 LOCO 는 어디까지 회복되는가')
rows = []
rm = []
for i in range(0, 6):
    if i > 0:
        rm += [int(x) for x in M.iloc[i-1]['산업'].split(',')]
    m, bb, n = eval_spec(d[~d['ksic'].isin(rm)])
    rows.append(dict(제거셀수=i, 제거산업수=len(rm), 잔여산업수=n, LOCO평균=m, beta_over1=bb))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('  → 소수의 셀 제거만으로 LOCO 가 크게 회복되면 "전반적 약화"가 아니라 "소수 영향점".')

print()
print('=' * 110)
print('(N) 신규8이 합류한 셀들을 통째로 제거 vs 신규8만 제거 — 무엇이 실제 원인인가')
print('=' * 110)
new_cells = sorted(set(s17.loc[s17['ksic'].isin(NEW8), 'cell']))
in_newcells = sorted(int(x) for x in s17.loc[s17['cell'].isin(new_cells), 'ksic'])
rows = []
for lab, rmk in [('제거 없음', []), ('신규8만 제거', NEW8),
                 ('신규8이 속한 셀 전체 제거', in_newcells)]:
    m, bb, n = eval_spec(d[~d['ksic'].isin(rmk)])
    dd17 = d[(d.split == 2017) & (~d['ksic'].isin(rmk))]
    rows.append(dict(처리=lab, 제거산업수=len(rmk), 잔여산업수=n,
                     잔여셀수=dd17['cell'].nunique(), LOCO평균=m, beta_over1=bb))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print(f'  신규8이 속한 셀에 포함된 산업 = {in_newcells}')
print(f'  그중 기존39 소속 = {[x for x in in_newcells if x not in NEW8]}')

print()
print('=' * 110)
print('(O) 39→47 하락(-0.0660)의 분해 — 신규 2개(새 셀) vs 신규 6개(중복 합류)')
print('=' * 110)
newcell_only = [27199, 27301]           # 새 결과셀을 만든 신규
dup_only = [k for k in NEW8 if k not in newcell_only]   # 기존 셀에 합류한 신규
rows = []
for lab, keep_new in [('39산업(신규 0개)', []),
                      ('39 + 새셀 신규2', newcell_only),
                      ('39 + 중복합류 신규6', dup_only),
                      ('47산업(신규 8개 전부)', NEW8)]:
    ks = set(d['ksic'].unique()) - (set(NEW8) - set(keep_new))
    dd = d[d['ksic'].isin(ks)]
    m, bb, n = eval_spec(dd)
    rows.append(dict(표본=lab, n_산업=n, n_셀=dd[dd.split == 2017]['cell'].nunique(),
                     LOCO평균=m, beta_over1=bb))
O = pd.DataFrame(rows)
print(O.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('  → 중복 합류 6개만 넣었을 때의 하락 vs 새 셀 2개만 넣었을 때의 하락을 비교하면,')
print('    "신규 산업의 이질성"과 "중복 가중 왜곡" 중 어느 쪽이 주 원인인지 갈린다.')
print()
print('완료.')
