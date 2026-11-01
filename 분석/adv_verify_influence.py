# -*- coding: utf-8 -*-
"""
적대적 검증: diag_influence.py 의 '영향점 진단' 결론에 대한 대안설명 탐색.
기존 스크립트 수정 없음. 새 파일.

검증 축:
 (A) 대형층 LOCO 저하가 '관계 약화'가 아니라 SST(종속변수 분산) 분모 축소의 산물인가
 (B) 대형층에서 무너진 것이 PADI 축뿐인가, 아니면 지속성 축까지 함께 무너지는가
 (C) 규모 반분 결과가 결과셀 중복 가중을 제거해도 살아남는가
 (D) 39패널과 47패널의 공유 39산업 데이터가 정말 동일한가
 (E) 신규8의 LOO '유해성'이 기존 셀메이트에서 상속된 것인가 (셀 귀속 검정)
 (F) 규모 지표를 바꾸면(종사자수/부가가치) 층 결과가 유지되는가
 (G) 규모를 층이 아니라 3분위/다른 절단점으로 나눠도 단조적인가
"""
import os, re, glob
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)
pd.set_option('display.unicode.east_asian_width', True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel.csv')
PANEL39 = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel_39ind.csv')

NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
FEATS = ['dd_prod_early', 'over1']
YCOL = 'dd_ship_late'


def loco(X, y, g):
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try: b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception: return np.nan
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok]-pr[ok])**2).sum(); sst = ((y[ok]-y[ok].mean())**2).sum()
    return float(1 - sse/sst) if sst > 0 else np.nan


def loco_parts(X, y, g):
    """LOCO 의 SSE, SST, RMSE 를 분해해서 반환."""
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return (np.nan,)*4
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok]-pr[ok])**2).sum(); sst = ((y[ok]-y[ok].mean())**2).sum()
    n = ok.sum()
    return (float(np.sqrt(sse/n)), float(np.sqrt(sst/n)), float(sse), float(sst))


def load_panel(p):
    d = pd.read_csv(p, encoding='utf-8-sig')
    d.columns = [c.strip().lstrip('﻿') for c in d.columns]
    d['over1'] = (d['padi_e'] > 1.0).astype(float)
    return d


def eval_spec(d, feats=FEATS, ycol=YCOL):
    r2s, betas, n = [], [], None
    for s in SPLITS:
        sub = d[d['split'] == s].dropna(subset=feats + [ycol, 'mid'])
        if len(sub) < 10:
            r2s.append(np.nan); continue
        n = len(sub)
        X = sub[feats].to_numpy(float); y = sub[ycol].to_numpy(float)
        r2s.append(loco(X, y, sub['mid'].to_numpy()))
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        betas.append(dict(split=s, **{f: b[i+1] for i, f in enumerate(feats)}))
    return np.array(r2s, float), pd.DataFrame(betas), n


def load_size_items():
    fs = sorted(glob.glob(os.path.join(ROOT, '공모전데이터', '외부데이터', '산업분류별', '*.csv')))
    parts = []
    for p in fs:
        try: d = pd.read_csv(p, encoding='cp949')
        except Exception: continue
        if '[B]산업별' not in d.columns: continue
        d['code'] = d['[B]산업별'].astype(str)
        d = d[d['code'].str.fullmatch(r'[A-Z]\d{5}')].copy()
        d['ksic'] = d['code'].str[1:].astype(int)
        for c in [c for c in d.columns if re.match(r'^\d{4}\s*년', str(c))]:
            t = d[['ksic', '항목', c]].copy(); t.columns = ['ksic', 'item', 'val']
            t['yr'] = int(str(c)[:4]); parts.append(t)
    R = pd.concat(parts)
    R['val'] = pd.to_numeric(R['val'], errors='coerce')
    return R


d47 = load_panel(PANEL)
d39 = load_panel(PANEL39)
d47['cell'] = d47['dd_prod_early'].round(6).astype(str) + '|' + d47['dd_ship_late'].round(6).astype(str)

print('=' * 105)
print('(D) 39패널 vs 47패널 — 공유 39산업의 값이 정말 동일한가')
print('=' * 105)
k39 = sorted(d39['ksic'].unique()); k47 = sorted(d47['ksic'].unique())
print(f'  39패널 산업수={len(k39)} / 47패널 산업수={len(k47)} / 39패널이 47패널의 부분집합인가: {set(k39) <= set(k47)}')
print(f'  47-39 차집합 = {sorted(set(k47)-set(k39))}')
key = ['ksic', 'split']
cmpcols = ['padi_e', 'dd_prod_early', 'dd_ship_late', 'dd_prod_late', 'dd_ship_early']
cmpcols = [c for c in cmpcols if c in d39.columns and c in d47.columns]
m = d39[key + cmpcols].merge(d47[key + cmpcols], on=key, suffixes=('_39', '_47'))
print(f'  병합행수={len(m)} (기대 39*7=273)')
for c in cmpcols:
    dif = (m[c + '_39'] - m[c + '_47']).abs()
    print(f'    {c:16s} 최대절대차이={dif.max():.10g}  불일치행수={int((dif > 1e-9).sum())}')

# 39패널만으로 재적합 vs 47패널에서 신규8 제거 후 재적합
r_a, b_a, n_a = eval_spec(d39)
r_b, b_b, n_b = eval_spec(d47[~d47['ksic'].isin(NEW8)])
print(f'  39패널 LOCO평균={np.nanmean(r_a):+.6f} (n={n_a}산업)  |  47패널에서 신규8제거 LOCO평균={np.nanmean(r_b):+.6f} (n={n_b}산업)')
print(f'  차이={np.nanmean(r_a)-np.nanmean(r_b):+.6f}  → 0이면 "기존 데이터 불변, 순수 구성효과" 주장 성립')

# ------------------------------------------------------------------ 규모
R = load_size_items()
print()
print('=' * 105)
print('(F) 규모 지표 대안 — 광업제조업조사 항목 목록')
print('=' * 105)
items = [str(i) for i in R['item'].dropna().unique()]
print('  ', items)

SIZEDEF = {}
for it in items:
    s = (R[(R['item'] == it) & (R['yr'].between(2010, 2015))]
         .groupby('ksic')['val'].mean())
    SIZEDEF[it] = s

base_ind = d47[d47['split'] == 2017][['ksic', 'mid', 'padi_e', 'over1', 'dd_prod_early', 'dd_ship_late', 'cell']].copy()
base_ind['신규'] = base_ind['ksic'].isin(NEW8)

print()
print('=' * 105)
print('(F/G) 규모 지표·절단점을 바꿔가며 대형/소형 층 LOCO 재계산  [모든 행에 n 명기]')
print('=' * 105)
rows = []
for it, s in SIZEDEF.items():
    ind = base_ind.copy()
    ind['sz'] = ind['ksic'].map(s)
    if ind['sz'].notna().sum() < 40:
        print(f'  [건너뜀] {it}: 매칭 {ind["sz"].notna().sum()}/47')
        continue
    med = ind['sz'].median()
    for lab, ks in [('대형(상위50%)', set(ind.loc[ind['sz'] >= med, 'ksic'])),
                    ('소형(하위50%)', set(ind.loc[ind['sz'] < med, 'ksic']))]:
        dd = d47[d47['ksic'].isin(ks)]
        r2s, bt, n = eval_spec(dd)
        cells = dd[dd.split == 2017]['cell'].nunique()
        rows.append(dict(지표=it[:16], 층=lab, n_산업=n, 유효셀=cells,
                         유효mid=dd[dd.split == 2017]['mid'].nunique(),
                         over1수=int(dd[dd.split == 2017]['over1'].sum()),
                         LOCO평균=float(np.nanmean(r2s)), beta_over1=bt['over1'].mean()))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

# 기준 지표(출하액) 고정
SZ = None
for it in SIZEDEF:
    if '출하액' in it: SZ = SIZEDEF[it]; SZNAME = it
ind = base_ind.copy(); ind['sz'] = ind['ksic'].map(SZ)
med = ind['sz'].median()
ind['대형'] = ind['sz'] >= med
BIG = set(ind.loc[ind['대형'], 'ksic']); SMALL = set(ind.loc[~ind['대형'], 'ksic'])

print()
print('=' * 105)
print('(A) 대형층의 낮은 LOCO 는 관계 약화인가, SST(y 분산) 축소의 산물인가')
print('=' * 105)
print('  LOCO R² = 1 - SSE/SST. SST 가 작으면 예측오차가 같아도 R² 는 낮아진다.')
rows = []
for lab, ks in [('전체47', set(ind['ksic'])), ('대형', BIG), ('소형', SMALL),
                ('기존39 대형', BIG - set(NEW8)), ('기존39 소형', SMALL - set(NEW8))]:
    dd = d47[d47['ksic'].isin(ks)]
    rm, sd, sse, sst = [], [], [], []
    for s in SPLITS:
        sub = dd[dd['split'] == s].dropna(subset=FEATS + [YCOL, 'mid'])
        a, b, c, e = loco_parts(sub[FEATS].to_numpy(float), sub[YCOL].to_numpy(float), sub['mid'].to_numpy())
        rm.append(a); sd.append(b)
    dd17 = dd[dd.split == 2017]
    rows.append(dict(층=lab, n_산업=dd17.shape[0],
                     y표준편차=float(np.nanmean(sd)), LOCO_RMSE=float(np.nanmean(rm)),
                     LOCO_R2=1 - float(np.nanmean(np.array(rm)**2)) / float(np.nanmean(np.array(sd)**2)),
                     y평균=dd17[YCOL].mean()))
A = pd.DataFrame(rows)
print(A.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('  → 대형층의 y표준편차가 소형층보다 크게 작다면, 낮은 R² 는 "관계 약화"가 아니라 분모 효과.')
print('  → RMSE 를 직접 비교하는 것이 관계 강도의 더 정직한 지표.')

print()
print('=' * 105)
print('(B) 대형층에서 무너진 것이 PADI 축뿐인가 — 지속성만 모형과의 비교')
print('=' * 105)
rows = []
for lab, ks in [('전체47', set(ind['ksic'])), ('대형', BIG), ('소형', SMALL),
                ('기존39 전체', set(ind['ksic']) - set(NEW8)),
                ('기존39 대형', BIG - set(NEW8)), ('기존39 소형', SMALL - set(NEW8))]:
    dd = d47[d47['ksic'].isin(ks)]
    r_p, b_p, n = eval_spec(dd, feats=['dd_prod_early'])
    r_f, b_f, _ = eval_spec(dd, feats=FEATS)
    rows.append(dict(층=lab, n_산업=n,
                     지속성만_LOCO=float(np.nanmean(r_p)),
                     기준사양_LOCO=float(np.nanmean(r_f)),
                     PADI기여=float(np.nanmean(r_f)) - float(np.nanmean(r_p)),
                     beta_persist=b_f['dd_prod_early'].mean(),
                     beta_over1=b_f['over1'].mean()))
B = pd.DataFrame(rows)
print(B.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('  → 대형층에서 지속성만 모형까지 함께 붕괴한다면 "PADI 관계가 대형에 없다"가 아니라')
print('    "대형층에서는 어떤 모형도 안 맞는다(종속변수 품질/중복 문제)"는 제3설명이 성립한다.')

print()
print('=' * 105)
print('(C) 규모 반분 결과가 결과셀 축약(중복 가중 제거) 후에도 살아남는가')
print('=' * 105)
def eval_cells(dd):
    r2s, betas = [], []
    for s in SPLITS:
        ss = dd[dd['split'] == s]
        cg = ss.groupby('cell').agg(y=(YCOL, 'mean'), x=('dd_prod_early', 'mean'),
                                    o=('over1', 'mean'), mid=('mid', lambda v: v.mode().iloc[0]))
        if len(cg) < 8:
            r2s.append(np.nan); continue
        X = cg[['x', 'o']].to_numpy(float); y = cg['y'].to_numpy(float)
        r2s.append(loco(X, y, cg['mid'].to_numpy()))
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        betas.append(b[2])
    return float(np.nanmean(r2s)), float(np.mean(betas)) if betas else np.nan, len(cg)

rows = []
for lab, ks in [('전체47', set(ind['ksic'])), ('대형', BIG), ('소형', SMALL),
                ('기존39 대형', BIG - set(NEW8)), ('기존39 소형', SMALL - set(NEW8))]:
    dd = d47[d47['ksic'].isin(ks)]
    r_raw, b_raw, n = eval_spec(dd)
    r_c, b_c, ncell = eval_cells(dd)
    rows.append(dict(층=lab, n_산업=n, n_셀=ncell,
                     원본_LOCO=float(np.nanmean(r_raw)), 셀축약_LOCO=r_c,
                     원본_beta=b_raw['over1'].mean(), 셀축약_beta=b_c))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('  → 대형층은 24산업 중 유효셀 13개(중복률 46%), 소형층은 23산업 중 17개(26%).')
print('    셀 축약 후 격차가 사라지면 "규모 효과"가 아니라 "중복 가중 효과"였다는 뜻.')

print()
print('=' * 105)
print('(E) 신규8의 LOO 유해성은 새 정보인가, 기존 셀메이트에서 상속된 것인가')
print('=' * 105)
# 39패널에서 LOO 를 돌려 각 기존 산업의 유해도(dLOCO) 산출
r0, b0, n0 = eval_spec(d39)
base39 = float(np.nanmean(r0))
rec = []
for k in sorted(d39['ksic'].unique()):
    r2s, bt, n = eval_spec(d39[d39['ksic'] != k])
    rec.append(dict(ksic=int(k), dLOCO39=float(np.nanmean(r2s)) - base39))
L39 = pd.DataFrame(rec).sort_values('dLOCO39', ascending=False).reset_index(drop=True)
L39['rank39'] = np.arange(1, len(L39) + 1)

# 신규8이 합류한 셀의 '기존 셀메이트'가 39패널 LOO 에서 몇 위였나
s17 = d47[d47['split'] == 2017]
print('  신규8 각각: 합류한 셀의 기존 셀메이트가 39패널 LOO 유해도 순위 몇 위였나 (1위=가장 해로움/39)')
rows = []
for k in NEW8:
    cell = s17.loc[s17['ksic'] == k, 'cell'].iloc[0]
    mates = [int(x) for x in s17.loc[(s17['cell'] == cell) & (s17['ksic'] != k), 'ksic'] if int(x) not in NEW8]
    rr = L39[L39['ksic'].isin(mates)]
    rows.append(dict(신규=k, 기존셀메이트=','.join(map(str, mates)) or '(없음-새 셀)',
                     셀메이트_39LOO순위=','.join(f'{v}' for v in rr['rank39']) or '-',
                     셀메이트_최고순위=int(rr['rank39'].min()) if len(rr) else np.nan))
print(pd.DataFrame(rows).to_string(index=False))

# 47패널 LOO 유해도를 다시 계산해 셀 단위로 묶기
r0b, b0b, _ = eval_spec(d47); base47 = float(np.nanmean(r0b))
rec = []
for k in sorted(d47['ksic'].unique()):
    r2s, bt, n = eval_spec(d47[d47['ksic'] != k])
    rec.append(dict(ksic=int(k), dLOCO=float(np.nanmean(r2s)) - base47))
L47 = pd.DataFrame(rec).sort_values('dLOCO', ascending=False).reset_index(drop=True)
L47['rank'] = np.arange(1, len(L47) + 1)
L47 = L47.merge(s17[['ksic', 'cell']].astype({'ksic': int}), on='ksic', how='left')
L47['신규'] = L47['ksic'].isin(NEW8)

# 동률(같은 셀은 dLOCO 가 동일) 확인
tie = L47.groupby(L47['dLOCO'].round(8)).size()
print()
print(f'  47패널 LOO dLOCO 의 고유값 개수 = {L47["dLOCO"].round(8).nunique()} / 47산업  '
      f'(동률군 최대 크기 {tie.max()})')
print('  → 같은 결과셀 산업은 dLOCO 가 사실상 동일. 즉 LOO 순위는 산업이 아니라 셀의 성질.')

# 셀 단위 순열검정: 셀을 단위로 무작위 배정
cellmean = L47.groupby('cell')['dLOCO'].mean().rename('cell_dLOCO')
L47 = L47.merge(cellmean, on='cell')
obs = L47.loc[L47['신규'], 'rank'].sum()
rng = np.random.default_rng(20260813)
perm = np.array([rng.choice(np.arange(1, 48), 8, replace=False).sum() for _ in range(20000)])
print(f'  [저자 검정 재현] 산업 단위 순위합 순열검정: 관측={obs}, p={float((perm<=obs).mean()):.4f}')

# 대안 검정 1: 신규8 중 기존 셀에 합류한 6개는 셀메이트와 dLOCO 가 동일 → 정보 없음.
#            새 셀을 만든 2개(27199, 27301)만으로 검정
newcell = [27199, 27301]
sub = L47[(~L47['신규']) | (L47['ksic'].isin(newcell))]
print(f'  [대안1] 기존셀 합류 6개 제외 후(신규 중 새 셀 2개만): '
      f'그 2개의 순위 = {sorted(L47.loc[L47.ksic.isin(newcell),"rank"].tolist())} / 47')

# 대안 검정 2: 셀 단위 순열 (셀을 단위로 8개 산업을 뽑되 같은 셀은 함께 움직임)
cells = s17[['ksic', 'cell']].astype({'ksic': int})
cell_of = dict(zip(cells['ksic'], cells['cell']))
rank_of = dict(zip(L47['ksic'], L47['rank']))
uniq_cells = sorted(set(cell_of.values()))
members = {c: [k for k in cell_of if cell_of[k] == c] for c in uniq_cells}
sims = []
for _ in range(20000):
    order = rng.permutation(uniq_cells); picked = []
    for c in order:
        picked += members[c]
        if len(picked) >= 8: break
    picked = picked[:8]
    sims.append(sum(rank_of[k] for k in picked))
sims = np.array(sims)
print(f'  [대안2] 셀 단위 순열검정(같은 셀은 함께 표집): 관측 순위합={obs}, p={float((sims<=obs).mean()):.4f}')
print('     → 산업 단위 순열은 셀 내부 동률을 독립 관측으로 취급해 p 를 과소평가한다.')

print()
print('=' * 105)
print('(G) 규모 절단점 민감도 — 3분위 / 상위25% / 상위33%')
print('=' * 105)
rows = []
for lab, mask in [('상위25%', ind['sz'] >= ind['sz'].quantile(0.75)),
                  ('상위33%', ind['sz'] >= ind['sz'].quantile(2/3)),
                  ('상위50%', ind['sz'] >= ind['sz'].quantile(0.50)),
                  ('하위50%', ind['sz'] < ind['sz'].quantile(0.50)),
                  ('하위33%', ind['sz'] < ind['sz'].quantile(1/3)),
                  ('중간33%', (ind['sz'] >= ind['sz'].quantile(1/3)) & (ind['sz'] < ind['sz'].quantile(2/3)))]:
    ks = set(ind.loc[mask, 'ksic'])
    dd = d47[d47['ksic'].isin(ks)]
    r2s, bt, n = eval_spec(dd)
    dd17 = dd[dd.split == 2017]
    rows.append(dict(층=lab, n_산업=n, 유효셀=dd17['cell'].nunique(), 유효mid=dd17['mid'].nunique(),
                     over1수=int(dd17['over1'].sum()),
                     LOCO평균=float(np.nanmean(r2s)) if np.isfinite(r2s).any() else np.nan,
                     beta_over1=bt['over1'].mean() if len(bt) else np.nan))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

print()
print('=' * 105)
print('(H) 층별 y 분산·중복률 요약 — 세 가지 후보설명의 교락 정도')
print('=' * 105)
s = d47[d47['split'] == 2017].copy()
s['sz'] = s['ksic'].map(SZ); s['대형'] = s['sz'] >= med
s['전자'] = s['mid'] == 26; s['신규'] = s['ksic'].isin(NEW8)
rows = []
for lab, m in [('전체47', np.ones(len(s), bool)), ('대형', s['대형'].values), ('소형', (~s['대형']).values),
               ('전자mid26', s['전자'].values), ('비전자', (~s['전자']).values),
               ('신규8', s['신규'].values), ('기존39', (~s['신규']).values)]:
    a = s[m]
    rows.append(dict(층=lab, n=len(a), 유효셀=a['cell'].nunique(),
                     중복률=1 - a['cell'].nunique()/len(a),
                     y표준편차=a[YCOL].std(), y평균=a[YCOL].mean(),
                     x표준편차=a['dd_prod_early'].std(),
                     over1비율=a['over1'].mean()))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
print()
print('완료.')
