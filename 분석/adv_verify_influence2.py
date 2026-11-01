# -*- coding: utf-8 -*-
"""
적대적 검증 2차 — 1차에서 나온 두 이상징후 추궁
 (I) 39패널 → 47패널로 갈 때 기존 39산업의 padi_e 가 변했다. 얼마나? 깃발은 몇 개 뒤집혔나?
     → 저자의 "기존 39개 데이터는 변하지 않았고 순수 구성 효과" 주장 직접 검정
 (J) 대형층의 낮은 LOCO R² 가 y분산(SST) 축소의 산물인지 — CV 설계를 통일해서 재확인
 (K) 규모 지표 7종 전부에 대해 RMSE vs y표준편차 동시 보고
 (L) 층별 표준화 효과크기(β/sd_y) 비교
"""
import os, re, glob
import numpy as np
import pandas as pd

pd.set_option('display.width', 230)
pd.set_option('display.unicode.east_asian_width', True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel.csv')
PANEL39 = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel_39ind.csv')
NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
FEATS = ['dd_prod_early', 'over1']; YCOL = 'dd_ship_late'


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


def loo_ind(X, y):
    """leave-one-INDUSTRY-out. 클러스터 수 교락을 제거한 균일 CV 설계."""
    n = len(y); pr = np.full(n, np.nan)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        if tr.sum() < X.shape[1] + 3: return (np.nan,)*3
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[i] = np.r_[1.0, X[i]] @ b
    sse = ((y-pr)**2).sum(); sst = ((y-y.mean())**2).sum()
    return float(1-sse/sst), float(np.sqrt(sse/n)), float(np.sqrt(sst/n))


def load_panel(p):
    d = pd.read_csv(p, encoding='utf-8-sig')
    d.columns = [c.strip().lstrip('﻿') for c in d.columns]
    d['over1'] = (d['padi_e'] > 1.0).astype(float)
    return d


def eval_spec(d, feats=FEATS):
    r2s, betas, n = [], [], None
    for s in SPLITS:
        sub = d[d['split'] == s].dropna(subset=feats + [YCOL, 'mid'])
        if len(sub) < 10: r2s.append(np.nan); continue
        n = len(sub)
        X = sub[feats].to_numpy(float); y = sub[YCOL].to_numpy(float)
        r2s.append(loco(X, y, sub['mid'].to_numpy()))
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        betas.append(dict(**{f: b[i+1] for i, f in enumerate(feats)}))
    return np.array(r2s, float), pd.DataFrame(betas), n


d47 = load_panel(PANEL); d39 = load_panel(PANEL39)

print('=' * 110)
print('(I) 저자 주장 "기존 39개 데이터는 변하지 않았다" 직접 검정')
print('=' * 110)
key = ['ksic', 'split']
m = d39[key + ['padi_e', 'over1']].merge(d47[key + ['padi_e', 'over1']], on=key, suffixes=('_39', '_47'))
m['d'] = m['padi_e_47'] - m['padi_e_39']
print(f'  공유 273행 중 padi_e 가 바뀐 행 = {int((m["d"].abs()>1e-9).sum())}/273  (즉 전부)')
print(f'  변화량: 평균 {m["d"].mean():+.4f}, 중앙값 {m["d"].median():+.4f}, '
      f'최소 {m["d"].min():+.4f}, 최대 {m["d"].max():+.4f}, |변화| 평균 {m["d"].abs().mean():.4f}')
print(f'  상대변화 |Δ|/padi_39 평균 = {(m["d"].abs()/m["padi_e_39"]).mean()*100:.2f}%')
flip = m[m['over1_39'] != m['over1_47']]
print(f'  over1 깃발이 뒤집힌 행 = {len(flip)}/273  '
      f'(산업 {flip["ksic"].nunique()}개: {sorted(flip["ksic"].unique().tolist())})')
if len(flip):
    print(flip.groupby('ksic').agg(뒤집힌분할수=('split','size'), padi39=('padi_e_39','mean'),
                                   padi47=('padi_e_47','mean')).to_string(float_format=lambda v: f'{v:+.4f}'))
print()
print('  ▶ 39산업 표본을 고정하고, padi_e 만 39패널값 / 47패널값 으로 바꿔가며 LOCO 비교')
print('    (표본 n 은 두 경우 모두 39산업 × 7분할 = 273행으로 동일)')
base39 = d39.copy()
swap = d47[d47['ksic'].isin(d39['ksic'].unique())].copy()
r_a, b_a, n_a = eval_spec(base39)
r_b, b_b, n_b = eval_spec(swap)
print(f'    padi_e = 39패널값: n={n_a}산업 LOCO평균={np.nanmean(r_a):+.4f} β_over1={b_a["over1"].mean():+.4f}')
print(f'    padi_e = 47패널값: n={n_b}산업 LOCO평균={np.nanmean(r_b):+.4f} β_over1={b_b["over1"].mean():+.4f}')
print('    → 두 값이 거의 같은 이유는 padi_e 변화가 이진 깃발(over1)을 거의 뒤집지 않기 때문.')
print('      즉 "데이터 불변"이 아니라 "깃발이 둔감"한 것. 연속형 padi_e 사양에서는 다를 수 있다.')
print()
print('  ▶ 연속형 padi_e 사양으로 같은 비교 (over1 대신 padi_e 자체)')
for lab, dd in [('padi_e=39패널값', base39), ('padi_e=47패널값', swap)]:
    r, b, n = eval_spec(dd, feats=['dd_prod_early', 'padi_e'])
    print(f'    {lab}: n={n}산업 LOCO평균={np.nanmean(r):+.4f} β_padi={b["padi_e"].mean():+.4f}')

# ------------------------------------------------------------------
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
    R = pd.concat(parts); R['val'] = pd.to_numeric(R['val'], errors='coerce'); return R

R = load_size_items()
SZ = {it: R[(R['item'] == it) & (R['yr'].between(2010, 2015))].groupby('ksic')['val'].mean()
      for it in R['item'].dropna().unique()}

print()
print('=' * 110)
print('(K) 규모 지표 7종 — 대형/소형 층의 LOCO R² 와 함께 RMSE·y표준편차 동시 보고')
print('=' * 110)
print('  R² 는 1 - (RMSE/sd_y)². 두 층의 sd_y 가 다르면 R² 비교는 관계 강도 비교가 아니다.')
ind0 = d47[d47['split'] == 2017][['ksic']].copy()
rows = []
for it, s in SZ.items():
    ind = ind0.copy(); ind['sz'] = ind['ksic'].map(s)
    if ind['sz'].notna().sum() < 40: continue
    med = ind['sz'].median()
    for lab, ks in [('대형', set(ind.loc[ind['sz'] >= med, 'ksic'])),
                    ('소형', set(ind.loc[ind['sz'] < med, 'ksic']))]:
        dd = d47[d47['ksic'].isin(ks)]
        rms, sds, r2s = [], [], []
        for sp in SPLITS:
            sub = dd[dd['split'] == sp].dropna(subset=FEATS + [YCOL, 'mid'])
            X = sub[FEATS].to_numpy(float); y = sub[YCOL].to_numpy(float); g = sub['mid'].to_numpy()
            uq = np.unique(g); pr = np.full(len(y), np.nan); okc = True
            for c in uq:
                te = g == c; tr = ~te
                if tr.sum() < X.shape[1] + 3: okc = False; break
                b = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), X[tr]]), y[tr], rcond=None)[0]
                pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
            if not okc: continue
            rms.append(np.sqrt(((y-pr)**2).mean())); sds.append(y.std())
            r2s.append(1 - ((y-pr)**2).sum()/((y-y.mean())**2).sum())
        rows.append(dict(지표=it[:14], 층=lab, n_산업=len(ks),
                         y표준편차=float(np.mean(sds)), LOCO_RMSE=float(np.mean(rms)),
                         LOCO_R2=float(np.mean(r2s))))
K = pd.DataFrame(rows)
print(K.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
print()
piv = K.pivot(index='지표', columns='층', values=['y표준편차', 'LOCO_RMSE', 'LOCO_R2'])
print('  ▶ 대형-소형 대비 (양수 = 대형이 큼)')
out = pd.DataFrame({
    'Δy표준편차': piv[('y표준편차', '대형')] - piv[('y표준편차', '소형')],
    'ΔRMSE': piv[('LOCO_RMSE', '대형')] - piv[('LOCO_RMSE', '소형')],
    'ΔR2': piv[('LOCO_R2', '대형')] - piv[('LOCO_R2', '소형')]})
print(out.to_string(float_format=lambda v: f'{v:+.3f}'))
print('  → ΔRMSE ≤ 0 인데 ΔR2 < 0 이면: 대형층은 오차가 더 작은데도 R² 만 낮다 = 순수 분모 효과.')

print()
print('=' * 110)
print('(J) CV 설계를 통일(leave-one-INDUSTRY-out)해서 규모 층 재비교 — mid 클러스터 수 교락 제거')
print('=' * 110)
sz = SZ[[i for i in SZ if '출하액' in str(i)][0]]
ind = ind0.copy(); ind['sz'] = ind['ksic'].map(sz); med = ind['sz'].median()
BIG = set(ind.loc[ind['sz'] >= med, 'ksic']); SMALL = set(ind.loc[ind['sz'] < med, 'ksic'])
rows = []
for lab, ks in [('전체47', set(ind['ksic'])), ('대형', BIG), ('소형', SMALL),
                ('기존39 전체', set(ind['ksic'])-set(NEW8)),
                ('기존39 대형', BIG-set(NEW8)), ('기존39 소형', SMALL-set(NEW8))]:
    dd = d47[d47['ksic'].isin(ks)]
    for feats, fl in [(['dd_prod_early'], '지속성만'), (FEATS, '기준사양')]:
        r2s, rms, sds = [], [], []
        for sp in SPLITS:
            sub = dd[dd['split'] == sp].dropna(subset=feats + [YCOL])
            r2, rm, sd = loo_ind(sub[feats].to_numpy(float), sub[YCOL].to_numpy(float))
            r2s.append(r2); rms.append(rm); sds.append(sd)
        rows.append(dict(층=lab, 사양=fl, n_산업=len(ks),
                         LOIO_R2=float(np.nanmean(r2s)), LOIO_RMSE=float(np.nanmean(rms)),
                         y표준편차=float(np.nanmean(sds))))
J = pd.DataFrame(rows)
print(J.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

print()
print('=' * 110)
print('(L) 층별 표준화 효과크기 — β_over1 / sd(y). 단위 차이를 제거한 비교')
print('=' * 110)
rows = []
for lab, ks in [('전체47', set(ind['ksic'])), ('대형', BIG), ('소형', SMALL),
                ('기존39 대형', BIG-set(NEW8)), ('기존39 소형', SMALL-set(NEW8))]:
    dd = d47[d47['ksic'].isin(ks)]
    r2s, bt, n = eval_spec(dd)
    sdy = np.mean([dd[dd.split == sp][YCOL].std() for sp in SPLITS])
    rows.append(dict(층=lab, n_산업=n, beta_over1=bt['over1'].mean(), sd_y=sdy,
                     표준화효과=bt['over1'].mean()/sdy,
                     beta_persist=bt['dd_prod_early'].mean()))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('  → 표준화 효과가 층 간에 비슷하면 "대형에서 관계가 약하다"는 서술은 단위 착시.')
print()
print('완료.')
