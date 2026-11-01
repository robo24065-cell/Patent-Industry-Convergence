# -*- coding: utf-8 -*-
"""synth_final_scope 보완: (a) 블록자체평균 대비 skill, (b) 절편(주효과)만 추가한
   사양, (c) 결과셀 축약 표본, (d) 무작위 8개 제외 귀무분포 대비 위치."""
import numpy as np, pandas as pd

NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = list(range(2013, 2020))
pan = pd.read_csv('model_search_panel.csv')
pan['over1'] = (pan['padi_e'] > 1.0).astype(float)
pan['elec'] = pan['mid'].isin([26, 27, 28]).astype(float)
pan['new8'] = pan['ksic'].isin(NEW8).astype(float)


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


def run(d, cols):
    out = []
    for s in SPLITS:
        sub = d[d['split'] == s].dropna(subset=list(cols)+['dd_ship_late'])
        X = sub[list(cols)].values.astype(float); y = sub['dd_ship_late'].values.astype(float)
        out.append(loco(X, y, sub['mid'].values))
    return float(np.nanmean(out)), float(np.nanmin(out))


# (b) 주효과만 추가한 사양 (기울기 공유, 절편만 분리)
for cols, tag in [(('dd_prod_early', 'over1'), '기준'),
                  (('dd_prod_early', 'over1', 'elec'), '+전자더미(주효과만)'),
                  (('dd_prod_early', 'over1', 'new8'), '+신규8더미(주효과만)')]:
    m, mn = run(pan, cols)
    print(f'{tag:22s} n산업=47 n행=329  LOCO 평균 {m:+.4f} 최소 {mn:+.4f}')

# (a) 블록 자체 평균 대비 skill (같은 폴드, 블록별 훈련평균을 기준선으로)
print('\n[블록 자체평균 대비 skill]  같은 47 LOCO 실행, 블록 훈련평균을 대조')
acc = {}
for s in SPLITS:
    sub = pan[pan['split'] == s].dropna(subset=['dd_prod_early', 'over1', 'dd_ship_late']).copy()
    sub['blk'] = np.where(sub['new8'] == 1, '신규8',
                          np.where(sub['elec'] == 1, '기존39전자기계', '기존39비전자'))
    pr = np.full(len(sub), np.nan); nb = np.full(len(sub), np.nan)
    X = sub[['dd_prod_early', 'over1']].values.astype(float)
    y = sub['dd_ship_late'].values.astype(float); g = sub['mid'].values
    for c in np.unique(g):
        te = g == c; tr = ~te
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
        for blk in sub['blk'].unique():
            mk = (sub['blk'].values == blk)
            trb = tr & mk
            if trb.sum() > 0:
                nb[te & mk] = y[trb].mean()
    sub['pr'] = pr; sub['nb'] = nb
    for blk, q in sub.groupby('blk'):
        acc.setdefault(blk, []).append((((q.dd_ship_late-q.pr)**2).sum(),
                                        ((q.dd_ship_late-q.nb)**2).sum(), len(q)))
for blk, v in acc.items():
    sse = sum(a for a, b, c in v); ssn = sum(b for a, b, c in v); n = sum(c for a, b, c in v)
    print(f'  {blk:16s} n행={n:4d} RMSE={np.sqrt(sse/n):6.2f} '
          f'블록평균RMSE={np.sqrt(ssn/n):6.2f} skill={1-sse/ssn:+.3f}')

# (c) 결과셀 축약 (중복 제거, 셀당 1행)
print('\n[결과셀 축약]')
for nm, d in [('47산업', pan), ('39산업', pan[~pan.ksic.isin(NEW8)])]:
    r2 = []
    for s in SPLITS:
        sub = d[d['split'] == s].copy()
        cell = sub.groupby(['dd_prod_early', 'dd_ship_late']).agg(
            over1=('over1', 'mean'), mid=('mid', 'first')).reset_index()
        r2.append(loco(cell[['dd_prod_early', 'over1']].values.astype(float),
                       cell['dd_ship_late'].values.astype(float), cell['mid'].values))
    print(f'  {nm} 축약: 셀수={len(cell)} LOCO 평균 {np.nanmean(r2):+.4f} 최소 {np.nanmin(r2):+.4f}')

# (d) 무작위 8개 제외 귀무분포 (n산업=39 고정)
rng = np.random.default_rng(20260813)
ind = sorted(pan['ksic'].unique())
null = []
for _ in range(2000):
    drop = rng.choice(ind, 8, replace=False)
    m, _mn = run(pan[~pan.ksic.isin(drop)], ('dd_prod_early', 'over1'))
    null.append(m)
null = np.array(null)
obs_new8 = run(pan[~pan.ksic.isin(NEW8)], ('dd_prod_early', 'over1'))[0]
print(f'\n[무작위 8개 제외 귀무분포 n=2000, 모두 n산업=39]')
print(f'  중위 {np.median(null):+.4f}  5% {np.percentile(null,5):+.4f}  '
      f'95% {np.percentile(null,95):+.4f}')
print(f'  신규8 제외 {obs_new8:+.4f} → 상위 {(null>=obs_new8).mean()*100:.1f}%')

# 전자·기계 20개 제외 vs 무작위 20개 제외 (n산업=27 고정)
null20 = []
for _ in range(1000):
    drop = rng.choice(ind, 20, replace=False)
    m, _mn = run(pan[~pan.ksic.isin(drop)], ('dd_prod_early', 'over1'))
    null20.append(m)
null20 = np.array(null20)
obs_e = run(pan[~pan.mid.isin([26, 27, 28])], ('dd_prod_early', 'over1'))[0]
print(f'[무작위 20개 제외 n=1000, 모두 n산업=27] 중위 {np.median(null20):+.4f} '
      f'95% {np.percentile(null20,95):+.4f}')
print(f'  전자·기계20 제외 {obs_e:+.4f} → 상위 {(null20>=obs_e).mean()*100:.1f}%')
