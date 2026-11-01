# s2: C04-A/C P2(사후 표지)·P3(연결 편향) 집계. 개발 원점만(잠금 S0는 모집단 크기만 s1에서).
import sys, json
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libA import *
ap = pd.read_parquet(C + r'\A_applicants.parquet')
ap = ap[ap.s0.isin(DEV_A)].copy()
u = pd.read_parquet(C + r'\A_units_dev.parquet')
rows = []
def add(cand, check, metric, value, note=''):
    rows.append(dict(candidate=cand, check=check, metric=metric, value=value, note=note)); print(cand, check, metric, value, note)
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return round(float((a.mean() - b.mean()) / sp), 3) if sp > 0 else 0.0
ap['log_nvis'] = np.log1p(ap.n_vis)
L, N = ap[ap.linked], ap[~ap.linked]
for v in ['log_nvis', 'first_ay', 'reg_by_ref', 'dom_prio', 'later_reg']:
    add('A/C', 'P3' if v != 'later_reg' else 'P2b', f'SMD 연결-비연결 {v}', smd(L[v], N[v]), f'연결 평균 {L[v].mean():.3f} / 비연결 {N[v].mean():.3f}')
# 원점별 later_reg SMD
for s0, g in ap.groupby('s0'):
    add('A/C', 'P2b', f'SMD later_reg {s0}', smd(g[g.linked].later_reg, g[~g.linked].later_reg))
# KSIC TVD
for s0, g in ap.groupby('s0'):
    pl = g[g.linked].ksic_mode.value_counts(normalize=True); pn = g[~g.linked].ksic_mode.value_counts(normalize=True)
    tvd = 0.5 * pl.reindex(pl.index.union(pn.index), fill_value=0).sub(pn.reindex(pl.index.union(pn.index), fill_value=0)).abs().sum()
    add('A/C', 'P3', f'KSIC 최빈 TVD 연결-비연결 {s0}', round(float(tvd), 3))
# KSIC×원점 칸 공존(양성 가중)
ul = u.copy()
cell_pos = ul[ul.yA == 1].groupby(['s0', 'ksic_mode']).size()
unl = ap[~ap.linked].groupby(['s0', 'ksic_mode']).size()
co = cell_pos.index.isin(unl.index)
add('A', 'P3', '양성 가중 KSIC×원점 칸 공존율(비연결 출원인 존재)', round(float((cell_pos[co].sum()) / cell_pos.sum()), 4))
# 연결률
lr = ap.groupby('s0').linked.mean()
add('A/C', 'P2e/P3', '연결률 원점별', json.dumps(lr.round(4).to_dict()), f'평균 {lr.mean():.4f}, 최고/최저 {lr.max()/lr.min():.3f}')
# P2c: 사후 표지와 양성률 관계(연결 단위 안)
for y in ['yA', 'yC']:
    hi = u.later_reg > 0.5
    r1, r0 = u.loc[hi, y].mean(), u.loc[~hi, y].mean()
    add('A' if y == 'yA' else 'C', 'P2c', f'{y} 양성률 later_reg>0.5 / ≤0.5', round(float(r1 / r0), 3), f'{r1:.4f} / {r0:.4f} (n {int(hi.sum())} / {int((~hi).sum())})')
    q = pd.qcut(u.later_reg.rank(method='first'), 4, labels=False)
    add('A' if y == 'yA' else 'C', 'P2c', f'{y} 양성률 later_reg 사분위', json.dumps(u.groupby(q)[y].mean().round(4).to_dict()))
# NPS 쪽 분포: 연결 단위 vs NPS 법인 전체(E0≥5), 원점 2019·2021
for s0 in ['201912', '202112']:
    f = firm_table(load_nps(s0)); f = f[f.E >= 5]
    uu = u[u.s0 == s0]
    def tvd(a, b):
        pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True); ix = pa.index.union(pb.index)
        return round(float(0.5 * (pa.reindex(ix, fill_value=0) - pb.reindex(ix, fill_value=0)).abs().sum()), 3)
    add('A/C', 'P3', f'업종 앞2자리 TVD 연결단위-NPS법인 {s0}', tvd(uu.ind2.fillna('NA'), f.ind2.fillna('NA')),
        f"연결단위 제조(10–33) 비중 {uu.ind2.fillna('00').astype(str).between('10','33').mean():.3f} / NPS {f.ind2.fillna('00').astype(str).between('10','33').mean():.3f} (국세청 코드 앞2자리, 근사)")
    add('A/C', 'P3', f'시도 TVD 연결단위-NPS법인 {s0}', tvd(uu.sido, f.sido), f"수도권(11·41·28) 연결 {uu.sido.isin(['11','41','28']).mean():.3f} / NPS {f.sido.isin(['11','41','28']).mean():.3f}")
    sz = lambda e: pd.cut(e, [4, 9, 29, 99, 299, 1e9], labels=['5-9', '10-29', '30-99', '100-299', '300+'])
    add('A/C', 'P3', f'규모 구간 분포 {s0}', json.dumps({'연결': sz(uu.E0).value_counts(normalize=True).round(3).to_dict(), 'NPS': sz(f.E).value_counts(normalize=True).round(3).to_dict()}, ensure_ascii=False))
    add('A/C', 'P3', f'E0 중앙값 연결/NPS {s0}', f'{uu.E0.median():.0f} / {f.E.median():.0f}')
# 개발 기저율
ra = u.groupby('s0').yA.mean()
add('A', 'P4d', '개발 양성률 원점별', json.dumps(ra.round(4).to_dict()), f'최고/최저 {ra.max()/ra.min():.3f}')
add('A', 'P4d', '개발 양성률 규모별(E0 구간)', json.dumps({str(k): float(v) for k, v in u.groupby(pd.cut(u.E0, [4, 9, 29, 99, 1e9]), observed=True).yA.mean().round(4).items()}))
pd.DataFrame(rows).to_csv(C + r'\s2_bias_A.csv', index=False, encoding='utf-8-sig')
