# -*- coding: utf-8 -*-
"""s5c: 민감도 결과 정리. S0 재현(공식 예측과 p 최대차), 시나리오 Δskill·ΔAUC, IPC 메인그룹 군집 짝 부트스트랩(1,000회, 시드 20261013).
출력: sens_results.csv, res/s5c_compare.json"""
import json, os
from common import *
import numpy as np, pandas as pd
from scipy.stats import rankdata
C = W + "/cache"
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def skill(y, p, b):
    a, ab = auc(y, p), auc(y, b); return (a - ab) / (1 - ab), a, ab
def load(s, mode):
    if mode == 'final': return pd.read_parquet(f"{C}/preds_{s}_final.parquet")
    return pd.concat([pd.read_parquet(f"{C}/preds_{s}_dev_2010-2011-2012-2013-2014.parquet"), pd.read_parquet(f"{C}/preds_{s}_dev_2015-2016-2017-2018.parquet")], ignore_index=True)
R = {}
P = {(s, m): load(s, m) for s in ('S0', 'V1', 'V2', 'V3') for m in ('dev', 'final')}
# S0 재현: 공식 최종 예측
off = pd.read_parquet(f"{T3R3}/preds/FINAL_R3_t0_2021.parquet")
pc = 'p' if 'p' in off else [c for c in off.columns if c.startswith('p')][0]
m = P[('S0', 'final')].merge(off[['cell', pc, 'yb']].rename(columns={pc: 'p_off', 'yb': 'yb_off'}), on='cell')
R['S0_final_vs_official'] = dict(n=len(m), max_abs_dp=float(np.abs(m.p - m.p_off).max()), label_mismatch=int((m.yb != m.yb_off).sum()))
log('S0 vs official', R['S0_final_vs_official'])
rng = np.random.default_rng(20261013)
def boot(df_s0, df_v, B=1000):
    # 같은 셀 정렬
    a = df_s0.sort_values(['t0', 'cell']).reset_index(drop=True); b = df_v.sort_values(['t0', 'cell']).reset_index(drop=True)
    assert (a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all()
    out = {}
    pt = []
    for t0, ga in a.groupby('t0'):
        gb = b[b.t0 == t0]; y = ga.yb.values
        s0, a0, ab = skill(y, ga.p.values, ga.p_base.values); s1, a1, _ = skill(y, gb.p.values, gb.p_base.values)
        pt.append(dict(t0=int(t0), skill_S0=s0, skill_V=s1, d_skill=s1 - s0, auc_S0=a0, auc_V=a1, d_auc=a1 - a0, base_auc=ab))
    pt = pd.DataFrame(pt)
    # 군집 부트스트랩: 원점별로 mg 재표본, 원점 평균 Δ
    groups = {t0: (ga.mg.values, ga.yb.values, ga.p.values, b[b.t0 == t0].p.values, ga.p_base.values) for t0, ga in a.groupby('t0')}
    ds, s0s, s1s = [], [], []
    for _ in range(B):
        dd, x0, x1 = [], [], []
        for t0, (mg, y, p0, p1, pb) in groups.items():
            u = np.unique(mg); pick = rng.choice(u, len(u), replace=True)
            idx = np.concatenate([np.flatnonzero(mg == g) for g in pick]) if len(u) < 50 else None
            if idx is None:
                order = np.argsort(mg, kind='stable'); mgs = mg[order]; starts = np.searchsorted(mgs, u); ends = np.searchsorted(mgs, u, side='right')
                pos = {g: i for i, g in enumerate(u)}; ii = [pos[g] for g in pick]
                idx = np.concatenate([order[starts[i]:ends[i]] for i in ii])
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == len(yy): continue
            k0 = skill(yy, p0[idx], pb[idx])[0]; k1 = skill(yy, p1[idx], pb[idx])[0]
            dd.append(k1 - k0); x0.append(k0); x1.append(k1)
        ds.append(np.mean(dd)); s0s.append(np.mean(x0)); s1s.append(np.mean(x1))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return pt, dict(d_skill_ci=q(ds), skill_S0_ci=q(s0s), skill_V_ci=q(s1s))
rows = []
for mode in ('dev', 'final'):
    for V in ('V1', 'V2', 'V3'):
        pt, ci = boot(P[('S0', mode)], P[(V, mode)], B=1000 if mode == 'final' else 300)
        rows.append(dict(scenario=V, mode=mode, n_origins=len(pt), skill_S0=pt.skill_S0.mean(), skill_V=pt.skill_V.mean(), d_skill=pt.d_skill.mean(),
                         d_skill_ci_lo=ci['d_skill_ci'][0], d_skill_ci_hi=ci['d_skill_ci'][1], skill_V_ci_lo=ci['skill_V_ci'][0], skill_V_ci_hi=ci['skill_V_ci'][1],
                         auc_S0=pt.auc_S0.mean(), auc_V=pt.auc_V.mean(), d_auc=pt.d_auc.mean(), wins=int((pt.d_skill > 0).sum()),
                         max_abs_d_origin=float(pt.d_skill.abs().max()), rank_corr_p=float(np.corrcoef(rankdata(P[('S0', mode)].sort_values(['t0','cell']).p), rankdata(P[(V, mode)].sort_values(['t0','cell']).p))[0, 1])))
        R[f'{V}_{mode}_by_origin'] = pt.round(5).to_dict('records')
        log(V, mode, rows[-1]['d_skill'], ci['d_skill_ci'])
S = pd.DataFrame(rows)
S['official_ci_width'] = 0.473 - 0.366; S['threshold_25pct'] = 0.25 * (0.473 - 0.366)
S['verdict_delta'] = np.where(S.d_skill.abs() < S.threshold_25pct, '|Δ|<25% CI폭', '|Δ|≥25% CI폭')
S.round(5).to_csv(W + "/sens_results.csv", index=False, encoding="utf-8-sig")
print(S.round(4).to_string())
# 최종 상위 10% 겹침
for V in ('V1', 'V2', 'V3'):
    a = P[('S0', 'final')].sort_values('cell'); b = P[(V, 'final')].sort_values('cell')
    k = int(round(0.1 * len(a))); ta = set(a.nlargest(k, 'p').cell); tb = set(b.nlargest(k, 'p').cell)
    R[f'{V}_final_top10_overlap'] = [len(ta & tb), k, int(a[a.cell.isin(ta)].yb.sum()), int(b[b.cell.isin(tb)].yb.sum())]
log('top10', {k: v for k, v in R.items() if 'top10' in k})
json.dump(R, open(W + "/res/s5c_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
