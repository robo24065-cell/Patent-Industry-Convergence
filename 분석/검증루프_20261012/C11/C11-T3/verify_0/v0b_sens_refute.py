# -*- coding: utf-8 -*-
"""verify_0b: C11-T3 민감도 발견(V1/V2/V3 Δskill '영향 없음') 반박 시도 — 독립 재채점.
s5c 코드 미사용. sklearn roc_auc_score, 다른 시드(777)·다른 부트스트랩 구현(mg 단위 가중치 방식), 피처 실제 교체 여부, 공식 피처 일치 확인.
읽기만: cache/preds_*, cache/scen_ros_*, 트랙 R3 extB_cell_pub*.parquet, preds/FINAL_R3_t0_2021.parquet. 쓰기: verify_0/v0b_out.json"""
import json, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
W = B + "/분석/검증루프_20261012/C11/C11-T3"; C = W + "/cache"
R3 = B + "/분석/모델경쟁_20260927/T3_기술셀부상예측/R3"
out = {}
def load(s, m):
    if m == 'final': return pd.read_parquet(f"{C}/preds_{s}_final.parquet")
    return pd.concat([pd.read_parquet(f"{C}/preds_{s}_dev_{x}.parquet") for x in ('2010-2011-2012-2013-2014', '2015-2016-2017-2018')], ignore_index=True)
P = {(s, m): load(s, m).sort_values(['t0', 'cell']).reset_index(drop=True) for s in ('S0', 'V1', 'V2', 'V3') for m in ('dev', 'final')}
def sk(y, p, b):
    a = roc_auc_score(y, p); ab = roc_auc_score(y, b); return (a - ab) / (1 - ab)
# 1) 공식 피처 = S0 피처, V 피처 = scen_ros 값, 바뀐 비율
for m, suf, sf in (('dev', '', 'dev'), ('final', '_final', 'fin')):
    X = pd.read_parquet(f"{R3}/extB_cell_pub{suf}.parquet", columns=['cell', 't0', 'ros_sci', 'ros_ppp'])
    S = pd.read_parquet(f"{C}/scen_ros_{sf}.parquet")
    a = P[('S0', m)].merge(X, on=['cell', 't0'], suffixes=('', '_off'))
    out[f'{m}_S0_vs_extB'] = dict(n=len(a), n_preds=len(P[('S0', m)]), dsci=float((a.ros_sci - a.ros_sci_off).abs().max()), dppp=float((a.ros_ppp - a.ros_ppp_off).abs().max()))
    for V in ('V1', 'V2', 'V3'):
        v = P[(V, m)]; s0 = P[('S0', m)]
        assert (v.cell.values == s0.cell.values).all() and (v.yb.values == s0.yb.values).all()
        mm = v.merge(S[['cell', 't0', f'ros_sci_{V}', f'ros_ppp_{V}']], on=['cell', 't0'])
        out[f'{m}_{V}_feat'] = dict(match_scen_sci=float((mm.ros_sci - mm[f'ros_sci_{V}']).abs().max()), match_scen_ppp=float((mm.ros_ppp - mm[f'ros_ppp_{V}']).abs().max()),
                                    share_sci_changed=float((np.abs(v.ros_sci.values - s0.ros_sci.values) > 1e-9).mean()),
                                    share_ppp_changed=float((np.abs(v.ros_ppp.values - s0.ros_ppp.values) > 1e-9).mean()),
                                    spearman_sci=float(pd.Series(v.ros_sci.values).rank().corr(pd.Series(s0.ros_sci.values).rank())),
                                    logit_identical=float(np.abs(v.p_logit.values - s0.p_logit.values).max()),
                                    xgbc_maxdiff=float(np.abs(v.p_xgbc.values - s0.p_xgbc.values).max()), nb_maxdiff=float(np.abs(v.p_nb.values - s0.p_nb.values).max()),
                                    base_maxdiff=float(np.abs(v.p_base.values - s0.p_base.values).max()))
# 2) 점추정 재채점
rows = []
for m in ('dev', 'final'):
    for s in ('S0', 'V1', 'V2', 'V3'):
        d = P[(s, m)]
        for t0, g in d.groupby('t0'):
            rows.append(dict(scen=s, mode=m, t0=int(t0), skill=sk(g.yb.values, g.p.values, g.p_base.values), auc=roc_auc_score(g.yb.values, g.p.values)))
T = pd.DataFrame(rows)
piv = T.pivot_table(index=['mode', 't0'], columns='scen', values='skill')
summ = {}
for m in ('dev', 'final'):
    q = piv.loc[m]
    summ[m] = {s: round(float(q[s].mean()), 5) for s in q.columns}
    for V in ('V1', 'V2', 'V3'):
        summ[m][f'd_{V}'] = round(float((q[V] - q['S0']).mean()), 5); summ[m][f'wins_{V}'] = int((q[V] > q['S0']).sum())
        summ[m][f'maxabs_{V}'] = round(float((q[V] - q['S0']).abs().max()), 5)
out['skill'] = summ
out['by_origin'] = {f'{m}_{t}': {k: round(float(v), 5) for k, v in r.items()} for (m, t), r in piv.iterrows()}
# 3) 다른 구현의 군집 부트스트랩(시드 777): mg 군집 재표본 = 다항 가중치 → 가중 AUC
rng = np.random.default_rng(777)
def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)
def boot(m, V, Bn):
    a, b = P[('S0', m)], P[(V, m)]; res = []
    G = {}
    for t0, ga in a.groupby('t0'):
        gb = b.loc[ga.index]; codes, uniq = pd.factorize(ga.mg.values)
        G[t0] = (codes, len(uniq), ga.yb.values, ga.p.values, gb.p.values, ga.p_base.values)
    for _ in range(Bn):
        ds = []
        for t0, (codes, k, y, p0, p1, pb) in G.items():
            cnt = rng.multinomial(k, np.ones(k) / k); w = cnt[codes].astype(float)
            if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0: continue
            ab = wauc(y, pb, w); ds.append(((wauc(y, p1, w) - ab) - (wauc(y, p0, w) - ab)) / (1 - ab))
        res.append(np.mean(ds))
    return [round(float(np.percentile(res, 2.5)), 5), round(float(np.percentile(res, 97.5)), 5)]
out['ci'] = {f'{m}_{V}': boot(m, V, 1000 if m == 'final' else 200) for m in ('final', 'dev') for V in ('V1', 'V2', 'V3')}
# 4) 상위 10% 겹침
ov = {}
for V in ('V1', 'V2', 'V3'):
    a = P[('S0', 'final')]; b = P[(V, 'final')]; k = int(round(0.1 * len(a)))
    ta = set(a.nlargest(k, 'p').cell); tb = set(b.nlargest(k, 'p').cell)
    ov[V] = [len(ta & tb), k, int(a[a.cell.isin(ta)].yb.sum()), int(b[b.cell.isin(tb)].yb.sum())]
out['top10'] = ov
# 5) 공식 예측과 S0
off = pd.read_parquet(f"{R3}/preds/FINAL_R3_t0_2021.parquet")
mm = P[('S0', 'final')].merge(off[['cell', 'p', 'p_base', 'yb']], on='cell', suffixes=('', '_o'))
out['S0_vs_official'] = dict(n=len(mm), dp=float((mm.p - mm.p_o).abs().max()), skill_off=sk(off.yb.values, off.p.values, off.p_base.values))
out['threshold'] = 0.25 * (0.473 - 0.366)
json.dump(out, open(W + "/verify_0/v0b_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
