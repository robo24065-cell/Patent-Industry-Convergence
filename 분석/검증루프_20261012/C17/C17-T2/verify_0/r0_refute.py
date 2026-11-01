# -*- coding: utf-8 -*-
"""C17-T2 verify_0: 반박 검증. 독립 재구현으로 E2 재계산 + 층 세분(근거 문서 수, 첫 출원연도) 민감도 + Δ 독립 채점.
 읽기: ../cache/pair_judg.parquet, ../cache/eval_base.npz, ../cache/p_point.parquet. 쓰기: verify_0/ 안에만."""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
H = pathlib.Path(__file__).resolve().parent; CC = H.parent/'cache'
out = {}
PJ = pd.read_parquet(CC/'pair_judg.parquet'); P = PJ[PJ.grp == 'P'].reset_index(drop=True)
P['nd1'] = np.where(P.n_docs == 1, '1', '2+')
t1c = (P.pv == 'correct') & (P.ptier == 1)
out['n_docs_dist'] = {
  'code_correct': P[t1c].nd1.value_counts(normalize=True).round(3).to_dict(),
  'undet': P[P.pv == 'undet'].nd1.value_counts(normalize=True).round(3).to_dict(),
  'F_all': P[P.F].nd1.value_counts(normalize=True).round(3).to_dict(),
  'NF_all': P[~P.F].nd1.value_counts(normalize=True).round(3).to_dict()}
out['f_by_ndocs_codecorrect'] = {k: [int(g.F.sum()), int(len(g)), round(float(g.F.mean()), 4)] for k, g in P[t1c].groupby('nd1')}
out['ay_dist'] = {'code_correct': P[t1c].ay_min.value_counts(normalize=True).sort_index().round(3).to_dict(),
                  'undet': P[P.pv == 'undet'].ay_min.value_counts(normalize=True).sort_index().round(3).to_dict(),
                  'F': P[P.F].ay_min.value_counts(normalize=True).sort_index().round(3).to_dict()}
out['f_by_ay_codecorrect'] = {int(k): [int(g.F.sum()), int(len(g))] for k, g in P[t1c].groupby('ay_min')}


def e2(X, keys, minbase=30, fallback=('size',)):
    """독립 E2: 칸 = keys. f = 칸 안 1순위 정확 쌍의 F 비율(30 미만이면 fallback 칸), w = 규모별 판정 가능 NF 쌍 오매칭률(s2와 동일 정의)."""
    pv = X.pv.values; F = X.F.values; t1 = (pv == 'correct') & (X.ptier.values == 1)
    WF = 0.0; WNF = 0.0; cells = {}; p = np.where(pv == 'wrong', 1.0, 0.0)
    for kv, idx in X.groupby(keys).groups.items():
        c = np.zeros(len(X), bool); c[X.index.get_indexer(idx)] = True
        base = t1 & c
        if base.sum() < minbase:
            fb = np.ones(len(X), bool)
            for fk in fallback: fb &= (X[fk].values == X.loc[idx[0], fk])
            base = t1 & fb
        f = F[base].mean() if base.sum() else 0.0
        det = (~F) & ((pv == 'wrong') | t1); sz = X.loc[idx[0], 'size']; dc = det & (X['size'].values == sz)
        if dc.sum() < 30: dc = det
        w = (pv[dc] == 'wrong').mean()
        nF = (F & c).sum(); nNF = (~F & c).sum(); kwF = (F & c & (pv == 'wrong')).sum(); kwN = (~F & c & (pv == 'wrong')).sum()
        wnf = max(nNF * w, kwN); C = (nNF - wnf) / (1 - f); wf = max(nF - min(nF, f * C), kwF)
        WF += wf; WNF += wnf; cells[str(kv)] = dict(f=round(float(f), 4), nbase=int(base.sum()), nF=int(nF), WF=round(float(wf), 2))
        uF = c & F & (pv == 'undet'); uN = c & ~F & (pv == 'undet')
        p[uF] = np.clip((wf - kwF) / max(uF.sum(), 1), 0, 1); p[uN] = np.clip((wnf - kwN) / max(uN.sum(), 1), 0, 1)
    return dict(e=float((WF + WNF) / len(X)), WF=float(WF), WNF=float(WNF), cells=cells), p


r_main, p_main = e2(P, ['foreign', 'size'])
r_nd, p_nd = e2(P, ['foreign', 'size', 'nd1'], fallback=('size', 'nd1'))
r_nd2, p_nd2 = e2(P, ['size', 'nd1'], fallback=('nd1',))
r_ay, p_ay = e2(P, ['foreign', 'size', 'ay_min'], fallback=('size',))
r_ndonly, p_ndonly = e2(P, ['nd1'], fallback=('nd1',))
out['E2_reimpl_main'] = r_main
out['E2_cells_ndocs'] = r_nd; out['E2_size_x_ndocs'] = r_nd2; out['E2_cells_ay'] = r_ay; out['E2_ndocs_only'] = r_ndonly
pp = pd.read_parquet(CC/'p_point.parquet')
out['p_main_match_s2'] = float(np.abs(pp.p_main.values - p_main).max())
# ---- 독립 채점(전체 AUC 공식, 양성 제거) ----
Z = np.load(CC/'eval_base.npz', allow_pickle=True); s = Z['s']; y = Z['y'].astype(np.int8); base = Z['base']
key = pd.Series(np.arange(len(y)), index=pd.MultiIndex.from_arrays([Z['firm'], Z['k']]))
rows = key.reindex(pd.MultiIndex.from_arrays([P.firm.astype(str).values, P.k.values])).values.astype(int)
assert (pp.firm.astype(str).values == P.firm.astype(str).values).all()
R = [rankdata(s)] + [rankdata(b) for b in base]   # 고정 순위(제거 시 남은 양성의 음성 대비 순위 기여 불변 → 순위합 보정)
n1 = int(y.sum()); n0 = len(y) - n1
negrank = []  # 각 점수에서 양성 i 의 '음성보다 위' 수 = rank_i - (양성 중 i 이하 순위 수)
for r in R:
    rp = r[y == 1]; within = rankdata(rp); negrank.append(rp - within)
pos = np.where(y == 1)[0]; pm = -np.ones(len(y), int); pm[pos] = np.arange(n1); prow = pm[rows]
def skill(drop):
    k = np.ones(n1, bool); k[drop] = False
    a = [nr[k].sum() / (k.sum() * n0) for nr in negrank]; M0 = max(a[1:]); return (a[0] - M0) / (1 - M0)
OFF = skill(np.array([], int)); out['OFF_indep'] = OFF
def dist(p, n=1000, seed=31):
    rs = np.random.RandomState(seed); d = np.array([skill(prow[rs.rand(len(p)) < p]) - OFF for _ in range(n)])
    return dict(mean=float(d.mean()), p2_5=float(np.percentile(d, 2.5)), p97_5=float(np.percentile(d, 97.5)), max=float(d.max()), share_ge_0_0112=float((d >= 0.0112).mean()), exp_removed=float(p.sum()))
out['delta_main_point'] = dist(p_main)
out['delta_cells_ndocs'] = dist(p_nd)
out['delta_size_x_ndocs'] = dist(p_nd2)
out['delta_cells_ay'] = dist(p_ay)
out['delta_ndocs_only'] = dist(p_ndonly)
out['C16_flag220_indep'] = float(skill(prow[P.c16_flag.values]) - OFF)
out['F_vs_c16'] = dict(F=int(P.F.sum()), c16=int(P.c16_flag.sum()), both=int((P.F & P.c16_flag).sum()))
out['all_F_removed'] = float(skill(prow[P.F.values]) - OFF)
json.dump(out, open(H/'r0_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
