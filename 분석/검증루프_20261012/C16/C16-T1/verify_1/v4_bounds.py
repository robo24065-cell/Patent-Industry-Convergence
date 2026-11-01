# -*- coding: utf-8 -*-
"""verify_1 v4: 플래그 중 실제 오류 비율 q(19%·46%·100%)일 때 Δskill 범위: 실제 오류가 가장 낮게 매겨진 q (상한) / 가장 높게 매겨진 q (하한) / 무작위 q (기댓값)."""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'; FC = T1/'cache_final'/'o2019'; RD = T1/'runs'/'FROZEN_final'
HERE = B/'분석/검증루프_20261012/C16/C16-T1'; CC = HERE/'cache'; OUT = HERE/'verify_1'
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'prior_any']); fz = pd.read_parquet(RD/'o2019.parquet', columns=['px', 'pl'])
bo = pd.read_parquet(FC/'baseline_scores.parquet'); L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
kept = ~te.prior_any.values; yk = L.y_u.values[kept].astype(np.int8); neg = yk == 0
s = 0.9 * pd.Series(fz.px.values[kept]).rank(pct=True).values + 0.1 * pd.Series(fz.pl.values[kept]).rank(pct=True).values
def contrib(v):
    sn = np.sort(v[neg]); sp = v[~neg]; lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right'); return (lo + 0.5 * (hi - lo)) / neg.sum()
Cm = contrib(s); Cb = np.vstack([contrib(bo[c].values[kept]) for c in BK])
def sk(drop):
    keep = ~drop; a = Cm[keep].mean(); M0 = Cb[:, keep].mean(1).max(); return (a - M0) / (1 - M0)
pf = te.firm.astype(str).values[kept][~neg]; pk = te.k.values[kept][~neg]
pP = pd.read_parquet(CC/'alias_pairs_P.parquet'); fl = set(zip(pP.firm.astype(str)[pP.all_wrong], pP.k[pP.all_wrong]))
fi = np.where([(f, k) in fl for f, k in zip(pf, pk)])[0]
base = sk(np.zeros(len(pf), bool)); order = fi[np.argsort(Cm[fi])]
rng = np.random.RandomState(5); out = dict(base=base, n_flag=len(fi))
for q in (0.19, 0.46, 1.0):
    n = int(round(q * len(fi)))
    lo_ = np.zeros(len(pf), bool); lo_[order[:n]] = True; hi_ = np.zeros(len(pf), bool); hi_[order[len(order) - n:]] = True
    rnd = []
    for _ in range(500):
        d = np.zeros(len(pf), bool); d[rng.choice(fi, n, replace=False)] = True; rnd.append(sk(d) - base)
    out[f'q{q}'] = dict(n=n, max_if_errors_lowest=sk(lo_) - base, min_if_errors_highest=sk(hi_) - base, random_mean=float(np.mean(rnd)), random_p2_5=float(np.percentile(rnd, 2.5)), random_p97_5=float(np.percentile(rnd, 97.5)))
out['flag_contrib_quantiles'] = [float(x) for x in np.percentile(Cm[fi], [10, 25, 50, 75, 90])]
json.dump(out, open(OUT/'v4_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
