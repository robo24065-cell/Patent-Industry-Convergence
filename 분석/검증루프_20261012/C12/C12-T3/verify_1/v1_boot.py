# -*- coding: utf-8 -*-
"""verify_1 V3(추가, 사후 — 판정 아님): 최종시험 no_pi − full 짝 Δskill의 셀 부트스트랩(1,000회, 셀 단위·메인그룹 군집 단위).
예측은 작업 cache\\b3_p_{S0,S3}_s{k}.parquet(V1에서 skill이 공식 경로 재계산과 5e-7 안 일치 확인)."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding='utf-8')
OUT = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(os.path.dirname(OUT), 'cache')
P = {(s, k): pd.read_parquet(os.path.join(C, f'b3_p_{s}_s{k}.parquet')) for s in ['S0', 'S3'] for k in range(5)}
base = P[('S0', 0)][['cell', 'yb', 'p_base']]
for v in P.values(): assert (v.cell.values == base.cell.values).all()
y = base.yb.values; pb = base.p_base.values
mg = base.cell.str.split('|').str[1].values; ug, gi = np.unique(mg, return_inverse=True)
S = {key: v.p.values for key, v in P.items()}
def auc(yy, pp):
    r = rankdata(pp); n1 = yy.sum(); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))
def dskill(idx):
    yy = y[idx]; a0 = auc(yy, pb[idx]); d = []
    for k in range(5):
        a = auc(yy, S[('S0', k)][idx]); b = auc(yy, S[('S3', k)][idx]); d.append(((b - a0) - (a - a0)) / (1 - a0))
    return float(np.mean(d))
obs = dskill(np.arange(len(y)))
rng = np.random.default_rng(20261017); res = {}
for mode in ['cell', 'cluster_mg']:
    bs = []
    for _ in range(1000):
        if mode == 'cell': idx = rng.integers(0, len(y), len(y))
        else:
            g = rng.integers(0, len(ug), len(ug)); cnt = np.bincount(g, minlength=len(ug)); idx = np.repeat(np.arange(len(y)), cnt[gi])
        bs.append(dskill(idx))
    bs = np.array(bs); res[mode] = dict(obs=obs, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), p_le0=float((bs <= 0).mean()))
    print(mode, res[mode], flush=True)
json.dump(res, open(os.path.join(OUT, 'v1_boot.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
