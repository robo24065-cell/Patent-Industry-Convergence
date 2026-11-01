# -*- coding: utf-8 -*-
"""위약 보완: verify_final_r5.py의 층(연결 x M0 10분위)은 M0 순위를 층으로 보존해 M0에 유리하게 기울어진다(위약 skill 평균 -0.135).
여기서는 구조적 층만 둔다: 연결 여부(미연결 기업은 구조적으로 y=0) 안에서 시험 레이블 1,000회 섞기. 모형 재학습 없음."""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE)
P = pd.read_parquet(os.path.join(OUT, "final_preds_r5.parquet"))
import duckdb
# 연결 여부: final_run과 같은 firm_link를 다시 쓰지 않고, 동결 패널 기준 대신 독립 재구성(verify) 결과를 쓴다
V = json.load(open(os.path.join(HERE, "verify_final_r5.json"), encoding="utf-8"))
import sys; sys.path.insert(0, OUT)
from n6_lib import load_ntis, firm_link
fl = firm_link(load_ntis(2008, 2021)).set_index("bz").linked
lk = P.bz.map(fl).fillna(False).astype(int).values
def auc_mw(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
y = P.y.values; pc = P.p_champ.values; pb = P.p_m0.values
obs = (auc_mw(y, pc) - auc_mw(y, pb)) / (1 - auc_mw(y, pb))
rng = np.random.default_rng(2027); sk, a1s, a0s = [], [], []
for _ in range(1000):
    ys = y.copy()
    for s in (0, 1):
        ii = np.where(lk == s)[0]; ys[ii] = rng.permutation(ys[ii])
    a1, a0 = auc_mw(ys, pc), auc_mw(ys, pb); sk.append((a1 - a0) / (1 - a0)); a1s.append(a1); a0s.append(a0)
sk = np.array(sk)
out = dict(strata="linked (2)", n_linked=int(lk.sum()), pos_in_unlinked=int(y[lk == 0].sum()), n_perm=1000, observed=float(obs),
           mean=float(sk.mean()), sd=float(sk.std()), q025=float(np.percentile(sk, 2.5)), q975=float(np.percentile(sk, 97.5)),
           p_ge_observed=float((sk >= obs).mean()), mean_auc_champ=float(np.mean(a1s)), mean_auc_M0=float(np.mean(a0s)))
json.dump(out, open(os.path.join(HERE, "placebo_linked_r5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
