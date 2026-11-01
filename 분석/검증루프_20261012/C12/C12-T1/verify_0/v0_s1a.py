# -*- coding: utf-8 -*-
"""verify_0: S1a(동결 예측 + 완전 키 레이블) 독립 재계산 — v0_refl.parquet(독립 파서), 순위합 AUC(자체 구현), 자체 군집 부트스트랩.
또 S1 재적합 패널(s5 s1_2022)의 시험 단위·레이블이 독립 완전 키 규칙과 같은지 대조."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding="utf-8")
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
TR = os.path.join(BASE, r"분석\모델경쟁_20260927\N5_표준특허선언레이더")
FP = pd.read_parquet(os.path.join(TR, "final_preds_r4.parquet"))
R = pd.read_parquet(os.path.join(V, "v0_refl.parquet"))
rf = dict(zip(R.appl_no, R.refl_full))


def auc_rank(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


d = FP.copy(); d["rf"] = d.appl_no.map(rf)
d = d[~(d.rf <= 20221231)].copy(); d["y2"] = ((d.rf > 20221231) & (d.rf <= 20241231)).astype(int)
a1, a0 = auc_rank(d.y2, d.p_champ), auc_rank(d.y2, d.p_M0)
sk = (a1 - a0) / (1 - a0)
rng = np.random.default_rng(12345)
groups = d.groupby("app").indices; keys = list(groups)
bs = []
for _ in range(1000):
    pick = rng.choice(len(keys), len(keys), replace=True)
    ii = np.concatenate([groups[keys[j]] for j in pick])
    yy = d.y2.values[ii]
    if yy.min() == yy.max():
        continue
    b1, b0 = auc_rank(yy, d.p_champ.values[ii]), auc_rank(yy, d.p_M0.values[ii])
    bs.append((b1 - b0) / (1 - b0))
out = dict(n=int(len(d)), pos=int(d.y2.sum()), dropped_already_declared=int(len(FP) - len(d)), auc_champ=a1, auc_M0=a0, skill=sk,
           ci_own_bootstrap=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
P = pd.read_parquet(os.path.join(W, "cache", "panels", "s1_2022.parquet"), columns=["appl_no", "y"])
mg = P.merge(d[["appl_no", "y2"]], on="appl_no", how="outer", indicator=True)
out["s1_panel_vs_v0"] = dict(only_panel=int((mg._merge == "left_only").sum()), only_v0=int((mg._merge == "right_only").sum()),
                             label_mismatch=int((mg[mg._merge == "both"].y != mg[mg._merge == "both"].y2).sum()))
json.dump(out, open(os.path.join(V, "v0_s1a.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
