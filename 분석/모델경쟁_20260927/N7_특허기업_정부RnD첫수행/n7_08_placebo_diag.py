# -*- coding: utf-8 -*-
r"""위약 B 진단(사후, 모형 수정 없음): 학습 레이블 섞기 방식별 시험 AUC. 층 내 섞기는 원점별 기저율(8.5%→4.6%)을 남기므로
원점을 대리하는 피처(국민연금 이력 결측 등)로 '구성 효과'를 배울 수 있다 → 전역 섞기와 비교. 산출: final_placebo_diag_r5.json"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from n7lib import OUT, CACHE, predict
SPEC = json.load(open(os.path.join(OUT, "final_spec_r5.json"), encoding="utf-8")); FE, M0F = SPEC["features"], SPEC["baseline_M0"]
FP = pd.read_parquet(os.path.join(CACHE, "final_panel.parquet"))
tr = FP[FP.t0 <= 2020].reset_index(drop=True); te = FP[FP.t0 == 2022].reset_index(drop=True); y = te.y.values.astype(int)
def auc_rs(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
out = {}
rng = np.random.default_rng(7)
for nm, grp in [("global", np.zeros(len(tr))), ("within_t0", tr.t0.values)]:
    ys = tr.y.values.copy()
    for g, idx in pd.Series(np.arange(len(tr))).groupby(grp):
        ys[idx.values] = rng.permutation(ys[idx.values])
    pc = predict(tr[FE].astype(float), ys, te[FE].astype(float), model="ens", seeds=(0, 1, 2))
    p0 = predict(tr[M0F].astype(float), ys, te[M0F].astype(float), model="xgb", seeds=(0, 1, 2))
    a, b = auc_rs(y, pc), auc_rs(y, p0)
    out[nm] = dict(auc_champ=a, auc_M0=b, skill=(a - b) / (1 - b)); print(nm, out[nm], flush=True)
json.dump(out, open(os.path.join(OUT, "final_placebo_diag_r5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
