# -*- coding: utf-8 -*-
"""v1 check: (1) 공식 dev 패널 vs 작업자 off_2012-2020 패널 동일성 (2) 시험 패널 단위·레이블 vs final_preds_r4
(3) 시드 (0,1,2) 평균 예측이 공식 예측과 같은지, 공식 skill·CI 재현 (자체 부트스트랩 구현, 공식과 같은 RNG 규칙)."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
TR = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N5_표준특허선언레이더"
SPEC = json.load(open(os.path.join(TR, "final_spec_r4.json"), encoding="utf-8"))
FE, F0 = SPEC["champion_features"], SPEC["baseline_M0"]
R = {}
od = pd.read_parquet(os.path.join(TR, "cache", "panel_dev_26429.parquet"))
wk = pd.concat([pd.read_parquet(os.path.join(W, "cache", "panels", f"off_{t}.parquet")) for t in range(2012, 2021)], ignore_index=True)
m = od[["appl_no", "t0", "y"] + FE].merge(wk[["appl_no", "t0", "y"] + FE], on=["appl_no", "t0"], how="outer", indicator=True, suffixes=("_o", "_w"))
b = m[m._merge == "both"]
R["train_rows_official"] = len(od); R["train_rows_worker"] = len(wk)
R["train_only_official"] = int((m._merge == "left_only").sum()); R["train_only_worker"] = int((m._merge == "right_only").sum())
R["train_y_mismatch"] = int((b.y_o != b.y_w).sum())
R["train_feat_maxabsdiff"] = float(max(np.nanmax(np.abs(b[f + "_o"].astype(float).values - b[f + "_w"].astype(float).values)) if b[f + "_o"].notna().any() else 0 for f in FE))
R["train_feat_nan_mismatch"] = int(sum((b[f + "_o"].isna() != b[f + "_w"].isna()).sum() for f in FE))
FP = pd.read_parquet(os.path.join(TR, "final_preds_r4.parquet"))
p = [pd.read_parquet(os.path.join(V, "v1_preds", "off", f"seed_{s}.parquet")) for s in (0, 1, 2)]
d = p[0][["appl_no", "app", "y"]].copy(); d["pc"] = np.mean([x.p_champ for x in p], axis=0); d["p0"] = np.mean([x.p_M0 for x in p], axis=0)
mm = FP.merge(d, on="appl_no", how="outer", indicator=True, suffixes=("", "_v"))
R["test_n_official"] = len(FP); R["test_n_v1"] = len(d); R["test_only_one_side"] = int((mm._merge != "both").sum())
R["test_y_mismatch"] = int((mm.y != mm.y_v).sum()); R["test_app_mismatch"] = int((mm.app != mm.app_v).sum())
R["pred_maxdiff_champ"] = float(np.abs(mm.p_champ - mm.pc).max()); R["pred_maxdiff_M0"] = float(np.abs(mm.p_M0 - mm.p0).max())


def boot(y, pm, pb, g, n=1000, seed=2022):
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(np.asarray(g), return_inverse=True)
    order = np.argsort(inv, kind="stable"); cuts = np.cumsum(np.bincount(inv))[:-1]
    idx = np.split(order, cuts); G = len(idx); out = []
    for _ in range(n):
        pick = rng.integers(0, G, G); i = np.concatenate([idx[j] for j in pick])
        if y[i].min() == y[i].max():
            continue
        a1 = roc_auc_score(y[i], pm[i]); a0 = roc_auc_score(y[i], pb[i]); out.append((a1 - a0) / (1 - a0))
    return np.percentile(out, [2.5, 97.5]).tolist()


y = d.y.values; a1 = roc_auc_score(y, d.pc); a0 = roc_auc_score(y, d.p0)
R["skill_012"] = (a1 - a0) / (1 - a0); R["ci_012"] = boot(y, d.pc.values, d.p0.values, d.app.values)
R["official_skill"] = 0.16496503719255987; R["official_ci"] = [0.0067307970348523885, 0.35964244917890836]
json.dump(R, open(os.path.join(V, "v1_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
