# -*- coding: utf-8 -*-
"""반박 검증 r1: 예측 파일에서 AUC를 sklearn으로 독립 재계산, 평가 셀 동일성, 보존율(대안 A/B/R8) 재계산. 읽기만, 출력은 verify_1/refute/."""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(os.path.join(HERE, "..", ".."))
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
R7 = os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "results")
C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10")
C10 = os.path.abspath(os.path.join(W, "..", "..", "C10", "C10-T3"))
ORIG = list(range(2010, 2019))
def load(tag):
    fs = sorted(glob.glob(os.path.join(W, "res", "t3", f"{tag}__*_preds.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
M = {"drop_ros": pd.read_parquet(os.path.join(R7, "pub_drop_ros_preds.parquet")),
     "kr": pd.read_parquet(os.path.join(R7, "pub_kr_rebuild_preds.parquet")),
     "pkr_stored": pd.read_parquet(os.path.join(C5, "t3_results", "hyb_pkr_preds.parquet")),
     "txt_reg": pd.read_parquet(os.path.join(C10, "res", "t3", "hyb_txt_reg_preds.parquet")),
     "free_reg": pd.read_parquet(os.path.join(C10, "res", "t3", "hyb_free_reg_preds.parquet")),
     "pkr_A": load("hyb_pkr_reg"), "pkr_B": load("hyb_pkr_pure"), "pkr_r8rebuilt": load("rebuilt_r8_hyb_pkr")}
ref = M["drop_ros"].set_index(["t0", "cell"]).sort_index()
out = {"cells": {}, "auc_by_origin": {}, "auc_mean": {}}
for k, d in M.items():
    x = d.set_index(["t0", "cell"]).sort_index()
    out["cells"][k] = dict(n=len(x), dup=int(x.index.duplicated().sum()), same_index=bool(x.index.equals(ref.index)),
                           same_yb=bool(x.index.equals(ref.index) and (x.yb.values == ref.yb.values).all()), nan_p=int(x.p.isna().sum()))
    a = [roc_auc_score(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG]
    out["auc_by_origin"][k] = [round(z, 5) for z in a]; out["auc_mean"][k] = float(np.mean(a))
am = out["auc_mean"]
ret = lambda h, r: (r[h] - r["kr"]) / (r["drop_ros"] - r["kr"])
r4 = {k: round(v, 4) for k, v in am.items()}
out["retention_raw"] = {k: ret(k, am) for k in ["txt_reg", "free_reg", "pkr_stored", "pkr_A", "pkr_B", "pkr_r8rebuilt"]}
out["retention_auc4"] = {k: ret(k, r4) for k in ["txt_reg", "free_reg", "pkr_stored", "pkr_A", "pkr_B", "pkr_r8rebuilt"]}
# 잡음 민감도: 보존율 정수가 61%로 돌아가려면 필요한 AUC 하락, 66%로 내려가려면 필요한 하락
den = am["drop_ros"] - am["kr"]
out["auc_drop_needed_to_round_61"] = am["pkr_A"] - (am["kr"] + 0.615 * den)
out["auc_drop_needed_to_round_66"] = am["pkr_A"] - (am["kr"] + 0.665 * den)
out["ret_per_0.0002auc_pp"] = 0.0002 / den * 100
# 원점 제외(leave-one-origin-out) 보존율 범위
loo = []
for i in range(9):
    f = lambda k: np.mean([z for j, z in enumerate(out["auc_by_origin"][k]) if j != i])
    loo.append((f("pkr_A") - f("kr")) / (f("drop_ros") - f("kr")))
out["retention_A_leave_one_origin_out"] = [round(z, 4) for z in loo]
json.dump(out, open(os.path.join(HERE, "r1_auc_recompute.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ["cells", "auc_mean", "retention_raw", "retention_auc4", "auc_drop_needed_to_round_61", "auc_drop_needed_to_round_66", "ret_per_0.0002auc_pp", "retention_A_leave_one_origin_out"]}, ensure_ascii=False, indent=1))
