# -*- coding: utf-8 -*-
"""verify_0: 저장 예측에서 9원점 평균 AUC·보존율을 sklearn으로 독립 재계산, 평가 셀·레이블이 제공(drop_ros)과 같은지 확인."""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C15/C15-T3"
MC = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927"
R7 = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/results"; C5 = MC + "/_shared_report/fusion_advantage/c5_R10"
ORIG = list(range(2010, 2019))
def load(tag):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{W}/res/t3/{tag}__*_preds.parquet"))], ignore_index=True)
M = {"drop_ros": pd.read_parquet(R7 + "/pub_drop_ros_preds.parquet"), "kr": pd.read_parquet(R7 + "/pub_kr_rebuild_preds.parquet"),
     "stored_pkr": pd.read_parquet(C5 + "/t3_results/hyb_pkr_preds.parquet"),
     "reg": load("hyb_pkr_reg"), "pure": load("hyb_pkr_pure"), "repro": load("repro_hyb_pkr"), "rebuilt_r8": load("rebuilt_r8_hyb_pkr")}
out = {"by_origin": {}, "mean": {}, "cells_match_drop_ros": {}}
ref = M["drop_ros"]
for k, d in M.items():
    assert not d.duplicated(["t0", "cell"]).any(), k
    a = []; ok = True
    for v in ORIG:
        x = d[d.t0 == v].sort_values("cell"); r = ref[ref.t0 == v].sort_values("cell")
        ok &= (len(x) == len(r)) and bool((x.cell.values == r.cell.values).all()) and bool((x.yb.values == r.yb.values).all())
        a.append(roc_auc_score(x.yb.values, x.p.values))
    out["by_origin"][k] = [round(z, 5) for z in a]; out["mean"][k] = float(np.mean(a)); out["cells_match_drop_ros"][k] = bool(ok)
m = out["mean"]
ret = lambda h: (m[h] - m["kr"]) / (m["drop_ros"] - m["kr"])
m4 = {k: round(v, 4) for k, v in m.items()}
ret4 = lambda h: (m4[h] - m4["kr"]) / (m4["drop_ros"] - m4["kr"])
out["retention_raw"] = {h: ret(h) for h in ["stored_pkr", "reg", "pure", "rebuilt_r8"]}
out["retention_auc4"] = {h: ret4(h) for h in ["stored_pkr", "reg", "pure", "rebuilt_r8"]}
# 원점별 AUC 표(hyb_pkr_altA.csv)와 대조
T = pd.read_csv(W + "/hyb_pkr_altA.csv", encoding="utf-8-sig")
for arm, k in [("대안A_REG", "reg"), ("대안B_상속없음", "pure"), ("R8(재현)", "repro")]:
    t = T[T.arm == arm].sort_values("t0").auc.values
    out.setdefault("csv_maxdiff", {})[k] = float(np.abs(t - np.array(out["by_origin"][k])).max())
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(W + "/verify_0/v0_recalc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
