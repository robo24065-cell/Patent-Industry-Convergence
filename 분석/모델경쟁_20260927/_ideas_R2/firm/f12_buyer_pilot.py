# -*- coding: utf-8 -*-
"""아이디어 3 파일럿: 기술 구매기업 예측. 개발 원점 v=2015..2020, 학습 t0 ≤ v−2 (t0 ≥ 2012).
기준선(사전 고정) = 로지스틱[log 생존특허, log 최근3년 등록, 과거3년 양수 여부, 업력]."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evalkit import walk, summarize
C = os.path.join(os.path.dirname(__file__), "cache")
B = pd.read_parquet(os.path.join(C, "panel3.parquet"))
B["buy_past3_any"] = (B.buy_past3 > 0).astype(float)
B["buy_past_any"] = (B.buy_past_all > 0).astype(float)
DEV = list(range(2015, 2021)); H = 2
Y = sys.argv[1] if len(sys.argv) > 1 else "y_buy"
base_f = ["log_alive", "log_reg3", "buy_past3_any", "firm_age"]
pat_f = ["n_all", "n_alive", "n_reg3", "n_regp3", "n_lap2", "n_tr3", "pat_age", "claims", "n_pay2", "n_lic3", "n_ksic", "hhi",
         "n_newksic3", "reg_mom", "lap_rate", "tr_rate", "pay_int", "firm_pat_age", "firm_age"]
buy_f = ["buy_past3", "buy_past_all", "buy_small_past3"]
ext_f = [c for c in B.columns if c.startswith("x_")]
res = []
ab, _ = walk(B, Y, base_f, DEV, H, "logit")
res.append(summarize("B0_base_logit", ab, ab))
for name, feats, mdl in [
    ("M1_xgb_pat+buy", pat_f + buy_f, "xgb"),
    ("M2_xgb_pat+buy+ext", pat_f + buy_f + ext_f, "xgb"),
    ("M3_logit_pat+buy", pat_f + buy_f, "logit"),
    ("M4_logit_pat+buy+ext", pat_f + buy_f + ext_f, "logit"),
]:
    am, _ = walk(B, Y, feats, DEV, H, mdl)
    r = summarize(name, am, ab); res.append(r); print(r, flush=True)
R = pd.DataFrame(res); R.insert(1, "target", Y)
R.to_csv(os.path.join(os.path.dirname(__file__), f"pilot3_{Y}.csv"), index=False)
print(R[["exp", "mean_auc", "mean_base", "mean_skill"]])
