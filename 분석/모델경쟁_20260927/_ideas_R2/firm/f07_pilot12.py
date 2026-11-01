# -*- coding: utf-8 -*-
"""아이디어 1(y_hg 고성장)·2(y_dist 신규 부실화) 파일럿. 개발 원점 v=2015..2021, 학습 t0 ≤ v−2.
표본 = t0 말까지 제공범위 등록특허 ≥1건 보유(has_pat=1) — 미래 특허 보유로 표본을 고르는 선택편의 차단."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evalkit import walk, summarize
C = os.path.join(os.path.dirname(__file__), "cache")
B = pd.read_parquet(os.path.join(C, "panel12.parquet"))
B = B[B.has_pat == 1]
DEV = list(range(2015, 2022)); H = 2
Y = sys.argv[1]
fin_f = ["log_rev", "log_assets", "g_rev1", "g_rev2", "opm", "roa", "lev", "eq_cap", "cur_ratio", "re_assets", "op_neg", "op_neg_m1", "firm_age"]
base_f = {"y_hg": ["log_rev", "g_rev2", "opm", "log_assets", "firm_age"],
          "y_dist": ["opm", "roa", "lev", "eq_cap", "log_assets", "op_neg_m1"]}[Y]
pat_f = ["n_all", "n_alive", "n_reg3", "n_regp3", "n_lap2", "n_tr3", "pat_age", "claims", "n_pay2", "n_lic3", "n_ksic", "hhi",
         "n_newksic3", "reg_mom", "lap_rate", "tr_rate", "pay_int", "firm_pat_age"]
ext_f = [c for c in B.columns if c.startswith("x_")]
res = []
ab, _ = walk(B, Y, base_f, DEV, H, "logit")
res.append(summarize("B0_base_logit", ab, ab))
exps = [("M1_logit_fin", fin_f, "logit"), ("M2_xgb_fin", fin_f, "xgb"),
        ("M3_logit_fin+pat", fin_f + pat_f, "logit"), ("M4_xgb_fin+pat", fin_f + pat_f, "xgb"),
        ("M5_logit_fin+pat+ext", fin_f + pat_f + ext_f, "logit"), ("M6_xgb_fin+pat+ext", fin_f + pat_f + ext_f, "xgb"),
        ("M7_logit_base+ext", base_f + ext_f, "logit"), ("M8_logit_base+pat", base_f + pat_f, "logit")]
for name, feats, mdl in exps:
    am, _ = walk(B, Y, feats, DEV, H, mdl)
    r = summarize(name, am, ab); res.append(r)
R = pd.DataFrame(res); R.insert(1, "target", Y)
R.to_csv(os.path.join(os.path.dirname(__file__), f"pilot12_{Y}.csv"), index=False)
pd.set_option("display.width", 250)
print("n by origin", B[B[Y].notna()].groupby("t0")[Y].agg(["size", "sum"]).T.to_dict())
print(R[["exp"] + [c for c in R.columns if c.startswith("auc_")] + ["mean_auc", "mean_base", "mean_skill"]].to_string())
