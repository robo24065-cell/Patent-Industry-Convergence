# -*- coding: utf-8 -*-
"""IPO 레이더 파일럿 3단계: 사전 선언(PLAN_IPO.md) 실험 실행. 개발 원점만. 예측은 cache/preds_<exp>.parquet에 캐시(있으면 건너뜀).
사용: python p3_eval.py [exp ...]   (인자 없으면 남은 실험 전부; 9분 넘으면 다시 호출)"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
sys.dont_write_bytecode = True
R7 = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(R7, "cache")
T = time.time()
M0 = ["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type", "g_emp2", "log_wkpl", "g_wage1", "d_loss_rate", "d_hire_rate", "nps_withdrawn", "nps_exit1y"]
M1 = ["log_app", "yrs_since_last", "app_trend", "known_1y", "log_live", "live_ratio", "lapse_1y", "lapse_2y", "pay_1y", "pay_trend", "late_share_2y", "overdue_share", "live_age_mean", "trf_out_2y", "merge_2y"]
M2 = ["ntis_any5", "ntis_n3", "ntis_funds", "ntis_active", "ntis_trend", "ntis_yrs_since", "gov_pat_n", "gov_pat_share", "gov_pat_recent"]
SB = M0 + M1 + M2
PK = ["n_ksic", "ksic_hhi", "ksic_heat", "ksic_g3", "ksic_pos"]
PG = ["fam_n_3y", "fam_share", "fam_prio_2y"]
PT = ["ipc_div", "cell_hot", "ipc_new_share"]
PKA = ["alt_n", "alt_hhi", "alt_heat", "alt_g3", "alt_pos"]
FULL = SB + PK + PG + PT
KRONLY = SB + [c if c != "ksic_g3" else "ksic_g3kr" for c in PK] + [c if c != "cell_hot" else "cell_hot_kr" for c in PT]
EXPS = {"E0": M0, "E1": M0 + M1, "E2": SB,
        "FULL": FULL, "X1": SB + PG + PT, "X2": SB + PK + PT, "X3": SB + PK + PG,
        "X5": SB + PKA + PG + PT, "X6": KRONLY, "X7": SB + PKA,
        "C1": SB + PK, "C1alt": SB + PKA,
        # 사슬 완성(PLAN_IPO 순서 그대로: C1 롤백 시 SB 위에 PG, 그다음 챔피언 위에 PT)
        "C2": SB + PG, "C3a": SB + PT, "C3b": SB + PG + PT}
LOGIT = {"L_SB": SB, "L_FULL": FULL, "L_X5": SB + PKA + PG + PT}
VS = [2017, 2018, 2019, 2020, 2021]
P = pd.read_parquet(os.path.join(C, "ipo_panel.parquet"))
D = P[P.t0 <= 2021].reset_index(drop=True)
assert D.y.notna().all()
PRM = dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, n_jobs=2, tree_method="hist", verbosity=0)

def run(name, cols, logit=False):
    fn = os.path.join(C, f"preds_{name}.parquet")
    if os.path.exists(fn): return pd.read_parquet(fn)
    out = []
    for v in VS:
        tr = D[(D.t0 >= 2015) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        if logit:
            med = tr[cols].median().fillna(0)
            sc = StandardScaler().fit(tr[cols].fillna(med))
            m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr[cols].fillna(med)), tr.y)
            p = m.predict_proba(sc.transform(te[cols].fillna(med)))[:, 1]
        else:
            p = np.zeros(len(te))
            for s in (0, 1, 2):
                m = xgb.XGBClassifier(random_state=s, **PRM).fit(tr[cols], tr.y)
                p += m.predict_proba(te[cols])[:, 1] / 3
        out.append(pd.DataFrame({"idx": te.index, "t0": v, "p": p}))
    o = pd.concat(out); o.to_parquet(fn, index=False)
    print(name, "done", f"[{round(time.time()-T)}s]", flush=True)
    return o

todo = sys.argv[1:] or list(EXPS) + list(LOGIT)
for e in todo:
    if time.time() - T > 480: print("시간 한도 — 다시 호출"); break
    if e in EXPS: run(e, EXPS[e])
    elif e in LOGIT: run(e, LOGIT[e], logit=True)
