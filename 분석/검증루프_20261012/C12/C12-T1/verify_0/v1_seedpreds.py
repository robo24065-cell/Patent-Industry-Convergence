# -*- coding: utf-8 -*-
"""verify_0 반박검증 v1 (2차 검증자, 독립 코드): 단일 시드 XGB 예측을 시드별로 저장.
학습 = 공식 트랙 cache\\panel_dev_26429.parquet(t0 2012-2020, 공식 n5_eval 산출물, 작업자 cache 아님)
시험 = 작업자 cache\\panels\\off_2022.parquet(단위·레이블은 check 단계에서 공식 final_preds_r4와 대조)
사용: python v1_seedpreds.py <seed_from> <seed_to_exclusive>  -> verify_0\\v1_preds\\seed_<s>.parquet"""
import os, sys, json, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
TR = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N5_표준특허선언레이더"
SPEC = json.load(open(os.path.join(TR, "final_spec_r4.json"), encoding="utf-8"))
FE, F0, XP = SPEC["champion_features"], SPEC["baseline_M0"], SPEC["xgb_params"]
scen = sys.argv[3] if len(sys.argv) > 3 else "off"
if scen == "off":
    tr = pd.read_parquet(os.path.join(TR, "cache", "panel_dev_26429.parquet"))
else:
    tr = pd.concat([pd.read_parquet(os.path.join(W, "cache", "panels", f"{scen}_{t}.parquet")) for t in range(2012, 2021)], ignore_index=True)
te = pd.read_parquet(os.path.join(W, "cache", "panels", f"{scen}_2022.parquet"))
od = os.path.join(V, "v1_preds", scen); os.makedirs(od, exist_ok=True)
t = time.time()
for s in range(int(sys.argv[1]), int(sys.argv[2])):
    fn = os.path.join(od, f"seed_{s}.parquet")
    if os.path.exists(fn):
        continue
    out = {}
    for nm, F in [("champ", FE), ("M0", F0)]:
        m = XGBClassifier(random_state=s, **XP)
        m.fit(tr[F].astype(float), tr.y)
        out[nm] = m.predict_proba(te[F].astype(float))[:, 1]
    pd.DataFrame({"appl_no": te.appl_no.values, "app": te.app.values, "y": te.y.values, "p_champ": out["champ"], "p_M0": out["M0"]}).to_parquet(fn, index=False)
    print(scen, s, round(time.time() - t), flush=True)
