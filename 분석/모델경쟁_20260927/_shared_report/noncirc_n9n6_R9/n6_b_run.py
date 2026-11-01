# -*- coding: utf-8 -*-
"""N6 단계 B: 동결 챔피언 C11(XGB+로지스틱 순위 평균 0.5/0.5, 시드 0, 학습 t0 2012..v-2) 팔별 재학습·예측.
사용: python n6_b_run.py PANEL OUTCOME ARM[,ARM..]
  PANEL: prov(동결 위험집합) | ntis(라벨 무관 위험집합);  OUTCOME: y(동결, 순환 참고) | y_ntis(라벨 무관)
  ARM: PROV | CW | TXT | NONE | NONE_ALL | M0
산출: preds/n6_{PANEL}_{OUTCOME}_{ARM}_{v}.parquet, preds/n6_{PANEL}_{OUTCOME}_meta_{v}.parquet"""
import sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from r9_common import log, CACHE, PRED, N6, wauc
sys.path.insert(0, N6)
from n6_lib import predict, MIN_TRAIN, DEV_ORIGINS
import numpy as np, pandas as pd, json

SPEC = json.load(open(os.path.join(N6, "final_spec_r5.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; MSPEC = SPEC["model"]; M0F = SPEC["baseline"]["feats"]
P2 = ["pt_mean", "pt_max", "pt_margin", "pt_all"]; DEF = ["k50_assigned", "k50_score", "b1_score"]
drop = lambda L: [f for f in FEATS if f not in L]
ARMS = {"PROV": (FEATS, None), "CW": (FEATS, "alt"), "TXT": (FEATS, "txt"), "NONE": (drop(P2), None), "NONE_ALL": (drop(P2 + DEF), None), "M0": (M0F, None)}
panel, outcome, arms = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
D = pd.read_parquet(os.path.join(CACHE, f"n6_panel_{panel}.parquet"))
D = D[D.t0 >= MIN_TRAIN].reset_index(drop=True)
log("panel", panel, outcome, D.shape, "base", round(D[outcome].mean(), 4))
for v in DEV_ORIGINS:
    tr = D[D.t0 <= v - 2]; te = D[D.t0 == v]
    assert tr.t0.max() + 2 <= 2021
    mp = os.path.join(PRED, f"n6_{panel}_{outcome}_meta_{v}.parquet")
    if not os.path.exists(mp):
        pd.DataFrame({"y": te[outcome].values, "bz": te.bz.values}).to_parquet(mp, index=False)
    for a in arms:
        fe, tag = ARMS[a]
        if tag:
            tr = tr.copy(); te = te.copy()
            for c in P2:
                tr[c] = tr[f"{c}__{tag}"].values; te[c] = te[f"{c}__{tag}"].values
        sp = {"model": "xgb", "seeds": [0]} if a == "M0" else MSPEC
        p = predict(tr, te, fe, ycol=outcome, spec=sp)
        if tag:   # 원래 값 복원
            tr = D[D.t0 <= v - 2]; te = D[D.t0 == v]
        pd.DataFrame({"p": np.asarray(p, float)}).to_parquet(os.path.join(PRED, f"n6_{panel}_{outcome}_{a}_{v}.parquet"), index=False)
        log(panel, outcome, a, v, round(wauc(p, te[outcome].values), 4))
log("end")
