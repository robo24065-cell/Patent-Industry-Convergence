# -*- coding: utf-8 -*-
"""N6 시드 묶음. 공식 final_run_r5.py final()의 학습·채점을 옮기고 XGB 시드(공식 스펙은 1개, [0])만 바꾼다.
패널: cache/n6/panel_final.parquet(n6_build.py가 공식 build("final")로 다시 만든 것). 공용 코드: scripts/n6/n6_lib.py(원본 동일).
사용: python n6_bundles.py 0,1 → cache/n6/b{b}.json"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n6"))
from c13util import *
import numpy as np, pandas as pd
import n6_lib as L

TR_DIR = os.path.join(MC, "N6_정부RnD기업_첫특허전환")
SPEC = json.load(open(os.path.join(TR_DIR, "final_spec_r5.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; MSPEC = SPEC["model"]; M0F = SPEC["baseline"]["feats"]
D = pd.read_parquet(os.path.join(tdir("n6"), "panel_final.parquet"))
tr = D[(D.t0 >= SPEC["final"]["train_t0"][0]) & (D.t0 <= SPEC["final"]["train_t0"][1])]
te = D[D.t0 == SPEC["final"]["test_t0"]]
y = te.y.values; gr = te.bz.values
OD = tdir("n6")
FP = pd.read_parquet(os.path.join(TR_DIR, "final_preds_r5.parquet"))
for b in [int(x) for x in sys.argv[1].split(",")]:
    fo = os.path.join(OD, f"b{b}.json")
    if os.path.exists(fo):
        tlog("skip", b); continue
    sp = dict(MSPEC); sp["seeds"] = [b]
    p = L.predict(tr, te, FEATS, spec=sp)
    pb = L.predict(tr, te, M0F, spec={"model": "xgb", "seeds": [b]})
    pe = L.predict(tr, te, M0F, spec={"model": "ens", "seeds": [b], "w_xgb": 0.5})
    tlog("fit", b)
    ci = L.boot_ci(y, p, pb, n=1000, seed=2021, groups=gr)
    c3 = L.boot_ci(y, p, pe, n=1000, seed=2023, groups=gr)
    a, a0, ae = L.auc(y, p), L.auc(y, pb), L.auc(y, pe)
    t1 = top1(y, p)
    extra = dict(strong_name="M0-ens", top1_prec_official_rule=L.topk(y, p, 0.01), peak_gb=rss())
    if b == 0:
        same = bool((FP.bz.values == gr).all() and (FP.y.values == y).all())
        extra["frozen_rows_labels_same"] = same
        extra["max_pred_diff_vs_frozen"] = dict(champ=float(np.abs(FP.p_champ.values - p).max()), M0=float(np.abs(FP.p_m0.values - pb).max())) if same else None
    r = row("N6", b, [b], a, a0, L.skill(a, a0), ci["skill_ci"], ae, L.skill(a, ae), c3["skill_ci"], t1, len(y), y.sum(), extra=extra)
    pd.DataFrame({"bz": gr, "y": y, "p_champ": p, "p_M0": pb, "p_M0ens": pe}).to_parquet(os.path.join(OD, f"preds_b{b}.parquet"), index=False)
    dump(r, fo)
    tlog("done", b, round(r["skill"], 4), [round(x, 4) for x in ci["skill_ci"]], "strong", round(r["skill_vs_strong"], 4), [round(x, 4) for x in c3["skill_ci"]])
