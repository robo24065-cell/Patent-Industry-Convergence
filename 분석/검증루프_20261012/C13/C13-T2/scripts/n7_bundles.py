# -*- coding: utf-8 -*-
"""N7 시드 묶음. 공식 final_run_r5.py(final)의 학습·채점을 옮기고 XGB 시드 묶음만 바꾼다.
패널: 트랙 cache/final_panel.parquet(공식 final 실행 21:30, 읽기만). 공용 코드: scripts/n7/n7lib.py(원본 동일).
사용: python n7_bundles.py 0,1 → cache/n7/b{b}.json"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n7"))
from c13util import *
import numpy as np, pandas as pd
import n7lib as L

TR_DIR = os.path.join(MC, "N7_특허기업_정부RnD첫수행")
SPEC = json.load(open(os.path.join(TR_DIR, "final_spec_r5.json"), encoding="utf-8"))
FE, M0F, SBF = SPEC["features"], SPEC["baseline_M0"], SPEC["strong_baseline_SB"]
CM = SPEC["model"]["type"]
D = pd.read_parquet(os.path.join(TR_DIR, "cache", "final_panel.parquet"))
tr = D[D.t0.isin(list(range(2015, 2021)))]; te = D[D.t0 == 2022]
assert tr.y.notna().all() and te.y.notna().all()
y = te.y.values; cl = te.bizno.values
OD = tdir("n7")
for b in [int(x) for x in sys.argv[1].split(",")]:
    fo = os.path.join(OD, f"b{b}.json")
    if os.path.exists(fo):
        tlog("skip", b); continue
    seeds = (0, 1, 2) if b == 0 else (3 * b, 3 * b + 1, 3 * b + 2)
    fit = lambda feats, model: L.predict(tr[feats].astype(float), tr["y"], te[feats].astype(float), model=model, seeds=seeds)
    pc = fit(FE, CM); p0 = fit(M0F, "xgb"); psb = fit(SBF, "xgb"); p0e = fit(M0F, "ens")
    tlog("fit", b, seeds)
    ci = L.cluster_boot(y, pc, p0, cl, n=1000, seed=2022)
    cs = L.cluster_boot(y, pc, psb, cl, n=1000, seed=2023)
    ce = L.cluster_boot(y, pc, p0e, cl, n=1000, seed=2024)
    a, a0, asb, ae = L.auc(y, pc), L.auc(y, p0), L.auc(y, psb), L.auc(y, p0e)
    prec, rec = L.topk(y, pc, 0.01)
    t1 = top1(y, pc)
    r = row("N7", b, list(seeds), a, a0, L.skill(a, a0), ci["skill_ci"], asb, L.skill(a, asb), cs["skill_ci"], t1, len(y), y.sum(),
            extra=dict(strong_name="SB(xgb)", top1_prec_official_rule=prec, top1_capture_official_rule=rec,
                       auc_M0ens=float(ae), skill_vs_M0ens=float(L.skill(a, ae)), M0ens_ci_lo=ce["skill_ci"][0], M0ens_ci_hi=ce["skill_ci"][1], peak_gb=rss()))
    pd.DataFrame({"bizno": cl, "y": y, "p_champ": pc, "p_M0": p0, "p_SB": psb, "p_M0ens": p0e}).to_parquet(os.path.join(OD, f"preds_b{b}.parquet"), index=False)
    dump(r, fo)
    tlog("done", b, round(r["skill"], 4), [round(v, 4) for v in ci["skill_ci"]], "SB", round(r["skill_vs_strong"], 4), [round(v, 4) for v in cs["skill_ci"]])
