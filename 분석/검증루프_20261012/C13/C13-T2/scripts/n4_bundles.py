# -*- coding: utf-8 -*-
"""N4 시드 묶음. 공식 final_run_r4.py(final 모드)의 학습·채점 부분을 그대로 옮기고 XGB 시드 묶음만 바꾼다.
패널: 트랙 cache/final_panel.parquet(공식 final 실행이 23:20에 쓴 것, 읽기만). 공용 코드: scripts/n4/n4lib.py(sha256 원본 동일).
사용: python n4_bundles.py 0,1,2  → cache/n4/b{b}.json, preds_b{b}.parquet"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n4"))
from c13util import *
import numpy as np, pandas as pd
import n4lib as L

TR_DIR = os.path.join(MC, "N4_특허행태_기업폐업조기경보")
SPEC = json.load(open(os.path.join(TR_DIR, "final_spec_r4.json"), encoding="utf-8"))
FE, M0F = SPEC["features"], SPEC["baseline_M0"]
D = pd.read_parquet(os.path.join(TR_DIR, "cache", "final_panel.parquet"))
tr = D[D.t0.isin(list(range(2015, 2022)))]; te = D[D.t0 == 2023]
assert tr.y.notna().all() and te.y.notna().all()
y = te.y.values; cl = te.bizno.values
OD = tdir("n4")
for b in [int(x) for x in sys.argv[1].split(",")]:
    fo = os.path.join(OD, f"b{b}.json")
    if os.path.exists(fo):
        tlog("skip", b); continue
    seeds = (0, 1, 2) if b == 0 else (3 * b, 3 * b + 1, 3 * b + 2)
    fit = lambda feats, model: L.predict(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, seeds=seeds)
    pc = fit(FE, "ens"); p0 = fit(M0F, "xgb"); p0e = fit(M0F, "ens")
    tlog("fit", b, seeds)
    ci = L.cluster_boot(y, pc, p0, cl, n=1000, seed=2023)
    ce = L.cluster_boot(y, pc, p0e, cl, n=1000, seed=2024)
    a, a0, ae = L.auc(y, pc), L.auc(y, p0), L.auc(y, p0e)
    k = max(1, int(0.01 * len(te)))
    t1 = top1(y, pc); t1["prec_official_rule"] = float(y[np.argsort(-pc)[:k]].mean())
    r = row("N4", b, list(seeds), a, a0, L.skill(a, a0), ci["skill_ci"], ae, L.skill(a, ae), ce["skill_ci"], t1, len(y), y.sum(),
            extra=dict(top1_prec_official_rule=t1["prec_official_rule"], strong_name="M0-ens", n_boot=ci["n_boot"], peak_gb=rss()))
    pd.DataFrame({"bizno": cl, "y": y, "p_champ": pc, "p_M0": p0, "p_M0ens": p0e}).to_parquet(os.path.join(OD, f"preds_b{b}.parquet"), index=False)
    dump(r, fo)
    tlog("done", b, round(r["skill"], 4), [round(v, 4) for v in ci["skill_ci"]], "strong", round(r["skill_vs_strong"], 4), [round(v, 4) for v in ce["skill_ci"]])
