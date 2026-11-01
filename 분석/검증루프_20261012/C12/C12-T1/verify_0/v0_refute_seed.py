# -*- coding: utf-8 -*-
"""verify_0 반박 시도: S1 'M0 대비 CI 하한 ≤ 0'이 (1) 부트스트랩 시드, (2) 학습 시드(XGB 무작위성)만으로도 생기는 정도인지 확인.
공식 스펙 시드는 (0,1,2)로 고정 — 다른 시드 묶음은 정보용 잡음 측정이며 공식·S1 판정을 바꾸지 않는다.
사용: python v0_refute_seed.py <off|s1>  → verify_0/v0_seed_<scen>.json"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); sys.dont_write_bytecode = True
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
sys.path.insert(0, os.path.join(W, "scripts"))
import n5lib
from n5lib import fit_pred, auc, skill, cluster_boot_ci
SPEC = json.load(open(os.path.join(n5lib.MC, "N5_표준특허선언레이더", "final_spec_r4.json"), encoding="utf-8"))
FE, F0 = SPEC["champion_features"], SPEC["baseline_M0"]
scen = sys.argv[1]; t = time.time()
P = lambda t0: pd.read_parquet(os.path.join(W, "cache", "panels", f"{scen}_{t0}.parquet"))
tr = pd.concat([P(t0) for t0 in range(2012, 2021)], ignore_index=True); te = P(2022)
R = {"scen": scen}
pr = pd.read_parquet(os.path.join(W, "cache", f"preds_{scen}.parquet"))
R["boot_seed_ci"] = {}
for s in [2022, 0, 1, 2, 3, 4]:
    (lo, hi), _ = cluster_boot_ci(pr.y.values, pr.p_champ.values, pr.p_M0.values, pr.app.values, n=1000, seed=s)
    R["boot_seed_ci"][s] = [float(lo), float(hi)]
R["train_seed_sets"] = {}
for ss in [(3, 4, 5), (6, 7, 8), (9, 10, 11)]:
    pc = fit_pred(tr[FE].astype(float), tr.y, te[FE].astype(float), seeds=ss)
    p0 = fit_pred(tr[F0].astype(float), tr.y, te[F0].astype(float), seeds=ss)
    a1, a0 = auc(te.y, pc), auc(te.y, p0)
    (lo, hi), _ = cluster_boot_ci(te.y.values, pc, p0, te.app.values, n=1000, seed=2022)
    R["train_seed_sets"][str(ss)] = dict(auc_champ=float(a1), auc_M0=float(a0), skill=float(skill(a1, a0)), ci=[float(lo), float(hi)])
    n5lib.tlog(t, scen, ss, round(skill(a1, a0), 4), round(lo, 4))
json.dump(R, open(os.path.join(V, f"v0_seed_{scen}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
