# -*- coding: utf-8 -*-
"""N8 P0-a: 시험 군집 키 확인. 공식 owner_v2는 시험 패널에 저장돼 있지 않다(final_test_feats.parquet에 없음).
C01-T3 _clusters_internal.parquet의 g_owner(공식 방식 재구성)로 동결 예측의 공식 CI가 비트 단위로 나오는지 본다. → cache/n8/p0_cluster.json"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n8"))
from c13util import *
import numpy as np, pandas as pd
import n8lib as L

TR_DIR = os.path.join(MC, "N8_특허분쟁표적레이더")
P = pd.read_parquet(os.path.join(TR_DIR, "final_preds_r6.parquet"))
R = json.load(open(os.path.join(TR_DIR, "final_result_r6.json"), encoding="utf-8"))
CL = pd.read_parquet(os.path.join(BASE, r"분석\검증루프_20261012\C01\C01-T3\_clusters_internal.parquet"))
P["rgst_no"] = P.rgst_no.astype(str); CL["rgst_no"] = CL.rgst_no.astype(str)
g = P[["rgst_no"]].merge(CL, on="rgst_no", how="left").g_owner
assert g.notna().all() and len(g) == len(P)
y = P.y.values.astype(int)
out = {"n": len(P), "n_groups": int(g.nunique())}
ci, cia, cib = L.boot_ci(y, P.p_champ.values, P.p_M0.values, g.values, n=1000, seed=2022)
cs, _, csb = L.boot_ci(y, P.p_champ.values, P.p_SB.values, g.values, n=1000, seed=2023)
out["ci_M0"] = ci; out["ci_SB"] = cs; out["official_ci_M0"] = R["skill_vs_M0_ci"]; out["official_ci_SB"] = R["skill_vs_SB_ci"]
out["max_diff"] = float(max(abs(a - b) for a, b in zip(ci + cs, R["skill_vs_M0_ci"] + R["skill_vs_SB_ci"])))
out["pass"] = out["max_diff"] <= 1e-9
dump(out, os.path.join(tdir("n8"), "p0_cluster.json"))
print(json.dumps(out, ensure_ascii=False, indent=1))
tlog("end")
