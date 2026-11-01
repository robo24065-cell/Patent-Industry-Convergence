# -*- coding: utf-8 -*-
"""N9 시드 묶음. 공식 final_run_r6.py final()의 학습·채점을 옮기고 (XGB random_state s, 음성 30% 표집 시드 n)만 바꾼다.
패널: 트랙 cache/panel_final.parquet + txtclf_final_2022.parquet(공식 final 실행 01:54–01:56, 읽기만). 공용 코드: scripts/n9/n9_lib.py(원본 동일).
묶음 번호 b = 3*n + s (b=0 → 공식 (s,n)=(0,0)). 사용: python n9_bundles.py 0,1 → cache/n9/b{b}.json"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n9"))
from c13util import *
import numpy as np, pandas as pd
import n9_lib as L

TR_DIR = os.path.join(MC, "N9_정부RnD과제_특허성과예측")
SPEC = json.load(open(os.path.join(TR_DIR, "final_spec_r6.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; BS = {k: v["feats"] for k, v in SPEC["references"].items()}
D = pd.read_parquet(os.path.join(TR_DIR, "cache", "panel_final.parquet")).reset_index(drop=True)
tc = pd.read_parquet(os.path.join(TR_DIR, "cache", "txtclf_final_2022.parquet"))
X = D.copy(); X["txt_clf"] = np.nan; X.loc[tc.row_id.values, "txt_clf"] = tc.txt_clf.values
v = 2022; lo, hi = SPEC["final"]["train_t"]
te = X[X.t0 == v]; y = te.y.values; g = te.org.values
OD = tdir("n9")


def predict(feats, s, n):
    rng = np.random.default_rng(n)
    keep = (X.y.values == 1) | (rng.random(len(X)) < L.NEG_FRAC)
    w = np.where(X.y.values == 1, 1.0, 1.0 / L.NEG_FRAC)
    msk = (X.t0 >= lo).values & (X.t0 <= hi).values & keep
    tr = X[msk]
    return L.fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), model="xgb", seed=s, w=w[msk])


for b in [int(x) for x in sys.argv[1].split(",")]:
    fo = os.path.join(OD, f"b{b}.json")
    if os.path.exists(fo):
        tlog("skip", b); continue
    s, n = b % 3, b // 3
    p = predict(FEATS, s, n); p0 = predict(BS["M0_L2"], s, n); ps = predict(BS["strong_L2"], s, n)
    tlog("fit", b, (s, n))
    ci = L.boot_ci(y, p, p0, groups=g, n=1000, seed=2022)
    c2 = L.boot_ci(y, p, ps, groups=g, n=1000, seed=2023)
    a, a0, a2 = L.auc(y, p), L.auc(y, p0), L.auc(y, ps)
    t1 = top1(y, p)
    r = row("N9", b, dict(xgb_seed=s, neg_seed=n), a, a0, L.skill(a, a0), ci["skill_ci"], a2, L.skill(a, a2), c2["skill_ci"], t1, len(y), y.sum(),
            extra=dict(strong_name="strong_L2", top1_prec_official_rule=L.topk(y, p, 0.01)[0], hit100=L.topk(y, p, k=100)[1], peak_gb=rss()))
    pd.DataFrame({"org": g, "y": y, "p_champ": p, "p_M0_L2": p0, "p_strong_L2": ps}).to_parquet(os.path.join(OD, f"preds_b{b}.parquet"), index=False)
    dump(r, fo)
    tlog("done", b, round(r["skill"], 4), [round(x, 4) for x in ci["skill_ci"]], "strong", round(r["skill_vs_strong"], 4), [round(x, 4) for x in c2["skill_ci"]])
