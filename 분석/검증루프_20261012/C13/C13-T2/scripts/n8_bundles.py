# -*- coding: utf-8 -*-
"""N8 시드 묶음. 공식 final_run_r6.py final()의 fit_score·evaluate(주·강한 기준선 부분)를 옮기고 (XGB random_state s, 음성 5% 표집 시드 n)만 바꾼다.
학습: 트랙 cache/panel_dev.parquet t0 2008–2020(공식과 같은 열 집합·순서). 시험 피처: 트랙 cache/final_test_feats.parquet(v02가 봉인 추출본에서 다시 만든 t0=2022 피처, 읽기만).
시험 레이블: final_preds_r6.parquet의 y. 군집 키: C01-T3 g_owner(공식 CI 비트 재현 확인, cache/n8/p0_cluster.json).
묶음 번호 b = 3*n + s (b=0 → 공식). 사용: python n8_bundles.py 0,1 → cache/n8/b{b}.json"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "n8"))
from c13util import *
import numpy as np, pandas as pd
import n8lib as L

TR_DIR = os.path.join(MC, "N8_특허분쟁표적레이더")
SPEC = json.load(open(os.path.join(TR_DIR, "final_spec_r6.json"), encoding="utf-8"))
CH = SPEC["champion"]; REFS = SPEC["references"]
NEG_FRAC = CH["neg_frac"]
MODELS = {"champ": (CH["feats"], CH["model"]), "M0": (REFS["M0"]["feats"], REFS["M0"]["model"]), "SB": (REFS["SB"]["feats"], REFS["SB"]["model"])}
allf = list(CH["feats"])
for r_ in REFS.values():
    allf += r_["feats"]
# 메모리 4GB 규칙: 공식 load_dev_panel은 로지스틱 참고 모형 열까지 읽지만, 행 순서·행 수는 열 선택과 무관하므로
# 챔피언·M0·SB 피처와 t0·y만 읽고 t0 필터를 읽기 단계에서 건다(행 순서 보존: pyarrow 필터는 순서를 바꾸지 않는다).
need = list(dict.fromkeys(["t0", "y"] + [f for k in MODELS for f in MODELS[k][0]]))
TR = pd.read_parquet(os.path.join(TR_DIR, "cache", "panel_dev.parquet"), columns=need,
                     filters=[("t0", ">=", SPEC["final"]["train_t0"][0]), ("t0", "<=", SPEC["final"]["train_t0"][1])]).reset_index(drop=True)
TE = pd.read_parquet(os.path.join(TR_DIR, "cache", "final_test_feats.parquet"))
FP = pd.read_parquet(os.path.join(TR_DIR, "final_preds_r6.parquet"))
assert (TE.rgst_no.astype(str).values == FP.rgst_no.astype(str).values).all(), "시험 행 순서가 동결 예측과 다름"
CL = pd.read_parquet(os.path.join(BASE, r"분석\검증루프_20261012\C01\C01-T3\_clusters_internal.parquet"))
CL["rgst_no"] = CL.rgst_no.astype(str)
g = FP[["rgst_no"]].astype(str).merge(CL, on="rgst_no", how="left").g_owner.values
y = FP.y.values.astype(int)
tlog("loaded", len(TR), int(TR.y.sum()), len(TE), int(y.sum()))
OD = tdir("n8")


def fit_score(feats, model, s, n):
    rng = np.random.default_rng(n)
    yy = TR.y.values.astype(int)
    keep = (yy == 1) | (rng.random(len(TR)) < NEG_FRAC)
    w = np.where(yy == 1, 1.0, 1.0 / NEG_FRAC)[keep]
    return L.fit_pred(TR.loc[keep, feats].astype(float), yy[keep], TE[feats].astype(float), model, seed=s, w=w)


for b in [int(x) for x in sys.argv[1].split(",")]:
    fo = os.path.join(OD, f"b{b}.json")
    if os.path.exists(fo):
        tlog("skip", b); continue
    s, n = b % 3, b // 3
    P = {k: fit_score(f, m, s, n) for k, (f, m) in MODELS.items()}
    tlog("fit", b, (s, n))
    ci, cia, cib = L.boot_ci(y, P["champ"], P["M0"], g, n=1000, seed=2022)
    cs, _, csb = L.boot_ci(y, P["champ"], P["SB"], g, n=1000, seed=2023)
    a, a0, asb = L.auc(y, P["champ"]), L.auc(y, P["M0"]), L.auc(y, P["SB"])
    t1 = top1(y, P["champ"]); t1sb = top1(y, P["SB"])
    extra = dict(strong_name="SB", top1_capture_SB=t1sb["capture"], hit100=L.topk(y, P["champ"], k=100)[1], peak_gb=rss())
    if b == 0:
        extra["max_pred_diff_vs_frozen"] = {k: float(np.abs(P[k] - FP[f"p_{k}"].values).max()) for k in P}
    r = row("N8", b, dict(xgb_seed=s, neg_seed=n), a, a0, L.skill(a, a0), ci, asb, L.skill(a, asb), cs, t1, len(y), y.sum(), extra=extra)
    pd.DataFrame({"rgst_no": FP.rgst_no.values, "g": g, "y": y, **{f"p_{k}": v for k, v in P.items()}}).to_parquet(os.path.join(OD, f"preds_b{b}.parquet"), index=False)
    dump(r, fo)
    tlog("done", b, round(r["skill"], 4), [round(x, 4) for x in ci], "SB", round(r["skill_vs_strong"], 4), [round(x, 4) for x in cs])
