# -*- coding: utf-8 -*-
r"""C19-T2 s3: 변형 채점. 공식 score_final 경로(cls 3배깅 로짓 평균→float32 확률, haz HZ.combine, 로짓 가중 cls .2/haz .4/logit .2/hgb .2).
공식(frozen) = 공식 runs_final/F22/pred 의 배깅 파일로 같은 경로를 다시 계산(C13-T2 b0에서 AUC 공식과 비트 동일 확인된 경로).
 - 각 변형: AUC, skill(M0), M0 대비 CI(cboot B=1000 seed 0 = final_run_r2.cluster_boot), 강화 기준선 대비 skill(float64 경로)·CI(sboot B=300 seed 3 = VERIFY_R5)
 - 짝 부트스트랩(공식 대비): 같은 군집 재표본 B=1000 seed 0, Δskill·ΔAUC 95% 백분위 CI
 - 상위 1,000·상위 1%(2,876) 적중, hit100, 공식 상위 1,000과의 교집합
사용: python s3_score.py P0,F1,... → cache/score_{var}.json, cache/pred_{var}.npy"""
import os, sys, json
sys.dont_write_bytecode = True
os.environ["T4R2_MODE"] = "final"; os.environ.pop("T4R2_SAVE", None)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib19 import *
sys.path.insert(0, J(MIRROR, "R2"))
import numpy as np, pandas as pd
import r2lib as R
from r2boot import wauc_prep, wauc
from sklearn.metrics import roc_auc_score

FS = json.load(open(J(MIRROR, "R2", "final_spec_r2.json"), encoding="utf-8")); SPEC = FS["spec"]
ev = pd.read_parquet(J(F22, "ev.parquet"), columns=["pid", "acd", "y"])
row_h = pd.read_parquet(J(F22, "ev_haz.parquet"), columns=["row"]).row.values
N = len(ev); y = ev.y.values.astype(float); assert np.isfinite(y).all()
SB = np.load(J(T4, "VERIFY_R5", "cache", "strong_full.npy"))
FROZ = J(F22, "pred")
key = {c: R.comp_key(SPEC, c) for c in ("cls", "haz", "logit", "hgb")}
PB = np.load(J(FROZ, [f for f in os.listdir(FROZ) if f.startswith("base_")][0]))
cl = np.where(ev.acd.values == "NA", "P" + ev.pid.astype(str).values, ev.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ws = SPEC["ensemble"]["weights"]


def load(d):
    ps = {}
    for comp in ("cls", "haz"):
        z = np.mean([np.load(J(d, f"{key[comp]}_b{k}.npy")) for k in range(3)], 0)
        p = 1 / (1 + np.exp(-z))
        if comp == "haz":
            p = R.HZ.combine(pd.DataFrame({"row": row_h}), p, N)
        ps[comp] = p
    for comp in ("logit", "hgb"):
        ps[comp] = np.load(J(d, f"{key[comp]}.npy"))
    return ps


def ens(ps, f64=False):
    q = dict(ps)
    if f64:
        q["cls"] = q["cls"].astype("float64")
    z = sum(w * lg(q[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
    return 1 / (1 + np.exp(-z))


def topk(p, k):
    o = np.argsort(-p, kind="stable")[:k]
    return o, int(y[o].sum())


ps0 = load(FROZ); p0 = ens(ps0); p0_64 = ens(ps0, True)
a0, ab = float(roc_auc_score(y, p0)), float(roc_auc_score(y, PB))
pm0, pmb = wauc_prep(y, p0), wauc_prep(y, PB)
o0_1000, _ = topk(p0, 1000)


def score(var):
    d = FROZ if var == "OFFICIAL" else J(C, "runs", var, "pred")
    ps = load(d); p = ens(ps); p64 = ens(ps, True)
    a = float(roc_auc_score(y, p)); a64 = float(roc_auc_score(y, p64)); asb = float(roc_auc_score(y, SB))
    r = dict(var=var, n=N, pos=int(y.sum()), auc=a, auc_M0=ab, skill=(a - ab) / (1 - ab), auc_float64=a64, auc_strong=asb, skill_vs_strong=(a64 - asb) / (1 - asb),
             comps_auc={k: float(roc_auc_score(y, v)) for k, v in ps.items()})
    if var != "OFFICIAL":
        r["max_comp_diff_vs_official"] = {k: float(np.abs(ps[k] - ps0[k]).max()) for k in ps}
        r["max_ens_diff_vs_official"] = float(np.abs(p - p0).max())
    pm = wauc_prep(y, p)
    rng = np.random.default_rng(0); S, D, DA = [], [], []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        a_, ab_, a0_ = wauc(pm, w), wauc(pmb, w), wauc(pm0, w)
        s_ = (a_ - ab_) / (1 - ab_); s0_ = (a0_ - ab_) / (1 - ab_)
        S.append(s_); D.append(s_ - s0_); DA.append(a_ - a0_)
    r["ci_M0"] = [float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))]
    r["ci_M0_4dp"] = [round(x, 4) for x in r["ci_M0"]]
    r["d_skill"] = r["skill"] - (a0 - ab) / (1 - ab); r["d_auc"] = a - a0
    r["d_skill_ci"] = [float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))]
    r["d_auc_ci"] = [float(np.percentile(DA, 2.5)), float(np.percentile(DA, 97.5))]
    rng = np.random.default_rng(3); yy = y.astype(int); out_ = []
    for _ in range(300):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        a_ = roc_auc_score(yy, p64, sample_weight=w); b_ = roc_auc_score(yy, SB, sample_weight=w); out_.append((a_ - b_) / (1 - b_))
    r["ci_strong"] = [float(np.percentile(out_, 2.5)), float(np.percentile(out_, 97.5))]
    r["class_M0"] = "견고" if r["ci_M0"][0] > 0 else "비견고"
    r["class_strong"] = "강화 우위" if r["ci_strong"][0] > 0 else "조건부"
    o1000, h1000 = topk(p, 1000); _, h1 = topk(p, int(round(0.01 * N))); _, h100 = topk(p, 100)
    r.update(top1000_hits=h1000, top1pct_k=int(round(0.01 * N)), top1pct_hits=h1, hit100=h100,
             top1000_overlap_official=int(len(np.intersect1d(o1000, o0_1000))))
    np.save(J(C, f"pred_{var}.npy"), p)
    jdump(r, J(C, f"score_{var}.json"))
    out(var, round(r["auc"], 6), round(r["skill"], 6), "d", round(r["d_skill"], 5), [round(x, 5) for x in r["d_skill_ci"]], r["class_M0"], r["class_strong"], r["ci_strong"])
    return r


for v in sys.argv[1].split(","):
    score(v)
