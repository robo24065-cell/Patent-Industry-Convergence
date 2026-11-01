# -*- coding: utf-8 -*-
r"""verify_1 (반박 검증) — 작업자 코드(c10_common, b_lib, summ)를 import하지 않고 저장 예측에서 다시 계산.
(1) 공식 seed(k0,p0) 원점별 sklearn AUC: P50, FREE50, TXT50, FREEIPC50, HSC1_50, HMG3_50, P50FIPC, P50HSC1, P50HMG3
(2) 발견의 Δ: FIPC−FREE50, HSC1−FREE50, (P50+X)−P50, (P50+X)−X, 우세 원점 수
(3) seed 변동: 공개끼리 비교(FIPC−FREE50, HSC1−FREE50)를 가용 5구성 (k0,p0),(k0,p1),(k0,p2),(k1,p0),(k2,p0)에서
(4) 원거리 AUC(p_cls==0)·top1000 — 다른 지표에서도 TXT-FREE가 IPC 최빈표 계열보다 강한지
출력: verify_1/v1_point.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
NC8 = os.path.join(BASE, r"분석\모델경쟁_20260927\T1_산업대기실\NONCIRC_R8\cache")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE); C = os.path.join(W, "cache")
CS = [2008, 2009, 2010, 2011, 2012]
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)


def rd(c, k, tag, cols):
    return pd.read_parquet(os.path.join(C, f"pred_c{c}_k{k}_{tag}.parquet"), columns=cols)


out = {"auc": {}, "far": {}, "top1000": {}, "seed": {}}
for c in CS:
    a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["first_ky", "p_cls"])
    y = a.first_ky.notna().astype(int).values; far = (a.p_cls == 0).values
    P = pd.concat([rd(c, 0, "P50-FREE50-TXT50_p012", ["P50_p0", "FREE50_p0", "TXT50_p0", "FREE50_p1", "FREE50_p2"]),
                   rd(c, 0, "FREEIPC50-HSC1_50-P50HSC1_p0", None),
                   rd(c, 0, "HMG3_50-P50FIPC-P50HMG3_p0", None),
                   rd(c, 0, "FREEIPC50-HSC1_50_p1", None), rd(c, 0, "FREEIPC50-HSC1_50_p2", None)], axis=1)
    assert len(P) == len(y)
    arms = ["P50", "FREE50", "TXT50", "FREEIPC50", "HSC1_50", "HMG3_50", "P50FIPC", "P50HSC1", "P50HMG3"]
    out["auc"][c] = {m: roc_auc_score(y, P[m + "_p0"].astype(np.float64)) for m in arms}
    out["far"][c] = {m: roc_auc_score(y[far], P[m + "_p0"].astype(np.float64)[far]) for m in arms}
    out["top1000"][c] = {m: float(y[np.argsort(-P[m + "_p0"].values, kind="stable")[:1000]].mean()) for m in arms}
    sd = {}
    for p in (1, 2):
        sd[f"k0p{p}"] = {m: roc_auc_score(y, P[f"{m}_p{p}"].astype(np.float64)) for m in ("FREE50", "FREEIPC50", "HSC1_50")}
    sd["k0p0"] = {m: out["auc"][c][m] for m in ("FREE50", "FREEIPC50", "HSC1_50")}
    for k in (1, 2):
        q = pd.concat([rd(c, k, "P50-FREE50-TXT50_p0", ["FREE50_p0"]), rd(c, k, "FREEIPC50-HSC1_50_p0", None)], axis=1)
        sd[f"k{k}p0"] = {m: roc_auc_score(y, q[m + "_p0"].astype(np.float64)) for m in ("FREE50", "FREEIPC50", "HSC1_50")}
    out["seed"][c] = sd
    log(c, {m: round(v, 5) for m, v in out["auc"][c].items()})

D = lambda tab, a, b: [tab[c][a] - tab[c][b] for c in CS]
pairs = {"FIPC-FREE50": ("FREEIPC50", "FREE50"), "HSC1-FREE50": ("HSC1_50", "FREE50"), "HMG3-FREE50": ("HMG3_50", "FREE50"),
         "TXT50-FREE50": ("TXT50", "FREE50"),
         "P50FIPC-P50": ("P50FIPC", "P50"), "P50HSC1-P50": ("P50HSC1", "P50"), "P50HMG3-P50": ("P50HMG3", "P50"),
         "P50FIPC-FIPC": ("P50FIPC", "FREEIPC50"), "P50HSC1-HSC1": ("P50HSC1", "HSC1_50"), "P50HMG3-HMG3": ("P50HMG3", "HMG3_50"),
         "P50-FREE50": ("P50", "FREE50"), "P50-FIPC": ("P50", "FREEIPC50"), "P50-HSC1": ("P50", "HSC1_50")}
out["delta"] = {}
for nm, (a, b) in pairs.items():
    d = D(out["auc"], a, b)
    out["delta"][nm] = {"mean": float(np.mean(d)), "per_origin": d, "n_pos": int(sum(x > 0 for x in d)),
                        "far_mean": float(np.mean(D(out["far"], a, b))), "top1000_mean": float(np.mean(D(out["top1000"], a, b)))}
out["mean_auc"] = {m: float(np.mean([out["auc"][c][m] for c in CS])) for m in out["auc"][CS[0]]}
out["mean_far"] = {m: float(np.mean([out["far"][c][m] for c in CS])) for m in out["far"][CS[0]]}
out["mean_top1000"] = {m: float(np.mean([out["top1000"][c][m] for c in CS])) for m in out["top1000"][CS[0]]}
sk = {}
for cfg in ("k0p0", "k0p1", "k0p2", "k1p0", "k2p0"):
    f = [out["seed"][c][cfg]["FREEIPC50"] - out["seed"][c][cfg]["FREE50"] for c in CS]
    h = [out["seed"][c][cfg]["HSC1_50"] - out["seed"][c][cfg]["FREE50"] for c in CS]
    sk[cfg] = {"FIPC-FREE50": float(np.mean(f)), "FIPC_npos": int(sum(x > 0 for x in f)), "HSC1-FREE50": float(np.mean(h)), "HSC1_npos": int(sum(x > 0 for x in h))}
out["seed_summary"] = sk
json.dump(out, open(os.path.join(HERE, "v1_point.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("delta", json.dumps({k: (round(v["mean"], 5), v["n_pos"], round(v["far_mean"], 5), round(v["top1000_mean"], 4)) for k, v in out["delta"].items()}))
log("mean_auc", json.dumps({k: round(v, 5) for k, v in out["mean_auc"].items()}))
log("seed", json.dumps(sk))
