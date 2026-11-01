# -*- coding: utf-8 -*-
"""C15-T3 verify_2: 반박 검증 — 제공(drop_ros) − hyb_pkr 쌍을 독립 코드(sklearn AUC, 셀 키 병합 정렬)로 재계산.
rng: seed 20261005 새 Generator, c5 호출 순서상 앞 6쌍(txt 3, free 3)의 소비를 셀 수만으로 재생 후 7번째 쌍.
읽기만. 쓰기: verify_2/v2_paired.json"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V2 = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V2)
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
R7 = os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "results")
C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10")
ORIG = list(range(2010, 2019)); B = 1000

dr = pd.read_parquet(os.path.join(R7, "pub_drop_ros_preds.parquet"))
full = pd.read_parquet(os.path.join(R7, "pub_full_preds.parquet"))
kr = pd.read_parquet(os.path.join(R7, "pub_kr_rebuild_preds.parquet"))
stored = pd.read_parquet(os.path.join(C5, "t3_results", "hyb_pkr_preds.parquet"))


def load(tag):
    fs = sorted(glob.glob(os.path.join(W, "res", "t3", f"{tag}__*_preds.parquet")))
    parts = [pd.read_parquet(f) for f in fs]
    d = pd.concat(parts, ignore_index=True)
    info = dict(files=[os.path.basename(f) for f in fs], t0_per_file=[sorted(p.t0.unique().tolist()) for p in parts],
                dup=int(d.duplicated(["t0", "cell"]).sum()), n=len(d))
    # json by_origin AUC vs preds
    js = {}
    for f in fs:
        j = json.load(open(f.replace("_preds.parquet", ".json"), encoding="utf-8"))
        for k, v in j["by_origin"].items():
            js[int(k)] = v["model"]["auc"]
    info["json_auc"] = js
    return d, info


def auc(y, p):
    return roc_auc_score(y, p)


def pair(A, Bdf, rng):
    d, bs = [], np.full((len(ORIG), B), np.nan)
    for i, v in enumerate(ORIG):
        m = A[A.t0 == v][["cell", "yb", "p"]].merge(Bdf[Bdf.t0 == v][["cell", "yb", "p"]], on="cell", suffixes=("a", "b"), validate="1:1")
        assert len(m) == (A.t0 == v).sum() == (Bdf.t0 == v).sum()
        assert (m.yba == m.ybb).all()
        m = m.sort_values("cell").reset_index(drop=True)
        y = m.yba.values.astype(int); pa = m.pa.values; pb = m.pb.values
        d.append(auc(y, pa) - auc(y, pb))
        I = rng.integers(0, len(y), (B, len(y)))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            if 0 < yy.sum() < len(yy):
                bs[i, j] = auc(yy, pa[ii]) - auc(yy, pb[ii])
    d = np.array(d); bm = np.nanmean(bs, 0)
    return dict(dAUC_raw=float(d.mean()), dAUC=round(float(d.mean()), 4), ci=[round(float(np.nanpercentile(bm, 2.5)), 4), round(float(np.nanpercentile(bm, 97.5)), 4)],
                wins=f"{int((d > 0).sum())}/9", by_origin=[round(float(x), 5) for x in d])


def burn(rng, k):
    # 앞 k쌍의 rng 소비 재생: 쌍마다 원점별 integers(0, n_v, (B, n_v))
    for _ in range(k):
        for v in ORIG:
            n = int((dr.t0 == v).sum()); rng.integers(0, n, (B, n))


R = {}
arms = {"stored": (stored, None)}
for tag in ["hyb_pkr_reg", "hyb_pkr_pure", "repro_hyb_pkr"]:
    arms[tag] = load(tag)
for nm, (d, info) in arms.items():
    rec = {"info": info}
    rec["auc_by_origin"] = {v: round(auc(d[d.t0 == v].yb, d[d.t0 == v].p), 5) for v in ORIG}
    rec["auc_mean"] = float(np.mean(list(rec["auc_by_origin"].values())))
    if info:
        rec["json_vs_preds_maxdiff"] = max(abs(info["json_auc"][v] - rec["auc_by_origin"][v]) for v in ORIG)
    rng = np.random.default_rng(20261005); burn(rng, 6)
    rec["drop_ros_minus"] = pair(dr, d, rng)
    # 견고성: 다른 시드 3개(rng 순서 무관)
    rec["alt_seeds"] = {}
    for s in ([7] if nm in ("hyb_pkr_reg", "hyb_pkr_pure") else []):
        r2 = np.random.default_rng(s)
        rec["alt_seeds"][s] = pair(dr, d, r2)
    R[nm] = rec
    print(nm, round(rec["auc_mean"], 5), rec["drop_ros_minus"], {s: (x["dAUC"], x["ci"], x["wins"]) for s, x in rec["alt_seeds"].items()}, flush=True)
R["auc_drop_ros"] = float(np.mean([auc(dr[dr.t0 == v].yb, dr[dr.t0 == v].p) for v in ORIG]))
R["auc_kr"] = float(np.mean([auc(kr[kr.t0 == v].yb, kr[kr.t0 == v].p) for v in ORIG]))
json.dump(R, open(os.path.join(V2, "v2_paired.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("drop_ros", R["auc_drop_ros"], "kr", R["auc_kr"])
