# -*- coding: utf-8 -*-
# verify_5 (반박 검증) 1: 대안 B(hyb_pkr_pure) AUC·보존율·제공-hyb_pkr 쌍 부트스트랩을 독립 코드(sklearn AUC)로 재계산.
# 읽기만: R7 results pub_*_preds, C15-T3 res/t3/hyb_pkr_pure__*_preds, hyb_pkr_reg, C10-T3 hyb_txt_reg/hyb_free_reg(rng 순서용 n만)
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; W = BASE + "/분석/검증루프_20261012/C15/C15-T3"
R7 = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/results"
ORIG = list(range(2010, 2019))
def load(tag):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(W + f"/res/t3/{tag}__*_preds.parquet"))], ignore_index=True)
M = {"drop_ros": pd.read_parquet(R7 + "/pub_drop_ros_preds.parquet"), "kr": pd.read_parquet(R7 + "/pub_kr_rebuild_preds.parquet"),
     "full": pd.read_parquet(R7 + "/pub_full_preds.parquet"), "pure": load("hyb_pkr_pure"), "reg": load("hyb_pkr_reg")}
out = {}
for k, d in M.items():
    assert not d.duplicated(["t0", "cell"]).any(); assert sorted(d.t0.unique()) == ORIG, k
    out.setdefault("n_rows", {})[k] = len(d)
# 같은 셀·레이블인가
key = M["drop_ros"].set_index(["t0", "cell"]).yb.sort_index()
for k in ("pure", "reg", "kr", "full"):
    o = M[k].set_index(["t0", "cell"]).yb.sort_index()
    out.setdefault("same_cells_labels", {})[k] = bool(o.index.equals(key.index) and (o.values == key.values).all())
au = {k: [roc_auc_score(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG] for k, d in M.items()}
out["auc_by_origin"] = {k: [round(x, 5) for x in v] for k, v in au.items()}
mean = {k: float(np.mean(v)) for k, v in au.items()}; out["auc_mean"] = mean
ret = lambda h: (mean[h] - mean["kr"]) / (mean["drop_ros"] - mean["kr"])
out["retention_raw"] = {"pure": ret("pure"), "reg": ret("reg")}
r4 = {k: round(v, 4) for k, v in mean.items()}
out["retention_auc4"] = {h: (r4[h] - r4["kr"]) / (r4["drop_ros"] - r4["kr"]) for h in ("pure", "reg")}
# 부트스트랩: 원점 안 셀 복원추출 1000회. c5 순서 재현을 위해 rng 앞 6쌍 소비(형태는 원점별 n에만 의존)
B = 1000
def boot(A, Bm, rng):
    d_o, bs = [], np.zeros((9, B))
    for i, v in enumerate(ORIG):
        a = M[A][M[A].t0 == v].sort_values("cell"); b = M[Bm][M[Bm].t0 == v].sort_values("cell")
        y = a.yb.values; pa = a.p.values; pb = b.p.values; n = len(y)
        d_o.append(roc_auc_score(y, pa) - roc_auc_score(y, pb))
        I = rng.integers(0, n, (B, n))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            bs[i, j] = np.nan if yy.min() == yy.max() else roc_auc_score(yy, pa[ii]) - roc_auc_score(yy, pb[ii])
    d_o = np.array(d_o); bm = np.nanmean(bs, 0)
    return dict(d=float(d_o.mean()), ci=[float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))], wins=int((d_o > 0).sum()))
ns = [int((M["drop_ros"].t0 == v).sum()) for v in ORIG]
def skip(rng, k):
    for _ in range(k):
        for n in ns: rng.integers(0, n, (B, n))
for arm in ("pure",):
    rng = np.random.default_rng(20261005); skip(rng, 6)
    out[f"seq_{arm}"] = {"drop_ros-" + arm: boot("drop_ros", arm, rng), arm + "-kr": boot(arm, "kr", rng), "full-" + arm: boot("full", arm, rng)}
    rng2 = np.random.default_rng(777)
    out[f"seed777_{arm}"] = {"drop_ros-" + arm: boot("drop_ros", arm, rng2)}
json.dump(out, open(W + "/verify_5/v5_auc_boot.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "auc_by_origin"}, ensure_ascii=False, indent=1))
print(out["auc_by_origin"]["pure"])
