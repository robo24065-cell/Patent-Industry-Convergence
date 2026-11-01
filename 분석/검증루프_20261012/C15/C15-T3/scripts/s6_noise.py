# -*- coding: utf-8 -*-
r"""C15-T3 s6: 빌드 경로 잡음 폭(PREREG 변경 기록 1). 내 R8 라벨로 다시 빌드한 캐시(t3_pkr_r8)의 9원점 재학습 예측 vs c5 저장 예측.
캐시 비교(panel_ccy·feat2 차이), 원점별 AUC 차, paired()(seed 20261005), 같은 빌드 경로끼리 대안 A·B − 재빌드 R8.
쓰기: res/s6_noise.json"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy.stats import rankdata
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10"); R7 = os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "results")
ORIG = list(range(2010, 2019)); B = 1000


def auc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired(dfA, dfB, rng):
    aA, aB, bs = [], [], np.zeros((len(ORIG), B))
    for i, v in enumerate(ORIG):
        a = dfA[dfA.t0 == v].sort_values("cell"); b = dfB[dfB.t0 == v].sort_values("cell"); assert (a.cell.values == b.cell.values).all()
        y = a.yb.values; pa = a.p.values; pb = b.p.values; aA.append(auc(y, pa)); aB.append(auc(y, pb))
        I = rng.integers(0, len(y), (B, len(y)))
        for j in range(B):
            ii = I[j]; yy = y[ii]; bs[i, j] = np.nan if yy.sum() in (0, len(yy)) else auc(yy, pa[ii]) - auc(yy, pb[ii])
    d = np.array(aA) - np.array(aB); bm = np.nanmean(bs, 0)
    return dict(dAUC=float(d.mean()), ci=[float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))], wins=f"{int((d > 0).sum())}/9", by_origin=[round(float(x), 5) for x in d])


mg = lambda tag: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(W, "res", "t3", f"{tag}__*_preds.parquet")))], ignore_index=True)
M = {"stored": pd.read_parquet(os.path.join(C5, "t3_results", "hyb_pkr_preds.parquet")), "rebuilt_r8": mg("rebuilt_r8_hyb_pkr"),
     "reg": mg("hyb_pkr_reg"), "pure": mg("hyb_pkr_pure"),
     "drop_ros": pd.read_parquet(os.path.join(R7, "pub_drop_ros_preds.parquet")), "kr": pd.read_parquet(os.path.join(R7, "pub_kr_rebuild_preds.parquet"))}
R = {}
a = M["stored"].sort_values(["t0", "cell"]); b = M["rebuilt_r8"].sort_values(["t0", "cell"])
dp = np.abs(a.p.values - b.p.values)
R["pred_diff"] = dict(max_abs_dp=float(dp.max()), mean_abs_dp=float(dp.mean()), share_dp_gt_1e6=float((dp > 1e-6).mean()))
# 캐시 비교
c0 = os.path.join(C5, "t3cache_hyb_pkr"); c1 = os.path.join(W, "cache", "t3_pkr_r8"); cc = {}
for f, keys in (("panel_ccy.parquet", ["ksic", "mg", "cc", "year"]), ("feat2.parquet", ["ksic", "mg", "t0", "sub"])):
    x0 = pd.read_parquet(os.path.join(c0, f)); x1 = pd.read_parquet(os.path.join(c1, f))
    m = x0.merge(x1, on=keys, how="outer", suffixes=("_0", "_1"), indicator=True)
    num = [c for c in x0.columns if c not in keys]
    cc[f] = dict(rows=(len(x0), len(x1)), only_one_side=int((m._merge != "both").sum()),
                 max_abs_diff=max(float(np.nanmax(np.abs(m[c + "_0"].astype(float) - m[c + "_1"].astype(float)))) for c in num))
R["cache_diff"] = cc
au = {k: [auc(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG] for k, d in M.items()}
R["auc_mean"] = {k: float(np.mean(v)) for k, v in au.items()}
R["auc_by_origin_rebuilt_minus_stored"] = [round(x - y, 5) for x, y in zip(au["rebuilt_r8"], au["stored"])]
base = R["auc_mean"]["drop_ros"] - R["auc_mean"]["kr"]
R["retention_raw"] = {k: (R["auc_mean"][k] - R["auc_mean"]["kr"]) / base for k in ("stored", "rebuilt_r8", "reg", "pure")}
rng = np.random.default_rng(20261005)
R["paired"] = {"rebuilt_r8−stored": paired(M["rebuilt_r8"], M["stored"], rng), "reg−rebuilt_r8": paired(M["reg"], M["rebuilt_r8"], rng),
               "pure−rebuilt_r8": paired(M["pure"], M["rebuilt_r8"], rng)}
json.dump(R, open(os.path.join(W, "res", "s6_noise.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
