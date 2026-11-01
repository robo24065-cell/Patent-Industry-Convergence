# -*- coding: utf-8 -*-
r"""c5_R10 c5: T3 하이브리드 비교 요약. R8 s04 paired()(원점 안 셀 복원추출 1,000회, 9원점 평균 ΔAUC CI) 그대로.
기준점(읽기만): R7 results/pub_drop_ros_preds.parquet(제공, RoS 제외), pub_kr_rebuild_preds.parquet(KR만), pub_full_preds.parquet(참고)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
R7 = os.path.join(MC, r"T3_기술셀부상예측\ESSENTIAL_R7\results"); RD = os.path.join(HERE, "t3_results")
B = 1000; rng = np.random.default_rng(20261005); ORIG = list(range(2010, 2019))


def auc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired(dfA, dfB):
    aA, aB, bs = [], [], np.zeros((len(ORIG), B))
    for i, v in enumerate(ORIG):
        a = dfA[dfA.t0 == v].sort_values("cell"); b = dfB[dfB.t0 == v].sort_values("cell")
        assert (a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all()
        y = a.yb.values; pa = a.p.values; pb = b.p.values
        aA.append(auc(y, pa)); aB.append(auc(y, pb))
        I = rng.integers(0, len(y), (B, len(y)))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            bs[i, j] = np.nan if yy.sum() in (0, len(yy)) else auc(yy, pa[ii]) - auc(yy, pb[ii])
    d = np.array(aA) - np.array(aB); bm = np.nanmean(bs, 0)
    lo, hi = float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5)); w = int((d > 0).sum())
    verdict = "A 우위" if (d.mean() > 0 and lo > 0 and w >= 6) else ("B 우위" if (d.mean() < 0 and hi < 0 and (9 - w) >= 6) else "같다")
    return dict(aucA=round(float(np.mean(aA)), 4), aucB=round(float(np.mean(aB)), 4), dAUC=round(float(d.mean()), 4), ci=[round(lo, 4), round(hi, 4)],
                wins=f"{w}/9", by_origin=[round(float(x), 4) for x in d], verdict=verdict)


M = {"full": pd.read_parquet(os.path.join(R7, "pub_full_preds.parquet")), "drop_ros": pd.read_parquet(os.path.join(R7, "pub_drop_ros_preds.parquet")),
     "kr_rebuild": pd.read_parquet(os.path.join(R7, "pub_kr_rebuild_preds.parquet"))}
for k in ["kr_repro", "hyb_txt", "hyb_free", "hyb_pkr"]:
    f = os.path.join(RD, f"{k}_preds.parquet")
    if os.path.exists(f): M[k] = pd.read_parquet(f)
R = {}
if "kr_repro" in M:
    a = M["kr_repro"].sort_values(["t0", "cell"]); b = M["kr_rebuild"].sort_values(["t0", "cell"])
    R["repro_kr_rebuild"] = dict(same_rows=bool((a.cell.values == b.cell.values).all()), max_abs_dp=float(np.abs(a.p.values - b.p.values).max()))
    print("재현", R["repro_kr_rebuild"])
ref_kr = "kr_rebuild"
if "repro_kr_rebuild" in R and R["repro_kr_rebuild"]["max_abs_dp"] >= 1e-6:
    ref_kr = "kr_repro"; R["note"] = "재현 불일치 → 이 폴더 재실행값(kr_repro)을 KR만 기준으로 사용"
R["auc"] = {k: round(float(np.mean([auc(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG])), 4) for k, d in M.items()}
pairs = []
for h in ["hyb_txt", "hyb_free", "hyb_pkr"]:
    if h in M: pairs += [("drop_ros", h), (h, ref_kr), ("full", h)]
pairs += [("drop_ros", ref_kr)]
R["pairs"] = {f"{a}−{b}": paired(M[a], M[b]) for a, b in pairs}
base = R["auc"]["drop_ros"] - R["auc"][ref_kr]
R["retention"] = {h: round((R["auc"][h] - R["auc"][ref_kr]) / base, 3) for h in ["hyb_txt", "hyb_free", "hyb_pkr"] if h in M}
for h in ["hyb_txt", "hyb_free", "hyb_pkr"]:
    f = os.path.join(RD, f"{h}.json")
    if os.path.exists(f): R[f"{h}_coverage_D"] = json.load(open(f, encoding="utf-8"))["info"]["coverage_D"]
json.dump(R, open(os.path.join(HERE, "c5_t3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
