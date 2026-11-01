# -*- coding: utf-8 -*-
r"""C10-T3 s6: T3 하이브리드 비교 요약 — c5_R10 c5_t3_summary.py paired() 그대로(원점 안 셀 복원추출 1,000회, seed 20261005, 같은 호출 순서).
세트: stored(c5 저장 예측, 재현 점검) / repro(이 폴더 재실행 hyb_free) / reg(hyb_txt_reg, hyb_free_reg) / pub(hyb_txt_pub, hyb_free_pub)
읽기만: R7 results/pub_*_preds, c5_R10 t3_results/*_preds, C10-T3 res/t3/*_preds. 쓰기: res/s6_t3_summary.json"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
R7 = os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "results"); C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10")
B = 1000; ORIG = list(range(2010, 2019))


def auc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def make_paired(rng):
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
        return dict(aucA=round(float(np.mean(aA)), 4), aucB=round(float(np.mean(aB)), 4), dAUC=round(float(d.mean()), 4), dAUC_raw=float(d.mean()),
                    ci=[round(lo, 4), round(hi, 4)], wins=f"{w}/9", by_origin=[round(float(x), 4) for x in d], verdict=verdict)
    return paired


M = {"full": pd.read_parquet(os.path.join(R7, "pub_full_preds.parquet")), "drop_ros": pd.read_parquet(os.path.join(R7, "pub_drop_ros_preds.parquet")),
     "kr_rebuild": pd.read_parquet(os.path.join(R7, "pub_kr_rebuild_preds.parquet"))}
for k in ["hyb_txt", "hyb_free", "hyb_pkr"]:
    M[f"{k}@stored"] = pd.read_parquet(os.path.join(C5, "t3_results", f"{k}_preds.parquet"))
for k in ["repro_hyb_free", "hyb_free_reg", "hyb_txt_reg", "hyb_free_pub", "hyb_txt_pub"]:
    f = os.path.join(W, "res", "t3", f"{k}_preds.parquet")
    if os.path.exists(f): M[k] = pd.read_parquet(f)
R = {}
a = M["repro_hyb_free"].sort_values(["t0", "cell"]); b = M["hyb_free@stored"].sort_values(["t0", "cell"])
R["repro_hyb_free"] = dict(same_rows=bool((a.cell.values == b.cell.values).all()), max_abs_dp=float(np.abs(a.p.values - b.p.values).max()))
print("재현", R["repro_hyb_free"])
R["auc"] = {k: round(float(np.mean([auc(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG])), 5) for k, d in M.items()}
SETS = {"stored": ("hyb_txt@stored", "hyb_free@stored"), "reg": ("hyb_txt_reg", "hyb_free_reg"), "pub": ("hyb_txt_pub", "hyb_free_pub")}
c5 = json.load(open(os.path.join(C5, "c5_t3_summary.json"), encoding="utf-8"))
for nm, (ht, hf) in SETS.items():
    paired = make_paired(np.random.default_rng(20261005))
    order = [("drop_ros", ht), (ht, "kr_rebuild"), ("full", ht), ("drop_ros", hf), (hf, "kr_rebuild"), ("full", hf),
             ("drop_ros", "hyb_pkr@stored"), ("hyb_pkr@stored", "kr_rebuild"), ("full", "hyb_pkr@stored"), ("drop_ros", "kr_rebuild")]
    P = {f"{x}−{y}": paired(M[x], M[y]) for x, y in order}
    base = R["auc"]["drop_ros"] - R["auc"]["kr_rebuild"]
    P["retention"] = {"txt": round((R["auc"][ht] - R["auc"]["kr_rebuild"]) / base, 3), "free": round((R["auc"][hf] - R["auc"]["kr_rebuild"]) / base, 3)}
    R[nm] = P
# 재현 점검: stored 세트가 c5_t3_summary.json과 같은가
rep = {}
for k_mine, k_c5 in [("drop_ros−hyb_txt@stored", "drop_ros−hyb_txt"), ("drop_ros−hyb_free@stored", "drop_ros−hyb_free"),
                     ("hyb_txt@stored−kr_rebuild", "hyb_txt−kr_rebuild"), ("hyb_free@stored−kr_rebuild", "hyb_free−kr_rebuild"), ("drop_ros−kr_rebuild", "drop_ros−kr_rebuild")]:
    m, o = R["stored"][k_mine], c5["pairs"][k_c5]
    rep[k_c5] = dict(mine=(m["dAUC"], m["ci"], m["wins"]), c5=(o["dAUC"], o["ci"], o["wins"]), same=(m["dAUC"], m["ci"], m["wins"]) == (o["dAUC"], o["ci"], o["wins"]))
R["repro_vs_c5"] = rep
# 수정판 − 저장판 직접 비교(같은 평가 셀)
paired = make_paired(np.random.default_rng(20261005))
R["fix_minus_stored"] = {f"{x}−{y}": paired(M[x], M[y]) for x, y in [("hyb_free_reg", "hyb_free@stored"), ("hyb_txt_reg", "hyb_txt@stored"),
                                                                    ("hyb_free_pub", "hyb_free@stored"), ("hyb_txt_pub", "hyb_txt@stored")] if x in M}
json.dump(R, open(os.path.join(W, "res", "s6_t3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for nm in SETS:
    print(nm, {k: (v["dAUC"], v["ci"], v["wins"], v["verdict"]) for k, v in R[nm].items() if k != "retention" and "pkr" not in k and "full" not in k}, R[nm]["retention"])
print({k: (v["dAUC"], v["ci"], v["wins"]) for k, v in R["fix_minus_stored"].items()})
print("repro", {k: v["same"] for k, v in rep.items()})
