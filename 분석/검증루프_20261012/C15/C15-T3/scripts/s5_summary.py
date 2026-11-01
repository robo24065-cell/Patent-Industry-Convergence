# -*- coding: utf-8 -*-
r"""C15-T3 s5: hyb_pkr 비교 요약 — c5_R10 c5_t3_summary.py paired()(원점 안 셀 복원추출 1,000회, seed 20261005) 그대로, C10-T3 s6 세트 방식.
세트마다 rng 새로(seed 20261005), 호출 순서 c5와 같음: (drop_ros,txt) (txt,kr) (full,txt) (drop_ros,free) (free,kr) (full,free) (drop_ros,pkr) (pkr,kr) (full,pkr) (drop_ros,kr).
세트: stored(c5 저장 예측 전부, 재현) / repro(이 폴더 재실행 hyb_pkr + 저장 txt·free) / A(대안 A: C10 REG txt·free + hyb_pkr_reg) / B(대안 B: C10 REG txt·free + hyb_pkr_pure)
읽기만: R7 results/pub_*_preds, c5_R10 t3_results, c5_t3_summary.json, C10-T3 res/t3/hyb_{txt,free}_reg_preds, C15-T3 res/t3/*.
쓰기: res/s5_t3_summary.json, hyb_pkr_altA.csv(원점별)"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from scipy.stats import rankdata
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.abspath(os.path.join(W, "..", "..", "..", "모델경쟁_20260927"))
R7 = os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "results"); C5 = os.path.join(MC, "_shared_report", "fusion_advantage", "c5_R10")
C10 = os.path.abspath(os.path.join(W, "..", "..", "C10", "C10-T3"))
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
                    ci=[round(lo, 4), round(hi, 4)], ci_raw=[lo, hi], wins=f"{w}/9", by_origin=[round(float(x), 4) for x in d], verdict=verdict)
    return paired


def merged(tag):
    fs = sorted(glob.glob(os.path.join(W, "res", "t3", f"{tag}__*_preds.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    assert sorted(d.t0.unique()) == ORIG, (tag, sorted(d.t0.unique())); assert not d.duplicated(["t0", "cell"]).any()
    js = [json.load(open(f.replace("_preds.parquet", ".json"), encoding="utf-8")) for f in fs]
    byo = {k: v for j in js for k, v in j["by_origin"].items()}
    cov = js[0]["info"]["coverage_D"]; assert all(j["info"]["coverage_D"] == cov for j in js)
    return d, byo, cov, [j["info"].get("peak_ws_gb") for j in js]


M = {"full": pd.read_parquet(os.path.join(R7, "pub_full_preds.parquet")), "drop_ros": pd.read_parquet(os.path.join(R7, "pub_drop_ros_preds.parquet")),
     "kr_rebuild": pd.read_parquet(os.path.join(R7, "pub_kr_rebuild_preds.parquet"))}
for k in ["hyb_txt", "hyb_free", "hyb_pkr"]:
    M[f"{k}@stored"] = pd.read_parquet(os.path.join(C5, "t3_results", f"{k}_preds.parquet"))
for k in ["hyb_txt_reg", "hyb_free_reg"]:
    M[f"{k}@C10"] = pd.read_parquet(os.path.join(C10, "res", "t3", f"{k}_preds.parquet"))
BYO, COV, PEAK = {}, {}, {}
for tag in ["repro_hyb_pkr", "hyb_pkr_reg", "hyb_pkr_pure"]:
    M[tag], BYO[tag], COV[tag], PEAK[tag] = merged(tag)
R = {"peak_ws_gb": PEAK, "coverage_D": COV}
a = M["repro_hyb_pkr"].sort_values(["t0", "cell"]); b = M["hyb_pkr@stored"].sort_values(["t0", "cell"])
R["P0_repro_hyb_pkr"] = dict(same_rows=bool((a.cell.values == b.cell.values).all()), max_abs_dp=float(np.abs(a.p.values - b.p.values).max()),
                             coverage_same=COV["repro_hyb_pkr"] == json.load(open(os.path.join(C5, "c5_t3_summary.json"), encoding="utf-8"))["hyb_pkr_coverage_D"])
print("P0", R["P0_repro_hyb_pkr"])
R["auc5"] = {k: round(float(np.mean([auc(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG])), 5) for k, d in M.items()}
R["auc_raw"] = {k: float(np.mean([auc(d[d.t0 == v].yb, d[d.t0 == v].p) for v in ORIG])) for k, d in M.items()}
SETS = {"stored": ("hyb_txt@stored", "hyb_free@stored", "hyb_pkr@stored"), "repro": ("hyb_txt@stored", "hyb_free@stored", "repro_hyb_pkr"),
        "A": ("hyb_txt_reg@C10", "hyb_free_reg@C10", "hyb_pkr_reg"), "B": ("hyb_txt_reg@C10", "hyb_free_reg@C10", "hyb_pkr_pure")}
c5 = json.load(open(os.path.join(C5, "c5_t3_summary.json"), encoding="utf-8"))
ra = R["auc_raw"]; r4 = {k: round(v, 4) for k, v in ra.items()}
for nm, (ht, hf, hp) in SETS.items():
    paired = make_paired(np.random.default_rng(20261005))
    order = [("drop_ros", ht), (ht, "kr_rebuild"), ("full", ht), ("drop_ros", hf), (hf, "kr_rebuild"), ("full", hf),
             ("drop_ros", hp), (hp, "kr_rebuild"), ("full", hp), ("drop_ros", "kr_rebuild")]
    P = {f"{x}−{y}": paired(M[x], M[y]) for x, y in order}
    ret = lambda h, r: (r[h] - r["kr_rebuild"]) / (r["drop_ros"] - r["kr_rebuild"])
    P["retention_raw"] = {"txt": ret(ht, ra), "free": ret(hf, ra), "pkr": ret(hp, ra)}
    P["retention_auc4"] = {"txt": ret(ht, r4), "free": ret(hf, r4), "pkr": ret(hp, r4)}
    R[nm] = P
# 재현 점검: stored·repro 세트의 pkr 쌍이 c5_t3_summary.json과 같은가
rep = {}
for s in ("stored", "repro"):
    hp = SETS[s][2]
    for mine, ck in [(f"drop_ros−{hp}", "drop_ros−hyb_pkr"), (f"{hp}−kr_rebuild", "hyb_pkr−kr_rebuild"), (f"full−{hp}", "full−hyb_pkr"),
                     (f"drop_ros−{SETS[s][0]}", "drop_ros−hyb_txt"), ("drop_ros−kr_rebuild", "drop_ros−kr_rebuild")]:
        m, o = R[s][mine], c5["pairs"][ck]
        rep[f"{s}:{ck}"] = dict(mine=(m["dAUC"], m["ci"], m["wins"]), c5=(o["dAUC"], o["ci"], o["wins"]), same=(m["dAUC"], m["ci"], m["wins"]) == (o["dAUC"], o["ci"], o["wins"]))
    rep[f"{s}:retention_auc4_pkr"] = dict(mine=round(R[s]["retention_auc4"]["pkr"], 3), c5=c5["retention"]["hyb_pkr"], same=round(R[s]["retention_auc4"]["pkr"], 3) == c5["retention"]["hyb_pkr"])
R["repro_vs_c5"] = rep; R["repro_all_same"] = all(v["same"] for v in rep.values())
print("재현", R["repro_all_same"], {k: v for k, v in rep.items() if not v["same"]})
# 수정판 − 저장판(같은 평가 셀)
paired = make_paired(np.random.default_rng(20261005))
R["fix_minus_stored"] = {f"{x}−{y}": paired(M[x], M[y]) for x, y in [("hyb_pkr_reg", "hyb_pkr@stored"), ("hyb_pkr_pure", "hyb_pkr@stored"), ("hyb_pkr_reg", "hyb_pkr_pure")]}
json.dump(R, open(os.path.join(W, "res", "s5_t3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 원점별 표
rows = []
for tag, arm in [("repro_hyb_pkr", "R8(재현)"), ("hyb_pkr_reg", "대안A_REG"), ("hyb_pkr_pure", "대안B_상속없음")]:
    for v in ORIG:
        o = BYO[tag][str(v)]; st = M["hyb_pkr@stored"]; s = st[st.t0 == v]
        dr = M["drop_ros"][M["drop_ros"].t0 == v]; kr = M["kr_rebuild"][M["kr_rebuild"].t0 == v]
        rows.append(dict(arm=arm, t0=v, n=o["model"]["n"], n_evt=o["model"]["n_evt"], auc=round(o["model"]["auc"], 5), base_auc=round(o["baseline"]["auc"], 5),
                         skill=round(o["skill"], 5), p_at10=round(o["model"]["p_at10"], 4), auc_stored_R8=round(auc(s.yb, s.p), 5),
                         auc_drop_ros=round(auc(dr.yb, dr.p), 5), auc_kr_only=round(auc(kr.yb, kr.p), 5),
                         d_vs_stored=round(o["model"]["auc"] - auc(s.yb, s.p), 5)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(W, "hyb_pkr_altA.csv"), index=False, encoding="utf-8-sig")
for nm in SETS:
    P = R[nm]; hp = SETS[nm][2]
    print(nm, {k: (v["dAUC"], v["ci"], v["wins"], v["verdict"]) for k, v in P.items() if "retention" not in k and ("pkr" in k)},
          {k: round(v, 4) for k, v in P["retention_raw"].items()}, {k: round(v, 4) for k, v in P["retention_auc4"].items()})
print({k: (v["dAUC"], v["ci"], v["wins"], v["verdict"]) for k, v in R["fix_minus_stored"].items()})
print("AUC", {k: v for k, v in R["auc5"].items()})
print(T.groupby("arm")[["auc", "skill"]].mean().round(5))
