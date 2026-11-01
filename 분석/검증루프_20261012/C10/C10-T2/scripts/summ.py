# -*- coding: utf-8 -*-
r"""집계: [0] 재현, [1] seed 변동, [2][3] 군집 부트스트랩, 사전 판정 → res/step0.json, seed_configs.csv, seed_summary.json,
arms_official.csv, boot_summary.csv, verdict.json"""
import sys, glob
sys.dont_write_bytecode = True
from c10_common import *
from b_lib import AucPrep

R0 = {}
# ---------- [0] 재현
S = pd.read_csv(os.path.join(SRC_RES, "C_summary.csv"))
rep = []
for c in C_ORIG:
    f = os.path.join(RES, "auc", f"auc_c{c}_k0_P50-FREE50-TXT50_p012.json")
    r = json.load(open(f, encoding="utf-8"))
    for arm in ("P50", "FREE50", "TXT50"):
        o = S[(S.c == c) & (S.arm == arm)].iloc[0]; m = r["res"][f"{arm}_p0"]
        rep.append({"c": c, "arm": arm, "d_auc": m["auc"] - o.auc, "d_far": m["auc_far"] - o.auc_far, "d_top": m["top1000"] - o.top1000})
    # 저장 예측 = 공식 C_pred (float32 완전 일치?)
    off = pd.read_parquet(os.path.join(SRC_RES, f"C_pred_c{c}.parquet"), columns=["P50", "FREE50", "TXT50"])
    mine = pd.read_parquet(os.path.join(CACHE, f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0", "TXT50_p0"])
    R0[f"pred_equal_c{c}"] = {a: float(np.max(np.abs(off[a].values - mine[f"{a}_p0"].values))) for a in ("P50", "FREE50", "TXT50")}
rep = pd.DataFrame(rep); R0["max_abs_auc_diff"] = float(rep[["d_auc", "d_far", "d_top"]].abs().max().max()); R0["pass_auc"] = R0["max_abs_auc_diff"] <= 1e-6
seed = {}
for f in glob.glob(os.path.join(CACHE, "seed_k*_p*.npz")):
    z = np.load(f); k = int(os.path.basename(f).split("_")[1][1:])
    for q, p in enumerate(z["pairs"]):
        seed[(k, int(p))] = (z["pt"][:, q, :], z["bs"][:, :, q, :])
CP = {}
for a, b, ia, ib in (("P50", "FREE50", 0, 1), ("P50", "TXT50", 0, 2)):
    pt, bs = seed[(0, 0)]
    d200 = (bs[:200, ia, :] - bs[:200, ib, :]).mean(1)
    j = json.load(open(os.path.join(SRC_RES, f"C_pool_{a}_{b}.json"), encoding="utf-8"))
    mine = {"mean_d": float((pt[ia] - pt[ib]).mean()), "ci95": [float(np.percentile(d200, 2.5)), float(np.percentile(d200, 97.5))],
            "p_gt0": float((d200 > 0).mean()), "per_origin_d": list(map(float, pt[ia] - pt[ib]))}
    CP[f"{a}-{b}"] = {"mine": mine, "official": {k: j[k] for k in ("mean_d", "ci95", "p_gt0", "per_origin_d")},
                      "max_abs_diff": float(max(abs(mine["mean_d"] - j["mean_d"]), abs(mine["ci95"][0] - j["ci95"][0]), abs(mine["ci95"][1] - j["ci95"][1]),
                                                max(abs(x - y) for x, y in zip(mine["per_origin_d"], j["per_origin_d"]))))}
R0["pool"] = CP; R0["pass_pool"] = all(v["max_abs_diff"] <= 1e-6 for v in CP.values())
R0["auc_rows"] = rep.to_dict("records")
json.dump(R0, open(os.path.join(RES, "step0.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("step0", R0["max_abs_auc_diff"], R0["pass_pool"], {k: v["max_abs_diff"] for k, v in CP.items()})

# ---------- [1] seed 변동
rows = []
for (k, p), (pt, bs) in sorted(seed.items()):
    for a, b, ia, ib in (("P50", "FREE50", 0, 1), ("P50", "TXT50", 0, 2)):
        dd = pt[ia] - pt[ib]; db = (bs[:, ia, :] - bs[:, ib, :]).mean(1)
        rows.append({"pair": f"{a}-{b}", "k_neg": k, "train_pair": p, "seeds": f"{2*p},{2*p+1}", "neg_seed": f"{(k+1)*1000}+t",
                     "mean_d": dd.mean(), "ci_lo": np.percentile(db, 2.5), "ci_hi": np.percentile(db, 97.5), "p_gt0": (db > 0).mean(),
                     "wins": int((dd > 0).sum()), "per_origin": " ".join(f"{x:+.4f}" for x in dd),
                     "auc_a": pt[ia].mean(), "auc_b": pt[ib].mean()})
SC = pd.DataFrame(rows); SC.to_csv(os.path.join(RES, "seed_configs.csv"), index=False, encoding="utf-8-sig")
SS = {}
for pr, g in SC.groupby("pair"):
    SS[pr] = {"n_configs": len(g), "mean": g.mean_d.mean(), "sd": g.mean_d.std(ddof=1), "min": g.mean_d.min(), "max": g.mean_d.max(),
              "share_ci_lo_gt0": float((g.ci_lo > 0).mean()), "n_ci_lo_gt0": int((g.ci_lo > 0).sum()),
              "share_point_gt0": float((g.mean_d > 0).mean()), "wins_dist": g.wins.value_counts().sort_index().to_dict(),
              "sd_within_neg(train seed)": float(g.groupby("k_neg").mean_d.std(ddof=1).mean()),
              "sd_between_neg(means)": float(g.groupby("k_neg").mean_d.mean().std(ddof=1)),
              "median_boot_halfwidth": float(((g.ci_hi - g.ci_lo) / 2).median())}
# 앙상블: 30모형×3표집 평균 점수
ens = {}
for c in C_ORIG:
    a = pd.read_parquet(os.path.join(NC8, "cache", f"A_o{c}.parquet"), columns=["first_ky"]); y = a.first_ky.notna().values.astype(np.int8)
    acc = {}
    for f in glob.glob(os.path.join(CACHE, f"pred_c{c}_k*_P50-FREE50-TXT50_p*.parquet")):
        d = pd.read_parquet(f)
        kk = os.path.basename(f).split("_")[2]
        for col in d.columns:
            acc[(kk, col)] = d[col].values.astype(np.float64)
    for arm in ("P50", "FREE50", "TXT50"):
        cols = [v for (kk, col), v in acc.items() if col.startswith(arm + "_")]
        ens.setdefault(arm, {})[c] = auc_w(y, np.mean(cols, axis=0)); ens.setdefault(arm + "_n", {})[c] = len(cols)
for a, b in (("P50", "FREE50"), ("P50", "TXT50")):
    dd = np.array([ens[a][c] - ens[b][c] for c in C_ORIG])
    SS[f"{a}-{b}"]["ensemble"] = {"mean_d": float(dd.mean()), "per_origin": list(map(float, dd)), "n_models_per_arm": ens[a + "_n"]}
json.dump(SS, open(os.path.join(RES, "seed_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("seed", json.dumps(SS, ensure_ascii=False, default=float)[:1500])

# ---------- [2][3] 공식 seed 팔 · 군집 부트스트랩
ZTAGS = ["FREEIPC50-HSC1_50-P50HSC1", "HMG3_50-P50FIPC-P50HMG3"]
AO = []
for c in C_ORIG:
    r1 = json.load(open(os.path.join(RES, "auc", f"auc_c{c}_k0_P50-FREE50-TXT50_p012.json"), encoding="utf-8"))["res"]
    r2 = {}
    for zt in ZTAGS:
        r2.update(json.load(open(os.path.join(RES, "auc", f"auc_c{c}_k0_{zt}_p0.json"), encoding="utf-8"))["res"])
    for kk, v in list(r1.items()) + list(r2.items()):
        if kk.endswith("_p0"):
            AO.append({"c": c, "arm": kk[:-3], "auc": v["auc"], "auc_far": v["auc_far"], "top1000": v["top1000"]})
AO = pd.DataFrame(AO); AO.to_csv(os.path.join(RES, "arms_official.csv"), index=False, encoding="utf-8-sig")
MEAN = AO.groupby("arm")[["auc", "auc_far", "top1000"]].mean()
PAIRS = [("P50", "FREE50"), ("P50", "TXT50"), ("P50", "FREEIPC50"), ("P50", "HSC1_50"), ("P50", "HMG3_50"),
         ("P50FIPC", "P50"), ("P50FIPC", "FREEIPC50"), ("P50HSC1", "P50"), ("P50HSC1", "HSC1_50"), ("P50HMG3", "P50"), ("P50HMG3", "HMG3_50"),
         ("FREEIPC50", "FREE50"), ("HSC1_50", "FREE50"), ("HSC1_50", "FREEIPC50")]
BT = {}
for f in glob.glob(os.path.join(CACHE, "boot_*.npz")):
    z = np.load(f); clt = "_".join(os.path.basename(f)[5:-4].split("_")[:-2])
    BT.setdefault(clt, []).append((int(os.path.basename(f)[:-4].split("_")[-2]), z["auc"], list(z["arms"]), z["n_clusters"]))
rows = []
piv = AO.pivot(index="c", columns="arm", values="auc")
for a, b in PAIRS:
    dd = (piv[a] - piv[b]).values
    base = {"pair": f"{a}-{b}", "mean_d": dd.mean(), "wins": int((dd > 0).sum()), "per_origin": " ".join(f"{x:+.4f}" for x in dd),
            "d_far": MEAN.loc[a, "auc_far"] - MEAN.loc[b, "auc_far"], "d_top1000": MEAN.loc[a, "top1000"] - MEAN.loc[b, "top1000"]}
    for clt in ("firm", "ksic_t", "ksic_f", "sc", "firm_sc", "firm_ksic_t"):
        if clt not in BT:
            continue
        parts = sorted(BT[clt], key=lambda x: x[0]); arms = parts[0][2]
        A = np.concatenate([p[1] for p in parts]); ia, ib = arms.index(a), arms.index(b)
        db = (A[:, ia, :] - A[:, ib, :]).mean(1)
        base[f"{clt}_lo"] = np.percentile(db, 2.5); base[f"{clt}_hi"] = np.percentile(db, 97.5); base[f"{clt}_B"] = len(db)
        base[f"{clt}_ncl"] = "x".join(map(str, parts[0][3]))
    rows.append(base)
BS_ = pd.DataFrame(rows); BS_.to_csv(os.path.join(RES, "boot_summary.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(BS_.round(5).to_string())
print(MEAN.round(5).to_string())

# ---------- 사전 판정
V = {}
CL = ["firm", "ksic_t", "ksic_f", "sc", "firm_sc", "firm_ksic_t"]
def row(p): return BS_[BS_.pair == p].iloc[0]
for p in ("P50-FREE50", "P50-TXT50"):
    r = row(p); s = SS[p]
    cl_ok = {c: bool(r[f"{c}_lo"] > 0) for c in CL if f"{c}_lo" in r}
    seed_ok = s["share_ci_lo_gt0"] >= 0.8
    if s["mean"] <= 0: v = "소멸"
    elif seed_ok and all(cl_ok.values()) and len(cl_ok) == 6: v = "유지"
    elif r.mean_d > 0: v = "약화(단서 필요)"
    else: v = "약화(단서 필요) — 공식 점추정 ≤ 0"
    V[p] = {"verdict": v, "seed_share_ci_lo_gt0": s["share_ci_lo_gt0"], "cluster_lo_gt0": cl_ok, "seed_mean": s["mean"], "official": r.mean_d}
for x in ("FREEIPC50", "HSC1_50"):
    r = row(f"P50-{x}")
    cl_ok = {c: bool(r[f"{c}_lo"] > 0) for c in CL if f"{c}_lo" in r}
    if r.mean_d <= 0 or r.firm_lo <= 0: v = "근거 B를 IPC 최빈표 계열 대비로는 주장 불가"
    elif not all(cl_ok.values()): v = "약화(IPC 최빈표 대비 단서 필요)"
    elif r.wins >= 3: v = "유지(IPC 최빈표 대비)"
    else: v = "약화(IPC 최빈표 대비 단서 필요) — 우세 원점 <3/5"
    V[f"P50-{x}"] = {"verdict": v, "official": r.mean_d, "wins": int(r.wins), "cluster_lo_gt0": cl_ok}
json.dump(V, open(os.path.join(RES, "verdict.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("verdict", json.dumps(V, ensure_ascii=False, default=float))
