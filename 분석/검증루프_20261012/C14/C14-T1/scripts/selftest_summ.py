# -*- coding: utf-8 -*-
r"""자기 검증 집계 → selftest.json(폴더 루트)"""
import sys, glob
sys.dont_write_bytecode = True
from lib14 import *

out = {}
for S in ("S1", "S2"):
    fs = sorted(glob.glob(os.path.join(CACHE, f"st_{S}_*.npz")), key=lambda f: int(os.path.basename(f).split("_")[2]))
    Z = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
    n = len(d["pt"]); truth = float(d["pt"].mean()); true_sd = float(d["pt"].std(ddof=1))
    r = {"reps": n, "truth(MC mean Δ)": truth, "true_sd": true_sd, "mc_se_of_truth": true_sd / np.sqrt(n),
         "prev_mean": float(d["prev"].mean()), "n_rows_mean": float(d["n"].mean()),
         "vAB_share_mean": float(np.mean(d["vAB"] / (d["vA"] + d["vB"]))), "est": {}}
    for e in ("E1", "E2", "E3", "E4", "E4b"):
        cov = float(np.mean((d[f"{e}_lo"] <= truth) & (truth <= d[f"{e}_hi"])))
        se = d[f"{e}_se"]
        r["est"][e] = {"coverage": cov, "mean_se/true_sd": float(se.mean() / true_sd), "rms_se/true_sd": float(np.sqrt((se ** 2).mean()) / true_sd),
                       "pass_0.93_0.97": bool(0.93 <= cov <= 0.97),
                       "fix_rate": float(d[f"{e}_fix"].mean()) if f"{e}_fix" in d else 0.0}
    r["est"]["E1lin(참고)"] = {"mean_se/true_sd": float(d["E1lin_se"].mean() / true_sd)}
    out[S] = r
    log(S, n, {e: (v.get("coverage"), round(v["mean_se/true_sd"], 3)) for e, v in r["est"].items()})
out["pass_both"] = {e: out["S1"]["est"][e]["pass_0.93_0.97"] and out["S2"]["est"][e]["pass_0.93_0.97"] for e in ("E1", "E2", "E3", "E4", "E4b")}
out["mc_note"] = "반복 수 n에서 포함률 표준오차 ≈ sqrt(0.95·0.05/n)."
json.dump(out, open(safe_path(os.path.join(OUT, "selftest.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log(out["pass_both"])
