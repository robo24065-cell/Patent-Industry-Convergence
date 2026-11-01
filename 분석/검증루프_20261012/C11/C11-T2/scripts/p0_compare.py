# -*- coding: utf-8 -*-
r"""C11-T2 P0: r8(결함 재현) 팔 산출물이 저장 R9 산출물과 같은지 대조.
사용: python p0_compare.py panel | ctx | preds | ext
쓰기: res/p0_<what>.json"""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
R9 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측/TXT_R9"; W = BASE + "/분석/검증루프_20261012/C11/C11-T2"
what = sys.argv[1]; R = {}
if what == "panel":
    for f in ("panel_ccy.parquet", "feat2.parquet"):
        a = pd.read_parquet(f"{R9}/cache_txt_pub/{f}"); b = pd.read_parquet(f"{W}/cache/txt_r8/{f}")
        keys = [c for c in a.columns if a[c].dtype == object] + [c for c in ("year", "t0") if c in a.columns]
        a = a.sort_values(keys).reset_index(drop=True); b = b[a.columns].sort_values(keys).reset_index(drop=True)
        r = dict(shape_a=list(a.shape), shape_b=list(b.shape), same_keys=bool(a.shape == b.shape and (a[keys].values == b[keys].values).all()))
        if r["same_keys"]:
            num = [c for c in a.columns if c not in keys]
            r["max_abs_diff"] = {c: float(np.nanmax(np.abs(a[c].astype(float).values - b[c].astype(float).values))) if len(a) else 0.0 for c in num}
            r["nan_mismatch"] = int(sum(int((a[c].isna() != b[c].isna()).sum()) for c in num))
        R[f] = r
    bs = pd.read_csv(f"{R9}/cache_txt_pub/base_stats.csv"); bm = pd.read_csv(f"{W}/cache/txt_r8/base_stats.csv")
    R["base_stats_equal"] = bool((bs.values == bm.values).all())
elif what == "ctx":
    a = pd.read_parquet(f"{R9}/ctx_txt.parquet");
    for arm in ("stored", "r8"):
        fn = f"{W}/cache/ctx_txt_{arm}.parquet"
        if not os.path.exists(fn): continue
        b = pd.read_parquet(fn)
        a1 = a.sort_values(["mg", "t0"]).reset_index(drop=True); b1 = b[a.columns].sort_values(["mg", "t0"]).reset_index(drop=True)
        same = bool(a1.shape == b1.shape and (a1.mg.values == b1.mg.values).all() and (a1.t0.values == b1.t0.values).all())
        R[arm] = dict(same_keys=same, max_abs={c: float(np.nanmax(np.abs(a1[c].values - b1[c].values))) for c in a.columns if c.startswith("c_")} if same else None,
                      nan_mismatch=int(sum(int((a1[c].isna() != b1[c].isna()).sum()) for c in a.columns if c.startswith("c_"))) if same else None)
elif what == "preds":
    for tag in ("mg_ctx-txt", "txt50_task"):
        fn = f"{W}/res/runs/{tag}__r8_preds.parquet"
        if not os.path.exists(fn): continue
        a = pd.read_parquet(f"{R9}/results/{tag}_preds.parquet").sort_values(["t0", "cell"]).reset_index(drop=True)
        b = pd.read_parquet(fn).sort_values(["t0", "cell"]).reset_index(drop=True)
        same = bool(a.shape == b.shape and (a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all())
        R[tag] = dict(same_rows=same, max_abs_dp=float(np.abs(a.p.values - b.p.values).max()) if same else None,
                      max_abs_dp_base=float(np.abs(a.p_base.values - b.p_base.values).max()) if same else None)
elif what == "ext":
    a = pd.read_csv(f"{R9}/external_alignment_r9.csv", encoding="utf-8-sig")
    for arm in ("stored", "r8"):
        for fn in sorted(os.listdir(f"{W}/res/s6")):
            if not fn.startswith(f"s6_external__{arm}__"): continue
            b = pd.read_csv(f"{W}/res/s6/{fn}", encoding="utf-8-sig")
            m = a.merge(b, on=["비교", "외부지표", "신호"], suffixes=("_r9", "_c11"))
            cols = ["A_rho", "B_rho", "차이", "CI_lo", "CI_hi", "A우세_원점", "p_boot", "q_BH"]
            R[fn] = dict(n=len(m), max_abs={c: float((m[c + "_r9"] - m[c + "_c11"]).abs().max()) for c in cols}, verdict_same=bool((m["판정_r9"] == m["판정_c11"]).all()))
    a = pd.read_csv(f"{R9}/external_within_group_r9.csv", encoding="utf-8-sig")
    for arm in ("stored", "r8"):
        fn = f"{W}/res/s6/s6_within__{arm}.csv"
        if os.path.exists(fn):
            b = pd.read_csv(fn, encoding="utf-8-sig"); m = a.merge(b, on=["외부지표", "신호"], suffixes=("_r9", "_c11"))
            R[f"within_{arm}"] = dict(n=len(m), max_abs={c: float((m[c + "_r9"] - m[c + "_c11"]).abs().max()) for c in ["제공_r", "TXT_r", "차이", "CI_lo", "CI_hi", "제공우세_원점"]})
json.dump(R, open(f"{W}/res/p0_{what}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
