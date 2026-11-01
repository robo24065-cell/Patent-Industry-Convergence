# -*- coding: utf-8 -*-
"""C15-T3 verify_4: C-T3-COVD 중앙값 비 독립 재계산 (r3lib/r3core 미사용, pandas 직접).
목표 행 = R3 cache_pub feat_{2005..2018} 중 tot_W>=50 (r3core.add_labels와 같은 필터).
alt_totW = 대체 캐시 panel_ccy에서 같은 cell, 5개국 n 합, 연도 t0-2..t0 합(없으면 0).
비 = median((alt+1)/(tot+1))."""
import os, sys, json
import numpy as np, pandas as pd
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R3 = os.path.join(B, "모델경쟁_20260927", "T3_기술셀부상예측", "R3", "cache_pub")
C5 = os.path.join(B, "모델경쟁_20260927", "_shared_report", "fusion_advantage", "c5_R10")
C10 = os.path.join(B, "검증루프_20261012", "C10", "C10-T3", "cache")
C15 = os.path.join(B, "검증루프_20261012", "C15", "C15-T3", "cache")
T = pd.concat([pd.read_parquet(os.path.join(R3, f"feat_{t}.parquet"), columns=["cell", "t0", "tot_W"]) for t in range(2005, 2019)], ignore_index=True)
T = T[T.tot_W >= 50].reset_index(drop=True)
out = {"n_target_rows": int(len(T))}
caches = {"c5_R8_txt": os.path.join(C5, "t3cache_hyb_txt"), "c5_R8_free": os.path.join(C5, "t3cache_hyb_free"), "c5_R8_pkr": os.path.join(C5, "t3cache_hyb_pkr"),
          "C10_txt_reg": os.path.join(C10, "t3_txt_reg"), "C10_free_reg": os.path.join(C10, "t3_free_reg"),
          "C10_txt_pub": os.path.join(C10, "t3_txt_pub"), "C10_free_pub": os.path.join(C10, "t3_free_pub")}
for d in sorted(os.listdir(C15)) if os.path.isdir(C15) else []:
    if d.startswith("t3_pkr_"): caches["C15_" + d] = os.path.join(C15, d)
for k, d in caches.items():
    f = os.path.join(d, "panel_ccy.parquet")
    if not os.path.exists(f): out[k] = "missing"; continue
    p = pd.read_parquet(f, columns=["ksic", "mg", "cc", "year", "n"])
    p["cell"] = p.ksic + "|" + p.mg
    g = p.groupby(["cell", "year"], as_index=False).n.sum()
    rows = []
    for t0 in range(2005, 2019):
        s = g[(g.year >= t0 - 2) & (g.year <= t0)].groupby("cell").n.sum().rename("alt").reset_index(); s["t0"] = t0; rows.append(s)
    S = pd.concat(rows, ignore_index=True)
    M = T.merge(S, on=["cell", "t0"], how="left"); M["alt"] = M.alt.fillna(0.0)
    r = (M.alt.values + 1) / (M.tot_W.values + 1)
    out[k] = dict(median_ratio=float(np.median(r)), alt_pos=float((M.alt > 0).mean()), n=int(len(M)))
    print(k, out[k], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_covd.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
