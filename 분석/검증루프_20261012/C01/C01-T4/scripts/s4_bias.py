# s4: C-A 주 정의 단위의 NPS 연결 편향(연결 vs 비연결) + 시점 정합 월별 건수 요약
import os, json
import pandas as pd, numpy as np
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
U = pd.read_parquet(os.path.join(OUTD, "s3_units_main.parquet"))
out = {}
U["nps_hit"] = U.nps_hit.astype(bool)
dev = U[U.t0 <= 2023].copy(); dev["y"] = dev.y_main.astype(float)
out["link_rate_units_by_t0"] = U.groupby("t0").nps_hit.mean().round(4).to_dict()
def smd(a, b):
    a, b = a.dropna(), b.dropna()
    sp = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan
L, N = dev[dev.nps_hit], dev[~dev.nps_hit]
out["n_linked_units"], out["n_unlinked_units"] = int(len(L)), int(len(N))
out["smd"] = {c: round(smd(np.log1p(L[c]) if c in ("n_live", "f_live") else L[c], np.log1p(N[c]) if c in ("n_live", "f_live") else N[c]), 3)
              for c in ["n_live", "f_live", "f_nk", "first_appl_y", "t0"]}
out["pos_rate_linked"], out["pos_rate_unlinked"] = round(float(L.y.mean()), 5), round(float(N.y.mean()), 5)
out["pos_rate_ratio_linked_over_unlinked"] = round(float(L.y.mean() / N.y.mean()), 3)
# KSIC 분포 상이도(Duncan): 0.5*sum|pL-pN|
pk = pd.concat([L.ksic.value_counts(normalize=True).rename("L"), N.ksic.value_counts(normalize=True).rename("N")], axis=1).fillna(0)
out["ksic_dissimilarity"] = round(float(0.5 * (pk.L - pk.N).abs().sum()), 4)
ps = pd.concat([L.sido.value_counts(normalize=True).rename("L"), N.sido.value_counts(normalize=True).rename("N")], axis=1).fillna(0)
out["sido_dissimilarity"] = round(float(0.5 * (ps.L - ps.N).abs().sum()), 4)
out["sido_top_linked"] = ps.sort_values("L", ascending=False).head(6).round(3).to_dict()
# 보정 가능성: KSIC×원점 칸 중 연결·비연결 공존 비율
cell = dev.groupby(["ksic", "t0"]).nps_hit.agg(["min", "max", "size"])
out["cells_total"] = int(len(cell)); out["cells_both"] = int(((cell["min"] == False) & (cell["max"] == True)).sum())
out["cells_both_share"] = round(out["cells_both"] / out["cells_total"], 4)
# NPS 규모(연결 기업, 2015+): 양성 vs 음성 가입자수 중앙값
l15 = L[(L.t0 >= 2015) & L.n_emp.notna()]
out["linked_nps_emp_median_pos_neg"] = [float(l15[l15.y == 1].n_emp.median()), float(l15[l15.y == 0].n_emp.median())]
out["linked_with_nps_emp_share_2015plus"] = round(float(L[L.t0 >= 2015].n_emp.notna().mean()), 4)
# 양성의 KSIC 집중도
pos = dev[dev.y == 1]
out["pos_total_dev"] = int(len(pos)); out["pos_ksic_top5_share"] = round(float(pos.ksic.value_counts(normalize=True).head(5).sum()), 4)
out["pos_ksic_n"] = int(pos.ksic.nunique())
lag = json.load(open(os.path.join(OUTD, "s2_lag.json"), encoding="utf-8"))
m = pd.Series(lag["lapse_reg_monthly"])
out["lapse_reg_monthly_2025_2026"] = {k: v for k, v in m.items() if k >= "2025-01"}
out["lapse_reg_monthly_mean_2023_2024"] = float(m[(m.index >= "2023-01") & (m.index <= "2024-12")].mean())
json.dump(out, open(os.path.join(OUTD, "s4_bias.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str)[:3000])
