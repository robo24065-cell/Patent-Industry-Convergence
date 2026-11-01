# 3.1.1·3.1.2 독립 재계산: m1_counts_appl(원천 집계)에서 RTA(동일가중) 재생성
import pandas as pd, numpy as np, json, os, sys
from scipy.stats import spearmanr
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
M = B + r"\모델경쟁_20260927\_shared_report\fusion_advantage\map5"
F = B + r"\보고서_20261012\figs"
OUT = B + r"\검증루프_20261012\C02\C02-T3"
a = pd.read_parquet(M + r"\m1_counts_appl.parquet"); a["ksic"] = a.ksic.astype(str)
CCS = ["KR", "US", "JP", "CN", "EP"]
def rta(y0, y1, col="n_inv"):
    s = a[a.appl_year.between(y0, y1)].groupby(["ksic", "cc"])[col].sum().unstack().reindex(columns=CCS).fillna(0)
    sh = s / s.sum(0)
    return sh.div(sh.mean(1), axis=0), s
res = {}
r3, n3 = rta(2017, 2019)
res["n_ksic"] = len(r3)
res["kr_rta_gt1"] = int((r3.KR > 1).sum())
lead = r3.idxmax(1)
res["leaders"] = lead.value_counts().to_dict()
kr_lead = r3[lead == "KR"].KR.sort_values(ascending=False)
names = pd.read_csv(F + r"\fig05_rta_map_5x50.csv", dtype={"ksic": str}).set_index("ksic")
res["kr_leaders"] = [(k, names.loc[k, "name"], round(v, 4)) for k, v in kr_lead.items()]
ratio = r3.KR / r3.max(1)
res["gap_lt_0.7"] = int((ratio < 0.7).sum())
g = ratio.sort_values()
res["gap_smallest"] = [(k, names.loc[k, "name"], lead[k], round(v, 4)) for k, v in g.head(5).items()]
# 그림 CSV 대조
fc = names[CCS]
res["fig05_maxabs_diff"] = float((fc - r3.loc[fc.index]).abs().max().max())
res["fig05_leader_match"] = bool((names.leader == lead.loc[names.index]).all())
# 동일가중 vs Balassa 순위상관 (KR)
bal = (n3 / n3.values.sum()).div((n3.sum(1) / n3.values.sum()), axis=0).div(n3.sum(0) / n3.values.sum(), axis=1)
res["kr_ew_vs_bal_rho"] = round(spearmanr(r3.KR, bal.KR).statistic, 4)
# P2 (2015–17) RTA → 표5 사분면
r2, _ = rta(2015, 2017)
q = pd.read_csv(F + r"\fig06_quadrant_tech_export.csv", dtype={"ksic": str}).set_index("ksic")
res["fig06_patrta_maxabs_diff"] = float((q.pat_rta_P2 - r2.KR.loc[q.index]).abs().max())
def quad(t, e):
    if pd.isna(t) or pd.isna(e): return "판정불가"
    return ("기술 강" if t > 1 else "기술 약") + "·" + ("수출 강" if e > 1 else "수출 약")
q["quad_re"] = [quad(t, e) for t, e in zip(r2.KR.loc[q.index], q.exp_rca_P2)]
res["quad_counts_recomputed"] = q.quad_re.value_counts().to_dict()
res["quad_counts_fig06"] = q.quad_tech_export_P2.value_counts().to_dict()
res["quad_csv_rows"] = len(q)
res["quad_examples"] = {k: list(q[q.quad_re == k].name) for k in q.quad_re.unique()}
# 판정불가 2개는 fig06에 있나?
res["quad_rho_rta_rca"] = round(spearmanr(q.pat_rta_P2, q.exp_rca_P2, nan_policy="omit").statistic, 4)
qq = q.dropna(subset=["exp_rca_P2"])
res["quad_rho_rta_rca_n"] = len(qq)
mixed = qq.groupby("cw_group").quad_re.nunique()
grp_n = qq.groupby("cw_group").size()
multi = grp_n[grp_n >= 2].index
res["cw_multi_groups_in_fig06"] = int(len(multi)); res["cw_multi_groups_mixed"] = int((mixed.loc[multi] > 1).sum())
res["cw_ksic_in_multi"] = int(grp_n.loc[multi].sum())
json.dump(res, open(OUT + r"\s1_map_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
