# -*- coding: utf-8 -*-
r"""C07-T1 (b): 분포·결측·급변·조사 범위(10인 이상) 과소 집계·10↔11 겹침. 출력 cache\s5_*.csv/json"""
import os, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(B, r"분석\검증루프_20261012\C07\C07-T1")
sys.path.insert(0, os.path.join(B, r"분석\모델경쟁_20260927\_shared_report\label_validity"))
from lv_common import ksic_to_group
W = pd.read_csv(os.path.join(B, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
W.columns = ["basis", "code", "name", "item", "itemnm", "year", "v", "mask"]
K50 = sorted(W.code.unique()); G = pd.Series({k: ksic_to_group(k) for k in K50}); gs = G.value_counts(); K34 = [k for k in K50 if gs[G[k]] > 1]
st = {}
st["dup_keys"] = int(W.duplicated(["basis", "code", "item", "year"]).sum())
st["years_10"] = sorted(W[W.basis == "10차"].year.unique().tolist()); st["years_11"] = sorted(W[W.basis == "11차"].year.unique().tolist())
st["nan_values"] = int(W.v.isna().sum()); st["zero_values"] = int((W.v == 0).sum())
st["zero_rows"] = W[W.v == 0][["basis", "code", "item", "year"]].astype(str).values.tolist()[:20]
st["neg_values"] = int((W.v < 0).sum())
w10 = W[(W.basis == "10차") & (W["mask"] == 0)]
rows = []
for (k, it), g in w10[w10.item.isin(["T01", "T02", "T04"])].groupby(["code", "item"]):
    s = g.set_index("year").v.sort_index().dropna()
    dl = np.log(s).diff().dropna()
    big = dl[dl.abs() > 1.0]
    rows.append({"code": k, "item": it, "in_K34": k in K34, "n_years": len(s), "max_abs_dlog": round(float(dl.abs().max()), 3) if len(dl) else np.nan,
                 "n_jump_gt1": len(big), "jump_years": "|".join(f"{y}:{v:+.2f}" for y, v in big.items()),
                 "est_2016_17_mean": float(w10[(w10.code == k) & (w10.item == "T01") & w10.year.isin([2016, 2017])].v.mean())})
J = pd.DataFrame(rows); J.to_csv(os.path.join(H, r"cache\s5_jumps.csv"), index=False, encoding="utf-8-sig")
st["jumps"] = {"codes_with_any_jump": int(J[J.n_jump_gt1 > 0].code.nunique()), "K34_codes_with_jump": int(J[(J.n_jump_gt1 > 0) & J.in_K34].code.nunique()),
               "by_item": J.groupby("item").n_jump_gt1.apply(lambda s: int((s > 0).sum())).to_dict(),
               "list": J[J.n_jump_gt1 > 0][["code", "item", "jump_years"]].values.tolist()}
est = J[J.item == "T01"].set_index("code").est_2016_17_mean
st["small_est_K34(<20 in 2016-17)"] = est[est.index.isin(K34) & (est < 20)].round(1).to_dict()
# 10↔11 2020 겹침
o = W[W.year == 2020].pivot_table(index=["code", "item"], columns="basis", values="v")
o["r"] = o["11차"] / o["10차"]
st["overlap2020_ratio_11_over_10"] = {"n": int(o.r.notna().sum()), "min": float(o.r.min()), "max": float(o.r.max()),
                                      "share_within_1pct": float(((o.r - 1).abs() <= 0.01).mean()),
                                      "outside_1pct": o[(o.r - 1).abs() > 0.01].r.round(4).reset_index().astype(str).values.tolist()[:30]}
# 조사 범위: 전국사업체조사(전 규모) 대비
c = pd.DataFrame(json.load(open(os.path.join(H, r"download\census\DT_1K52C03_2016_2020.json"), encoding="utf-8")))
c["v"] = pd.to_numeric(c.DT, errors="coerce"); c["code"] = c.C2.str.lstrip("C"); c["year"] = c.PRD_DE.astype(int)
tot = c[c.C3 == "0"].pivot_table(index=["code", "year"], columns="ITM_ID", values="v")
ge10 = c[c.C3.isin(list("3456789"))].groupby(["code", "year", "ITM_ID"]).v.sum().unstack()
mm = w10[w10.item.isin(["T01", "T02"])].pivot_table(index=["code", "year"], columns="item", values="v")
cov = pd.DataFrame({"est_all": tot["T1"], "emp_all": tot["T2"], "est_ge10_census": ge10["T1"], "emp_ge10_census": ge10["T2"]}).join(mm.rename(columns={"T01": "est_mms", "T02": "emp_mms"}), how="left")
cov["est_cov"] = cov.est_mms / cov.est_all; cov["emp_cov"] = cov.emp_mms / cov.emp_all
cov["est_ge10_share_census"] = cov.est_ge10_census / cov.est_all; cov["emp_ge10_share_census"] = cov.emp_ge10_census / cov.emp_all
cov = cov.reset_index(); cov["in_K34"] = cov.code.isin(K34)
cov.to_csv(os.path.join(H, r"cache\s5_coverage.csv"), index=False, encoding="utf-8-sig")
c17 = cov[cov.year.isin([2016, 2017])].groupby("code")[["est_cov", "emp_cov", "est_ge10_share_census", "emp_ge10_share_census"]].mean()
st["coverage_2016_17"] = {"manufacturing_C": c17.loc["C"].round(3).to_dict() if "C" in c17.index else None,
                          "codes_est_cov_lt30": c17[(c17.est_cov < 0.30) & (c17.index != "C")].round(3).index.tolist(),
                          "n_est_cov_lt30_K34": int(((c17.est_cov < 0.30) & c17.index.isin(K34)).sum()),
                          "est_cov_median_50": float(c17.drop("C", errors="ignore").est_cov.median()),
                          "emp_cov_median_50": float(c17.drop("C", errors="ignore").emp_cov.median()),
                          "emp_cov_min": [c17.drop("C", errors="ignore").emp_cov.idxmin(), float(c17.drop("C", errors="ignore").emp_cov.min())],
                          "cases": c17.loc[[k for k in ["26211", "26212", "26111", "29280"] if k in c17.index]].round(3).to_dict("index")}
# 사례 그룹 합계(디스플레이 26211+26212+26219)
s10 = pd.concat([pd.DataFrame(json.load(open(os.path.join(H, rf"download\kosis10\{f}.json"), encoding="utf-8"))) for f in ("DT_1FS1001_2007_2020", "DT_1FS1001_siblings_2007_2020")])
s10["v"] = pd.to_numeric(s10.DT, errors="coerce")
disp = s10[s10.C2.isin(["C26211", "C26212", "C26219"]) & (s10.ITM_ID == "T04")].pivot_table(index="C2", columns="PRD_DE", values="v")
disp.loc["합계"] = disp.sum()
st["display_T04_share_of_26212_in_sum"] = (disp.loc["C26212"] / disp.loc["합계"]).round(3).to_dict()
g = lambda s, a, b, sp: round(100 * (np.log(s[b].mean()) - np.log(s[a].mean())) / sp, 2)
st["display_sum_growth_0809_1617"] = g(disp.loc["합계"], ["2008", "2009"], ["2016", "2017"], 8)
print(json.dumps(st, ensure_ascii=False, indent=1, default=str)[:7000])
json.dump(st, open(os.path.join(H, r"cache\s5_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
