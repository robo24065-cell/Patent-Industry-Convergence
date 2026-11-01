# -*- coding: utf-8 -*-
# 반박 검증: 그림4 RTA가 출원연도 원계열(보정 없음)인지, WIPO 보정이 들어갔다면 값이 달라지는지 독립 재계산
import os, sys, json
import numpy as np, pandas as pd
M5 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5"
F5 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\figs\fig05_rta_map_5x50.csv"
OUT = os.path.dirname(os.path.abspath(__file__))
CC = ["KR","US","JP","CN","EP"]
res = {}
a = pd.read_parquet(os.path.join(M5,"m1_counts_appl.parquet")); a["ksic"]=a.ksic.astype(str)
w = a[a.appl_year.between(2017,2019)].groupby(["ksic","cc"]).n_inv.sum().unstack().reindex(columns=CC).fillna(0)
s = w/w.sum(0); rta = s.div(s.mean(1),axis=0)
f = pd.read_csv(F5, encoding="utf-8-sig", dtype={"ksic":str}).set_index("ksic")
res["fig05_cols"] = list(f.columns)
d = (rta.reindex(f.index)[CC]-f[CC]).abs().values
res["max_absdiff_raw_appl_vs_fig05"] = float(np.nanmax(d)); res["n_rows"]=int(len(f))
# 공개연도 원계열/보정계열로 계산했을 때 (가설: 캡션이 맞다면 이쪽과 일치해야)
pc = pd.read_csv(os.path.join(M5,"m2_panel_pub_corrected.csv"), dtype={"ksic":str})
for col in ["n","n_corr"]:
    for yrs in [(2017,2019),(2018,2020),(2019,2021)]:
        ww = pc[pc.year.between(*yrs)].groupby(["ksic","cc"])[col].sum().unstack().reindex(columns=CC).fillna(0)
        ss = ww/ww.sum(0); rr = ss.div(ss.mean(1),axis=0).reindex(f.index)
        res[f"max_absdiff_pub_{col}_{yrs[0]}_{yrs[1]}_vs_fig05"] = float(np.nanmax((rr[CC]-f[CC]).abs().values))
        lead = rr[CC].idxmax(1)
        res[f"pub_{col}_{yrs[0]}_{yrs[1]}_KR_gt1"] = int((rr.KR>1).sum()); res[f"pub_{col}_{yrs[0]}_{yrs[1]}_KR_lead"]=int((lead=="KR").sum())
    # 보정계수 != 1 인 칸이 P3 공개연도에 있는지
cf = pd.read_csv(os.path.join(M5,"m2_coverage_factors_pub.csv"))
nz = cf[(cf.f.round(6)!=1)]
res["flagged_ccyears"] = sorted(set(map(tuple, nz[["cc","year"]].astype(str).values.tolist())))
ct = pd.read_csv(os.path.join(M5,"m2_coverage_factors_pub_total.csv"))
res["flagged_total"] = ct[ct.f_tot!=1][["cc","year","f_tot"]].round(3).values.tolist()
pc2 = pc.copy(); pc2["diff"]=(pc2.n_corr-pc2.n).abs()
res["pub_years_with_any_correction"] = sorted(pc2[pc2["diff"]>1e-9].groupby(["cc","year"]).size().index.map(lambda t: f"{t[0]}{t[1]}").tolist())
lead = f[CC].idxmax(1); res["fig05_KR_gt1"]=int((f.KR>1).sum()); res["fig05_lead_counts"]=lead.value_counts().to_dict()
# 원 map5 profile 컬럼과도 대조
pr = pd.read_csv(os.path.join(M5,"m2_ksic_profile.csv"), dtype={"ksic":str}, encoding="utf-8-sig").set_index("ksic")
res["max_absdiff_profile_rtaP3_vs_fig05"] = float(np.nanmax((pr.reindex(f.index)[[f"rta_{c}_P3" for c in CC]].values - f[CC].values).__abs__()))
json.dump(res, open(os.path.join(OUT,"v0_results.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
