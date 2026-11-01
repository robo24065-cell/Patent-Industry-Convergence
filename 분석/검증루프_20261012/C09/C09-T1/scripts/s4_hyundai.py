# -*- coding: utf-8 -*-
# 4단계: 현대자동차 사례 해부 (이름·코드 병행)
import sys, os, time, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
t0 = time.time(); res = {}
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"))
kr["sub"] = kr.main_ipc.str[:4]
kr10 = kr[kr.appl_no.str[:2] == "10"].copy()
HYN, KIAN = "현대자동차주식회사", "기아 주식회사"
# (1) 이름 기준 연도별
nm = kr10[kr10.applicant.isin([HYN, KIAN])]
t = nm.groupby(["appl_year", "applicant"]).size().unstack(fill_value=0)
t["합계"] = t.sum(1)
res["by_name_year"] = t.loc[2008:2023].reset_index().to_dict("records")
# (2) 2016년 현대 이름 문서 453건의 KSIC·서브클래스
h16 = kr10[(kr10.applicant == HYN) & (kr10.appl_year == 2016)]
res["hy2016_n"] = len(h16)
res["hy2016_ksic"] = h16.ksic.value_counts().head(15).to_dict()
res["hy2016_sub"] = h16["sub"].value_counts().head(15).to_dict()
res["hy2016_ksic_sub_top"] = h16.groupby(["ksic", "sub"]).size().sort_values(ascending=False).head(15).reset_index().astype(str).values.tolist()
# (3) 코드 기준(등록원부 출원인 코드): 현대 코드가 들어간 등록 문서의 제공 이름과 KSIC
rfa = pd.read_parquet(os.path.join(CACHE, "rf_applicant.parquet"))
rfp = pd.read_parquet(os.path.join(CACHE, "rf_patent.parquet"), columns=["rgst_no", "applNo", "ksic"])
hy_rg = set(rfa[rfa.applicantCd == HY].rgst_no); kia_rg = set(rfa[rfa.applicantCd == KIA].rgst_no)
rp = rfp[rfp.rgst_no.isin(hy_rg | kia_rg)].copy()
rp["hy"] = rp.rgst_no.isin(hy_rg); rp["kia"] = rp.rgst_no.isin(kia_rg)
rp = rp.merge(kr10[["appl_no", "applicant", "appl_year", "sub"]], left_on="applNo", right_on="appl_no", how="left")
rp["combo"] = np.select([rp.hy & ~rp.kia, rp.hy & rp.kia], ["현대만", "현대+기아"], "기아만")
x = rp.groupby(["appl_year", "combo", "applicant"]).size().reset_index(name="n")
x = x[x.appl_year.between(2012, 2021)]
res["code_combo_name"] = x.sort_values(["appl_year", "n"], ascending=[True, False]).groupby(["appl_year", "combo"]).head(2).astype(str).values.tolist()
# 현대 코드 등록 문서의 KSIC 분포(연도별) — 2016 대 2018
hk = rp[rp.hy].groupby(["appl_year", "ksic"]).size().unstack(fill_value=0)
res["hy_code_ksic_2014_2019"] = {int(y): hk.loc[y].sort_values(ascending=False).head(8).to_dict() for y in range(2014, 2020) if y in hk.index}
res["hy_code_reg_by_year"] = rp[rp.hy].groupby("appl_year").size().loc[2008:2023].to_dict()
# (4) 2016 현대 상위 서브클래스에서 현대·기아 이외 출원인의 범위 안 문서 수
subs = list(res["hy2016_sub"].keys())[:8]
oth = kr10[~kr10.applicant.isin([HYN, KIAN]) & kr10["sub"].isin(subs)]
o = oth.groupby(["sub", "appl_year"]).size().unstack(fill_value=0).loc[:, 2010:2023]
hk2 = kr10[kr10.applicant.isin([HYN, KIAN]) & kr10["sub"].isin(subs)].groupby(["sub", "appl_year"]).size().unstack(fill_value=0).reindex(columns=range(2010, 2024), fill_value=0)
res["hy_subs"] = subs
res["other_applicants_in_hy_subs"] = {s: o.loc[s].to_dict() for s in subs}
res["hy_kia_names_in_hy_subs"] = {s: hk2.loc[s].to_dict() for s in subs if s in hk2.index}
# 비율: 다른 출원인 2017–19 평균 / 2013–16 평균
res["other_ratio_1719_vs_1316"] = {s: round(float(o.loc[s, 2017:2019].mean() / o.loc[s, 2013:2016].mean()), 3) for s in subs}
res["hk_ratio_1719_vs_1316"] = {s: round(float(hk2.loc[s, 2017:2019].mean() / max(hk2.loc[s, 2013:2016].mean(), 1e-9)), 3) for s in subs if s in hk2.index}
# (5) business_rights 현대 권리자 포트폴리오 대비 범위 안 비율(코드 기준; s2와 같은 정의) + 단독/공동
U = pd.read_parquet(os.path.join(CACHE, "U_raw.parquet"), columns=["applicant_cd", "rgst_no", "appl_no", "appl_date", "ainfo", "title"])
U = U[U.applicant_cd == HY].copy()
U["ay"] = pd.to_numeric(U.appl_date.str[:4], errors="coerce")
U["inscope"] = U.appl_no.isin(set(kr.appl_no))
U["kia_in_ainfo"] = U.ainfo.fillna("").str.contains("기아")
U["hy_in_ainfo"] = U.ainfo.fillna("").str.contains("현대자동차")
g = U[U.ay.between(2008, 2023)].groupby(["ay", "kia_in_ainfo"]).agg(n=("rgst_no", "size"), n_in=("inscope", "sum")).unstack()
res["br_hy_by_kia_flag"] = {int(y): {"단독계_n": int(g.loc[y, ("n", False)]) if ("n", False) in g.columns and not np.isnan(g.loc[y, ("n", False)]) else 0,
                                    "단독계_in": int(g.loc[y, ("n_in", False)]) if not np.isnan(g.loc[y, ("n_in", False)]) else 0,
                                    "기아공동_n": int(g.loc[y, ("n", True)]) if not np.isnan(g.loc[y, ("n", True)]) else 0,
                                    "기아공동_in": int(g.loc[y, ("n_in", True)]) if not np.isnan(g.loc[y, ("n_in", True)]) else 0} for y in g.index}
res["br_hy_originally_applicant_share"] = float(U.hy_in_ainfo.mean())
# 범위 안 현대 코드 문서가 제공 데이터에서 어떤 이름으로 실렸나
ui = U[U.inscope].merge(kr10[["appl_no", "applicant"]], on="appl_no", how="left")
res["br_hy_inscope_provided_name"] = ui[ui.ay.between(2014, 2020)].groupby(["ay", "applicant"]).size().sort_values(ascending=False).groupby(level=0).head(3).reset_index().astype(str).values.tolist()
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(CACHE, "s4_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str)[:1500])
