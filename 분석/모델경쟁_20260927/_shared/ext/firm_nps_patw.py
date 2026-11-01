# -*- coding: utf-8 -*-
"""'특허 KSIC 라벨로 사업장 고용을 산업에 배분' — 특허 보유 기업(국민연금 매칭)의 가입자수를
그 기업의 제공범위 KR 등록특허 KSIC 비중으로 50 KSIC에 배분한 산업×연도 고용 패널.
(NPS 업종코드는 국세청 코드라 50 KSIC 중 28개가 2020년 이전 식별 불가 → 특허 라벨이 다리 역할)
시점정합: 스냅샷 연도 Y의 배분 가중치는 등록일 ≤ Y-12-31 특허만 사용.
출력: firm_nps_ksic50_patw_panel.parquet (ksic, snap, emp_patw, n_firms, emp_patw_exit[직전 스냅샷 대비 탈퇴기업 고용])
      firm_nps_test_industry.csv (개발 원점 t0=2016 산업 수준 검정)"""
import os, sqlite3
import numpy as np, pandas as pd, duckdb
import statsmodels.api as sm
from scipy.stats import spearmanr

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SH = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared")
EXT = os.path.join(SH, "ext")
pf = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache\pat_firm.parquet"), columns=["rgst_no", "applicantCd", "ksic", "rgstDate"])
fp = pd.read_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"))
pf = pf[pf.applicantCd.isin(set(fp.applicantCd))]
rows = []
snaps = sorted(fp.snap.unique())
for s in snaps:
    Y = int(s[:4])
    w = pf[pf.rgstDate.astype(str) <= f"{Y}1231"].groupby(["applicantCd", "ksic"]).size().rename("n").reset_index()
    w["sh"] = w.n / w.groupby("applicantCd").n.transform("sum")
    f = fp[(fp.snap == s) & fp.active][["applicantCd", "n_emp"]]
    x = w.merge(f, on="applicantCd")
    x["e"] = x.sh * x.n_emp
    g = x.groupby("ksic").agg(emp_patw=("e", "sum"), n_firms=("applicantCd", "nunique")).reset_index()
    g["snap"] = s
    rows.append(g)
P = pd.concat(rows, ignore_index=True)
P.to_parquet(os.path.join(EXT, "firm_nps_ksic50_patw_panel.parquet"), index=False)
print(P.groupby("snap").agg(k=("ksic", "nunique"), emp=("emp_patw", "sum"), firms=("n_firms", "sum")).round(0).to_string())

# ---- 개발 원점 t0=2016 검정: 2015-11→2016-11 고용 변화가 이후 3년(2017–19) KR 출원 증가를 설명하는가 ----
p = duckdb.sql(f"select ksic, appl_year y, count(distinct appl_no) n from read_parquet('{SH}/meta/*.parquet') where cc='KR' and appl_year between 2011 and 2019 group by 1,2").df()
W = p.pivot(index="ksic", columns="y", values="n").fillna(0)
E = P.pivot(index="ksic", columns="snap", values="emp_patw")
df = pd.DataFrame({
    "g_emp_patw": np.log(E["2016-12"] / E["2015-12"]),
    "y": np.log((W[[2017, 2018, 2019]].sum(1) + 1) / (W[[2014, 2015, 2016]].sum(1) + 1)),
    "past": np.log((W[[2014, 2015, 2016]].sum(1) + 1) / (W[[2011, 2012, 2013]].sum(1) + 1)),
}).dropna()
out = []
r = spearmanr(df.g_emp_patw, df.y); out.append(("spearman(g_emp_patw, y)", len(df), round(r.statistic, 3), round(r.pvalue, 3)))
r = spearmanr(df.past, df.y); out.append(("spearman(past, y)", len(df), round(r.statistic, 3), round(r.pvalue, 3)))
m0 = sm.OLS(df.y, sm.add_constant(df[["past"]])).fit(); m1 = sm.OLS(df.y, sm.add_constant(df[["past", "g_emp_patw"]])).fit()
out.append(("OLS y~past R2", len(df), round(m0.rsquared, 3), None))
out.append(("OLS y~past+g_emp_patw 계수/p", len(df), round(m1.params["g_emp_patw"], 3), round(m1.pvalues["g_emp_patw"], 3)))
out.append(("OLS y~past+g_emp_patw R2", len(df), round(m1.rsquared, 3), None))
O = pd.DataFrame(out, columns=["test", "n_ksic", "stat", "p"]); print(O.to_string())
O.to_csv(os.path.join(EXT, "firm_nps_test_industry.csv"), index=False, encoding="utf-8-sig")
