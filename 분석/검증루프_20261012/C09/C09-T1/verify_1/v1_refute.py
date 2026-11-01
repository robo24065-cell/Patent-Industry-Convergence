# -*- coding: utf-8 -*-
# verify_1: 반박 시도
#  H1 '2017년 이후 현대 특허가 KSIC 29·30 범위 밖 산업으로 정당하게 옮겨졌다'
#  H2 '현대 문서는 기아 표기로 옮겨진 것이 아니라 실제로 빠졌다'
#  H3 '(i) 비율 안정은 U(제공 범위 특허가 있는 법인만) 표본 선택의 산물이다'
import sys, os, json, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
WD = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C09\C09-T1"
sys.path.insert(0, os.path.join(WD, "scripts"))
from common import *
from breaks import break_test
import numpy as np, pandas as pd
t0 = time.time(); res = {}
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"), columns=["appl_no", "appl_year", "rgst_no", "rgst_date", "ksic", "applicant"])
U = pd.read_parquet(os.path.join(CACHE, "U_raw.parquet"), columns=["applicant_cd", "rgst_no", "rgst_date", "appl_no", "appl_date", "title", "ainfo"])
U["ay"] = pd.to_numeric(U.appl_date.str[:4], errors="coerce")
inmap = kr.set_index("appl_no")
U["inscope"] = U.appl_no.isin(inmap.index)
# ---- H1 ----
h = U[(U.applicant_cd == HY) & U.ay.between(2010, 2021)].copy()
h["ksic"] = h.appl_no.map(inmap.ksic)
h["k2"] = h.ksic.str[:2]
t = h[h.inscope].groupby(["ay", "k2"]).size().unstack(fill_value=0)
res["H1_hy_inscope_by_ksic2"] = t.to_dict("index")
res["H1_hy_share_29_30_of_inscope"] = {int(y): round(float(t.loc[y, [c for c in t.columns if c in ("29", "30")]].sum() / t.loc[y].sum()), 3) for y in t.index}
res["H1_hy_inscope_share"] = h.groupby("ay").inscope.mean().round(4).to_dict()
# 차량 키워드 제목 비중(범위 밖) — 범위 밖으로 '정당하게 이동'했다면 범위 밖 구성이 바뀌고 범위 안 비율이 떨어져야 한다
kw = h.title.fillna("").str.contains("차량|자동차|차체|시트|도어|서스펜션|변속")
res["H1_out_vehicle_kw_share"] = h[~h.inscope].assign(kw=kw[~h.inscope]).groupby("ay").kw.mean().round(3).to_dict()
res["H1_in_vehicle_kw_share"] = h[h.inscope].assign(kw=kw[h.inscope]).groupby("ay").kw.mean().round(3).to_dict()
# 2010–16 단독계 대 기아 공동 문서의 범위 안 비율(공동 전환 자체가 범위를 바꾸는지)
h["kia"] = h.ainfo.fillna("").str.contains("기아")
pre = h[h.ay.between(2010, 2016)]
res["H1_pre_inscope_share_solo_vs_joint"] = pre.groupby("kia").inscope.mean().round(4).to_dict()
# ---- H2 ----
# 현대 코드 범위 안 특허(2017–20)가 제공 데이터에 어떤 이름으로 실렸나, 그리고 registry_full에 빠진 것은 등록번호 공란 때문인가
hi = h[h.inscope & h.ay.between(2017, 2020)].copy()
hi["name"] = hi.appl_no.map(inmap.applicant)
hi["meta_rgst_blank"] = hi.appl_no.map(inmap.rgst_no).isna()
rf = set(pd.read_parquet(os.path.join(CACHE, "rf_patent.parquet"), columns=["rgst_no"]).rgst_no)
hi["in_rf"] = hi.rgst_no.isin(rf)
res["H2_names"] = hi.name.value_counts().head(5).to_dict()
res["H2_n"] = len(hi); res["H2_in_rf"] = int(hi.in_rf.sum()); res["H2_not_in_rf_and_meta_rgst_blank"] = int((~hi.in_rf & hi.meta_rgst_blank).sum())
res["H2_not_in_rf_rgst_year"] = hi[~hi.in_rf].rgst_date.str[:4].value_counts().sort_index().to_dict()
# 전체: 범위 안 U 특허 중 registry_full에 없는 것의 등록연월 분포(등록번호 공란 문제)
ui = U[U.inscope].drop_duplicates("rgst_no").copy()
ui["in_rf"] = ui.rgst_no.isin(rf); ui["ry"] = ui.rgst_date.str[:4]
res["H2_all_inscope_not_in_rf_by_rgst_year"] = ui.groupby("ry").in_rf.apply(lambda s: int((~s).sum())).loc["2020":"2025"].to_dict()
res["H2_all_inscope_in_rf_rate_by_rgst_year"] = ui.groupby("ry").in_rf.mean().round(4).loc["2020":"2025"].to_dict()
ui["rm"] = ui.rgst_date.str[:6]
mm = ui[ui.ry.isin(["2023", "2024", "2025"])].groupby("rm").in_rf.mean().round(3)
res["H2_in_rf_rate_by_month_2023_25"] = mm.to_dict()
# ---- H3 ----
P = U.drop_duplicates("rgst_no")
first = U.groupby("applicant_cd").ay.min()
old = set(first[first <= 2009].index)
Po = U[U.applicant_cd.isin(old)].drop_duplicates("rgst_no")
g = Po[Po.ay.between(2008, 2023)].groupby("ay").inscope.mean()
res["H3_panel_old_firms_share"] = g.round(4).to_dict()
res["H3_panel_break"] = break_test(g.to_dict())
# 개인·외국인 포함 전체 모집단: 제공 KR 등록(registry_full) / KOSIS 특허 등록(등록연도) — U 선택과 무관
s2 = json.load(open(os.path.join(CACHE, "s2_results.json"), encoding="utf-8"))
ser = {int(r["ry"]): r["rf_over_kosis"] for r in s2["res"]["by_rgst_year"]}
res["H3_rf_over_kosis"] = ser
res["H3_rf_over_kosis_break_2010_22"] = break_test(ser)
# 범위 안 U 특허 / KOSIS (등록연도, 출원번호 연결이라 등록번호 공란 영향 없음)
ser2 = {int(r["ry"]): r["U_in_over_kosis"] for r in s2["res"]["by_rgst_year"]}
res["H3_Uin_over_kosis_break_2010_22"] = break_test(ser2)
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(WD, "verify_1", "v1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
for k, v in res.items():
    print(k, json.dumps(v, ensure_ascii=False, default=float)[:900])
