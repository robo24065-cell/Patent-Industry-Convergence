# -*- coding: utf-8 -*-
"""J1 정부R&D 과제(과제연도 단위)의 '50개 산업 특허' 성과 조기 예측 — 학습표.
단위 = NTIS 과제고유번호(연차별 고유) × 과제연도 t (2010–2020, 보안과제 제외).
결과 = 그 과제의 성과로 등록된 제공범위(50 KSIC) KR 특허 중 출원연도 ∈ [t, t+1] 인 것이 1건 이상.
🔒 최종시험 t=2022 → 출원 2022–23 (F=2022). 개발 레이블은 출원연도 ≤ 2021만 읽는다(assert). 개발 평가 t 2014–2020, 학습 t ≤ v−2.
피처는 과제 선정 시점(과제연도 t 시작) 정보: 과제 속성(t년 확정) + 수행기관·선행과제의 t−1년까지 성과(성과 레코드 Year ≤ t−1, 출원연도 ≤ t−1).
NTIS는 현재 스냅샷(준-정합): 성과 등록 지연·사후 수정은 재구성 불가.
묶음(사전 선언)
 M0 행정 정보 + 수행기관 전체 특허 성과 이력(NTIS만): 연구비·민간부담·수행주체·연구단계·계속·연차·기간·부처·기관 과거 과제 수·과거 KR 특허 성과(전 분야)·선행과제 성과
 M1 과제 내용: KSIC50 텍스트 대응(b1·b2 점수·배정), 6T, 과학기술표준분류 대분류, 경제사회목적, 기술수명주기, 적용분야
 M2 제공데이터 연결: 수행기관·선행과제의 과거 '50개 산업' 특허 성과(건수·전 분야 대비 비중)
"""
import time
import numpy as np, pandas as pd
from r5_common import *

t = time.time()
def log(*a): print(*a, round(time.time() - t), flush=True)

P = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "research_agency", "perform_grp", "dev_phase", "continuous", "pre_pjt", "ministry", "sixt_cd",
                                                              "cls_L_canon", "econ_goal_cd", "tech_lifecycle", "apply1_cd", "gov_funds", "total_funds", "sbiz_funds", "nongov_funds",
                                                              "total_start", "total_end", "period_start", "period_end", "secret_flag"])
P = P[(P.year >= 2007) & (P.year <= 2020) & (P.secret_flag.astype(float).fillna(0) == 0)].copy()
P["year"] = P.year.astype(int)
P["org"] = P.research_agency.map(norm_name)
km = pd.read_parquet(f"{EXT}/ntis_project_ksic_text_map.parquet")
P = P.merge(km.drop(columns=["year"]), on="pjt_no", how="left")
log("projects", len(P))

# 성과: 제공범위 연결(50 KSIC) — 출원연도 ≤ 2021만
L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "ksic", "ProjectID", "appl_year"])
L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
L = L[L.appl_year <= J1_MAX_APPL_YEAR].drop_duplicates(["ProjectID", "appl_no"])
assert L.appl_year.max() <= J1_MAX_APPL_YEAR
# 성과: 전 분야 KR 특허(NTIS 원시) — 출원 레코드 Year=출원연도. 피처용(Year ≤ t−1)만 쓰며 2021 이하만 읽는다
R = pd.read_parquet(f"{EXT}/ntis_rpatent_raw.parquet", columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber"])
R["Year"] = pd.to_numeric(R.Year, errors="coerce")
R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.RegistType == "출원") & (R.Year <= J1_MAX_APPL_YEAR)].drop_duplicates(["ProjectID", "RegistNumber"])
log("links", len(L), len(R))

pid2 = P.set_index("pjt_no")
# 과제별 연도별 성과 수
Lc = L.groupby(["ProjectID", "appl_year"]).size().rename("n").reset_index()
Rc = R.groupby(["ProjectID", "Year"]).size().rename("n").reset_index().rename(columns={"Year": "appl_year"})
Lc = Lc.merge(P[["pjt_no", "org", "year"]], left_on="ProjectID", right_on="pjt_no", how="inner")
Rc = Rc.merge(P[["pjt_no", "org", "year"]], left_on="ProjectID", right_on="pjt_no", how="inner")

cat = {}
for cn in ["perform_grp", "dev_phase", "ministry", "sixt_cd", "cls_L_canon", "econ_goal_cd", "tech_lifecycle", "apply1_cd"]:
    codes, uniq = pd.factorize(P[cn].astype(str)); P[cn + "_c"] = codes
P["log_gov"] = np.log1p(P.gov_funds.fillna(0) / 1e6); P["log_tot"] = np.log1p(P.total_funds.fillna(0) / 1e6)
P["nongov_sh"] = (P.nongov_funds.fillna(0) / P.total_funds.replace(0, np.nan)).clip(0, 1)
P["sbiz_sh"] = (P.sbiz_funds.fillna(0) / P.total_funds.replace(0, np.nan)).clip(0, 1)
ts = pd.to_datetime(P.total_start.astype(str).str[:10], errors="coerce"); te_ = pd.to_datetime(P.total_end.astype(str).str[:10], errors="coerce")
P["series_yr"] = P.year - ts.dt.year; P["total_len"] = (te_ - ts).dt.days / 365.25
ps = pd.to_datetime(P.period_start.astype(str), format="%Y%m%d", errors="coerce"); pe = pd.to_datetime(P.period_end.astype(str), format="%Y%m%d", errors="coerce")
P["period_m"] = (pe - ps).dt.days / 30.4; P["end_in_year"] = (te_.dt.year.fillna(0).astype(int) == P.year).astype(int)
P["cont"] = (P.continuous.astype(str) == "2").astype(int)
P["has_pre"] = P.pre_pjt.notna().astype(int)
P["firm"] = P.perform_grp.astype(str).isin(["중소기업", "대기업", "중견기업"]).astype(int)
for c2 in ["b1_tfidf_assigned", "b2_dict_assigned"]:
    P[c2] = P[c2].astype(float).fillna(0).astype(int)

out = []
for v in range(2010, 2021):
    assert v + 1 <= J1_MAX_APPL_YEAR
    X = P[P.year == v].copy()
    # 결과
    pos = set(Lc[(Lc.year == v) & Lc.appl_year.between(v, v + 1)].ProjectID)
    X["y"] = X.pjt_no.isin(pos).astype(int)
    # 수행기관 이력(t−3..t−1 과제, 성과 출원연도 ≤ t−1)
    hist = P[P.year.between(v - 3, v - 1)]
    nproj = hist.groupby("org").size()
    X["org_nproj"] = np.log1p(X.org.map(nproj).fillna(0))
    ra = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= v - 1)].groupby("org").n.sum()
    la = Lc[Lc.year.between(v - 3, v - 1) & (Lc.appl_year <= v - 1)].groupby("org").n.sum()
    rp = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= v - 1)].groupby("org").pjt_no.nunique()
    lp = Lc[Lc.year.between(v - 3, v - 1) & (Lc.appl_year <= v - 1)].groupby("org").pjt_no.nunique()
    X["org_kr_pat"] = np.log1p(X.org.map(ra).fillna(0)); X["org_kr_rate"] = X.org.map(rp).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
    X["org_p50"] = np.log1p(X.org.map(la).fillna(0)); X["org_p50_rate"] = X.org.map(lp).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
    X["org_p50_share"] = X.org.map(la).fillna(0) / (X.org.map(ra).fillna(0) + 1)
    # 선행과제(pre_pjt) 성과(출원연도 ≤ t−1)
    rpre = Rc[Rc.appl_year <= v - 1].groupby("ProjectID").n.sum(); lpre = Lc[Lc.appl_year <= v - 1].groupby("ProjectID").n.sum()
    X["pre_kr_pat"] = np.log1p(X.pre_pjt.map(rpre).fillna(0)) * X.has_pre
    X["pre_p50"] = np.log1p(X.pre_pjt.map(lpre).fillna(0)) * X.has_pre
    # 민감도용(사후): 성과 보고 지연 1년 가정 — 출원연도 ≤ t−2 성과만 쓴 이력
    ra2 = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= v - 2)].groupby("org").n.sum()
    la2 = Lc[Lc.year.between(v - 3, v - 1) & (Lc.appl_year <= v - 2)].groupby("org").n.sum()
    rp2 = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= v - 2)].groupby("org").pjt_no.nunique()
    lp2 = Lc[Lc.year.between(v - 3, v - 1) & (Lc.appl_year <= v - 2)].groupby("org").pjt_no.nunique()
    X["org_kr_pat_L2"] = np.log1p(X.org.map(ra2).fillna(0)); X["org_kr_rate_L2"] = X.org.map(rp2).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
    X["org_p50_L2"] = np.log1p(X.org.map(la2).fillna(0)); X["org_p50_rate_L2"] = X.org.map(lp2).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
    X["org_p50_share_L2"] = X.org.map(la2).fillna(0) / (X.org.map(ra2).fillna(0) + 1)
    rpre2 = Rc[Rc.appl_year <= v - 2].groupby("ProjectID").n.sum(); lpre2 = Lc[Lc.appl_year <= v - 2].groupby("ProjectID").n.sum()
    X["pre_kr_pat_L2"] = np.log1p(X.pre_pjt.map(rpre2).fillna(0)) * X.has_pre
    X["pre_p50_L2"] = np.log1p(X.pre_pjt.map(lpre2).fillna(0)) * X.has_pre
    X["t0"] = v
    out.append(X)
    log(v, len(X), int(X.y.sum()), round(X.y.mean(), 4), "firm", round(X[X.firm == 1].y.mean(), 4))
D = pd.concat(out, ignore_index=True)
keep = ["pjt_no", "t0", "y", "org", "firm", "log_gov", "log_tot", "nongov_sh", "sbiz_sh", "perform_grp_c", "dev_phase_c", "cont", "series_yr", "total_len", "period_m", "end_in_year", "ministry_c", "has_pre",
        "org_nproj", "org_kr_pat", "org_kr_rate", "pre_kr_pat",
        "b1_tfidf_score", "b1_tfidf_assigned", "b2_dict_score", "b2_dict_assigned", "sixt_cd_c", "cls_L_canon_c", "econ_goal_cd_c", "tech_lifecycle_c", "apply1_cd_c",
        "org_p50", "org_p50_rate", "org_p50_share", "pre_p50", "b1_tfidf_top_ksic", "perform_grp",
        "org_kr_pat_L2", "org_kr_rate_L2", "org_p50_L2", "org_p50_rate_L2", "org_p50_share_L2", "pre_kr_pat_L2", "pre_p50_L2"]
D = D[keep]
for cn in D.columns:
    if D[cn].dtype == "float64":
        D[cn] = D[cn].astype("float32")
D.to_parquet(f"{CACHE}/J1_panel.parquet", index=False)
log("saved", len(D))
