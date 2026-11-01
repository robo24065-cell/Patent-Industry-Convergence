# -*- coding: utf-8 -*-
"""J3 특허기업 피합병(M&A 표적) 레이더 — 학습표.
단위 = R3 I2b 모집단(국내법인 출원인코드, t0까지 제공범위 KR 등록 ≥1, t0 말까지 국세청 폐업 아님)을 읽기 전용으로 재사용, t0 2015–2020.
결과 = (t0, t0+2] 안에 그 기업이 권리자였던 등록특허가 '법인합병'(또는 '합병') 사유로 다른 권리자에게 넘어감
       (등록원부 소유자 이력에서 합병 이전 직전 권리자의 정규화 상호 = 기업 상호).
🔒 최종시험 t0=2022 → 합병 이전등록 2023-01-01~2024-12-31 (F=2023). 소유자 이력은 _ideas_R4 봉인 캐시(< 2023-01-01)만. 개발 t0 ≤ 2020.
묶음(사전 선언)
 M0 외부 기준선(NPS 가입·규모·증감·업력·취득/상실률·임금, 사업자번호 유형) — I2b M0과 같음
 M1 특허 행태(I2b 정의: 출원 추세·등록료 납부·불납·유지·이전)
 M2 공동권리·합병 관계(등록원부: 다른 법인과 공동 보유 비중·공동 법인 수·최대 공동 법인 포트폴리오 규모·과거 인수자 경험)
 M3 NTIS 정부R&D 수행(사업자번호: 최근 3년 과제 수·연구비·수행 여부)
"""
import time, duckdb
import numpy as np, pandas as pd
from r5_common import *

t = time.time()
def log(*a): print(*a, round(time.time() - t), flush=True)

D = pd.read_parquet(f"{R3I}/cache/I2b_panel.parquet")
D = D[D.t0 <= 2020].drop(columns=["y", "x_s_kr", "x_s_fr", "x_s_sci", "x_s_ppi"]).copy()
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname", "end_dt"]).drop_duplicates("applicantCd")
D = D.merge(ns[["applicantCd", "bizno", "aname", "end_dt"]], on="applicantCd", how="left")
c = ro()
ap = pd.read_sql("select rgst_no, applicantCd, applicantName from applicant", c)
names = ap.groupby("applicantCd").applicantName.agg(lambda s: set(s.dropna()))
log("panel", len(D))

ow = pd.read_parquet(f"{R4C}/reg_owner_lt2023.parquet"); ow["d"] = pd.to_numeric(ow.d, errors="coerce")
assert ow.d.dropna().max() < SEAL_DATE_I
ow = ow.sort_values(["rgst_no", "seq"]).reset_index(drop=True)
ow["prev_name"] = ow.groupby("rgst_no").ownerName.shift(1)
mg = ow[ow.cs.isin(["법인합병", "합병"]) & ow.prev_name.notna()][["rgst_no", "d", "prev_name", "ownerName"]].copy()
mg["pn"] = mg.prev_name.map(norm_name); mg["nn"] = mg.ownerName.map(norm_name)
mg = mg[(mg.pn.str.len() >= 2) & (mg.pn != mg.nn)]
log("merger transfers", len(mg), mg.pn.nunique())

# 공동출원인(등록 시점) 관계: 기업 f의 등록특허 중 다른 법인(코드 1…)과 공동인 비중
pt = pd.read_parquet(f"{R4C}/reg_patent.parquet", columns=["rgst_no", "rgstDate"]); pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce")
apc = ap[["rgst_no", "applicantCd"]].drop_duplicates().merge(pt[["rgst_no", "rg"]], on="rgst_no")
pairs = apc.merge(apc, on=["rgst_no", "rg"]); pairs = pairs[(pairs.applicantCd_x != pairs.applicantCd_y) & (pairs.applicantCd_y.astype(str).str[0] == "1")]
port = apc.groupby("applicantCd").rgst_no.nunique()
log("co-applicant pairs", len(pairs))

NT = duckdb.sql(f"""select year, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, gov_funds from read_parquet('{EXT}/ntis_project_raw.parquet')
 where year between 2010 and 2020""").df()
NT = NT[NT.bz.str.len() == 10]

out = []
for t0 in range(2015, 2021):
    E = t0 * 10000 + 1231; W2 = (t0 + 2) * 10000 + 1231
    assert W2 < SEAL_DATE_I
    X = D[D.t0 == t0].copy()
    ev = set(mg[(mg.d > E) & (mg.d <= W2)].pn)
    X["names"] = [({norm_name(x) for x in names.get(a, set())} | ({norm_name(b)} if isinstance(b, str) else set())) - {""} for a, b in zip(X.applicantCd, X.aname)]
    X["y"] = X.names.map(lambda s: int(len(s & ev) > 0))
    # 합병 이력(과거): 이 기업이 과거에 다른 회사를 흡수한 적(인수자 경험)
    pastacq = set(mg[mg.d <= E].nn)
    X["past_acquirer"] = X.names.map(lambda s: int(len(s & pastacq) > 0))
    # M2 공동권리
    pp = pairs[pairs.rg <= E]
    g = pp.groupby("applicantCd_x")
    co_n = g.applicantCd_y.nunique(); co_pat = g.rgst_no.nunique()
    own = apc[apc.rg <= E].groupby("applicantCd").rgst_no.nunique()
    portE = apc[apc.rg <= E].groupby("applicantCd").rgst_no.nunique()
    top = pp.assign(ps=pp.applicantCd_y.map(portE)).groupby("applicantCd_x").ps.max()
    X["co_corp_n"] = np.log1p(X.applicantCd.map(co_n).fillna(0))
    X["co_corp_share"] = X.applicantCd.map(co_pat).fillna(0) / X.applicantCd.map(own).fillna(1).clip(lower=1)
    X["co_top_port"] = np.log1p(X.applicantCd.map(top).fillna(0))
    X["co_top_ratio"] = np.log1p(X.applicantCd.map(top).fillna(0)) - np.log1p(X.applicantCd.map(own).fillna(0))
    # M3 NTIS
    nt = NT[NT.year.between(t0 - 2, t0)].groupby("bz").agg(n=("year", "size"), f=("gov_funds", "sum"))
    X["ntis_n"] = np.log1p(X.bizno.map(nt.n).fillna(0)); X["ntis_funds"] = np.log1p(X.bizno.map(nt.f).fillna(0) / 1e6)
    X["ntis_any"] = (X.ntis_n > 0).astype(int)
    X["closed_win"] = ((X.end > E) & (X.end <= W2)).astype(int)  # 사후 점검용(피처 아님)
    out.append(X.drop(columns=["names"]))
    log(t0, len(X), int(X.y.sum()), round(X.y.mean() * 100, 3), "% | 합병기업 중 국세청 폐업(창 안)", round(X[X.y == 1].closed_win.mean(), 3))
P = pd.concat(out, ignore_index=True)
for cn in P.columns:
    if P[cn].dtype == "float64":
        P[cn] = P[cn].astype("float32")
P.drop(columns=["end_dt"]).to_parquet(f"{CACHE}/J3_panel.parquet", index=False)
log("saved", len(P))
