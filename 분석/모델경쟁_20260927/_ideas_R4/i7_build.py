# -*- coding: utf-8 -*-
"""I7 특허기업의 정부R&D 첫 주관(수행) 진입 예측: 특허 행태가 '정부 R&D 신규 수행기업'을 예고하는가.
단위 = 국내법인 출원인코드(t0까지 제공범위 KR 등록 1건 이상, t0까지 폐업 아님 — R3 I2b 모집단을 그대로 읽어 씀)
       중 사업자번호가 NTIS 과제(2010~t0) 수행기관 사업자번호에 한 번도 나오지 않은 기업.
결과 y = NTIS 과제연도 t0+1 또는 t0+2에 그 사업자번호가 수행기관으로 처음 등장.
[봉인] 최종시험 t0=2022 -> 과제연도 2023–2024 (F=2023). 과제연도 >= 2023은 읽지 않는다. 개발 원점 t0 <= 2020.
피처: M0 = NPS 외부 기준선(R3 I2b와 동일), M1 = +특허 행태(등록원부·제공 KR, R3 I2b와 동일 정의),
      M2 = +NTIS 성과 경유 '참여' 신호(자사 특허가 다른 기관 주관 정부과제 성과로 등록됨, 레코드 Year <= t0, 과제연도 <= t0),
      M3 = +KSIC 수준 정부연구비 성장(자사 특허 KSIC 구성 가중, a_link_pit avail_year <= t0) — 교훈 재확인용."""
import time, numpy as np, pandas as pd, duckdb
from r4_common import *
T = time.time()


def log(*a):
    print(*a, round(time.time() - T), flush=True)


P = pd.read_parquet(f"{R3I}/cache/I2b_panel.parquet")
P = P[P.t0 <= 2020].drop(columns=["y", "end", "x_s_kr", "x_s_fr", "x_s_sci", "x_s_ppi"])   # 폐업 결과·KSIC 신호는 쓰지 않음
ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno"]).drop_duplicates("applicantCd")
P = P.merge(ns, on="applicantCd", how="left")
EXTF = EXT.replace("\\", "/")
NT = duckdb.sql(f"""select regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, year, perform_grp, gov_funds
 from read_parquet('{EXTF}/ntis_project_raw.parquet') where year >= 2010 and year < {SEAL_YEAR} and corp_rgst_no is not null""").df()
assert NT.year.max() < 2023
first = NT.groupby("bz").year.min()
P["first_ntis"] = P.bizno.map(first)
log("panel", len(P), "bizno", round(P.bizno.notna().mean(), 3))
# 모집단: t0까지 수행 이력 없음
P = P[~(P.first_ntis <= P.t0)].copy()
P["y"] = P.first_ntis.between(P.t0 + 1, P.t0 + 2).astype(int)
assert (P.t0 + 2).max() <= 2022
# M2: 자사 특허의 NTIS 성과 연결(참여 신호)
AF = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_no", "known_date", "ksic"])
AF = AF[AF.applicantCd.isin(set(P.applicantCd)) & AF.known_date.notna()]
lk = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "Year", "ProjectYear", "perform_grp", "ProjectID"])
lk["Year"] = pd.to_numeric(lk.Year, errors="coerce"); lk["ProjectYear"] = pd.to_numeric(lk.ProjectYear, errors="coerce")
A = AF.merge(lk, on="appl_no", how="inner")
log("firm patents linked to NTIS", A.appl_no.nunique(), "of", AF.appl_no.nunique())
KY = pd.read_parquet(f"{EXT}/ntis_project_ksic_year.parquet"); KY = KY[KY.method == "a_link_pit"]
parts = []
for t0, Pt in P.groupby("t0"):
    end = t0 * 10000 + 1231
    a = A[(A.known_date <= end) & (A.Year <= t0) & (A.ProjectYear <= t0)]
    ga = a.groupby("applicantCd")
    f = pd.DataFrame({"g_part_n": ga.appl_no.nunique(), "g_part_proj": ga.ProjectID.nunique(),
                      "g_part_univ": ga.perform_grp.apply(lambda s: (s == "대학").any()),
                      "g_part_gri": ga.perform_grp.apply(lambda s: s.isin(["출연연", "국공립연"]).any()),
                      "g_part_firm": ga.perform_grp.apply(lambda s: (s == "기업").any()),
                      "g_part_rec": t0 - ga.ProjectYear.max()})
    Pt = Pt.merge(f, left_on="applicantCd", right_index=True, how="left")
    Pt["g_part_any"] = Pt.g_part_n.notna().astype(float)
    for c in ["g_part_n", "g_part_proj"]:
        Pt[c] = np.log1p(Pt[c].fillna(0))
    for c in ["g_part_univ", "g_part_gri", "g_part_firm"]:
        Pt[c] = Pt[c].astype(float).fillna(0)
    kn = AF[AF.known_date <= end]
    Pt["g_part_share"] = Pt.g_part_n.apply(np.expm1) / Pt.applicantCd.map(kn.groupby("applicantCd").appl_no.nunique()).fillna(1)
    # M3: KSIC 구성 × KSIC 정부연구비 성장·비중
    ky = KY[KY.avail_year <= t0]
    pv = ky.pivot_table(index="ksic", columns="year", values="gov_funds", aggfunc="sum")
    a_ = pv[[y for y in pv.columns if t0 - 2 <= y <= t0 - 1]].sum(1); b_ = pv[[y for y in pv.columns if t0 - 4 <= y <= t0 - 3]].sum(1)
    gr = np.log((a_ + 1) / (b_ + 1)); gr[b_ <= 0] = np.nan
    sh = ky[ky.year == t0 - 1].groupby("ksic").gov_share_of_year.sum()
    W = kn[kn.applicantCd.isin(set(Pt.applicantCd))].groupby(["applicantCd", "ksic"]).appl_no.nunique().rename("w").reset_index()
    W["w"] = W.w / W.groupby("applicantCd").w.transform("sum")
    W["gr"] = W.ksic.map(gr); W["sh"] = W.ksic.map(sh)
    for c, nmc in [("gr", "k_rnd_g"), ("sh", "k_rnd_sh")]:
        ww = W.dropna(subset=[c]); v = (ww.w * ww[c]).groupby(ww.applicantCd).sum() / ww.groupby("applicantCd").w.sum()
        Pt[nmc] = Pt.applicantCd.map(v)
    parts.append(Pt)
    log(t0, len(Pt), "y", int(Pt.y.sum()), round(Pt.y.mean(), 4), "part_any", round(Pt.g_part_any.mean(), 3))
D = pd.concat(parts, ignore_index=True)
D.to_parquet(f"{CACHE}/I7_panel.parquet", index=False)
log("done", D.shape)
