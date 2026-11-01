# -*- coding: utf-8 -*-
"""J2 특허 분쟁 표적 레이더 — 학습표 생성.
단위 = (등록특허·실용신안, t0): t0 말까지 등록, t0 말까지 소멸·말소 기록 없음, 존속만료 > t0 말.
결과 = (t0, t0+2] 안에 무효(심판)예고등록 또는 권리범위(심판)예고등록(제3자·권리자 분쟁의 공시) 1건 이상.
🔒 최종시험 t0=2022 → 2023-01-01~2024-12-31 (F=2023). 등록원부 사건은 _ideas_R4 봉인 캐시(날짜 < 2023-01-01)만 읽는다.
개발 원점 t0 2008–2020(결과창 끝 ≤ 2022-12-31).
묶음(사전 선언)
 M0 특허 고유 속성(연령·잔존·청구항·IPC 수·발명자 수·실용신안·출원인 유형·KSIC·IPC 섹션·납부 연차)
 M1 등록원부·제공데이터 이력(자기 과거 분쟁·이전·실시권·기술평가·권리자 포트폴리오·권리자 과거 분쟁·메인그룹/KSIC 분쟁률·시점 정합 해외 패밀리)
 M2 외부 단위 데이터(RoS: 미국 등록 패밀리·논문 인용 / KPSS 가치 / ETSI 표준 선언 / NTIS 정부R&D 기원·수행주체)
"""
import re, time, duckdb
import numpy as np, pandas as pd
from r5_common import *

t = time.time()
def log(*a): print(*a, round(time.time() - t), flush=True)

DISP = ["무효(심판)예고등록", "권리범위(심판)예고등록"]
OTHER_TRIAL = ["심판예고등록", "정정(심판)예고등록", "취소신청예고등록", "이의신청예고등록", "심결취소소송예고등록", "취소(심판)예고등록", "상고예고등록"]
pt = pd.read_parquet(f"{R4C}/reg_patent.parquet")
rt = pd.read_parquet(f"{R4C}/reg_right_lt2023.parquet"); assert rt.d.max() < SEAL_DATE
ow = pd.read_parquet(f"{R4C}/reg_owner_lt2023.parquet"); assert ow.d.dropna().max() < SEAL_DATE
py = pd.read_parquet(f"{R4C}/reg_pay_lt2023.parquet"); assert py.d.max() < SEAL_DATE
lc = pd.read_parquet(f"{R4C}/reg_lic_lt2023.parquet"); assert lc.d.max() < SEAL_DATE
ni = pd.read_parquet(f"{R4C}/reg_ninv.parquet")
for df in (rt, ow, py, lc):
    df["d"] = pd.to_numeric(df.d, errors="coerce")
pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce"); pt["ap"] = pd.to_numeric(pt.applDate, errors="coerce")
pt["ex"] = pd.to_numeric(pt.cndrtExptnDate, errors="coerce"); pt["claims"] = pd.to_numeric(pt.claimCount, errors="coerce")
pt["n_ipc"] = pt.ipcCd.fillna("").str.count(r"\(") .clip(lower=1)
_m = pt.ipcCd.fillna("").str.split(",").str[0].str.strip().str.extract(r"^([A-H]\d\d[A-Z])\s*(\d+)/")
pt["mg"] = (_m[0] + _m[1]).fillna("NA")
pt["ipc_sec"] = pt.ipcCd.fillna("Z").str[0].map(lambda c: "ABCDEFGHZ".find(c))
pt["utility"] = (pt.rgst_no.str[:2] == "20").astype(int)
pt["ksic_i"] = pd.to_numeric(pt.ksic, errors="coerce")
pt = pt.merge(ni, on="rgst_no", how="left")
log("load", len(pt))

c = ro()
ap = pd.read_sql("select rgst_no, seq, applicantCd, applicantName, applicantNatl from applicant", c)
ap0 = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no")
nap = ap.groupby("rgst_no").size().rename("n_appl")
pt = pt.merge(ap0[["rgst_no", "applicantCd", "applicantNatl"]], on="rgst_no", how="left").merge(nap, on="rgst_no", how="left")
pt["atype"] = pd.to_numeric(pt.applicantCd.astype(str).str[0], errors="coerce")
pt["foreign"] = (pt.applicantNatl.fillna("대한민국") != "대한민국").astype(int)
log("applicant")

# 소멸·말소 첫 날짜
ext_d = rt[rt.cs.isin(["소멸등록", "권리의 말소등록", "말소등록"])].groupby("rgst_no").d.min()
pt["ext_d"] = pt.rgst_no.map(ext_d)
disp = rt[rt.cs.isin(DISP)][["rgst_no", "d"]]
othr = rt[rt.cs.isin(OTHER_TRIAL)][["rgst_no", "d"]]
teval = rt[rt.cs.str.contains("기술평가", na=False)][["rgst_no", "d"]]
trf = ow[ow.cs.notna() & (ow.seq > 0)][["rgst_no", "d", "cs"]]
# 권리자 이름(정규화)
uniq = pd.Series(ow.ownerName.dropna().unique())
nm = dict(zip(uniq, uniq.map(norm_name)))
ow["on"] = ow.ownerName.map(nm)
ow = ow.sort_values(["rgst_no", "seq"])

# 시점 정합 해외 패밀리: 제공 KR 메타 family 토큰
fam = duckdb.sql(f"select appl_no, any_value(family) fam from read_parquet('{META}') where cc='KR' group by appl_no").df()
def fam_feats(s):
    toks = set(s.split("|")) if isinstance(s, str) else set()
    out = []
    for tk in toks:
        m = re.match(r"^(WO|US|JP)(\d{4})\d+", tk)
        if m and 1978 <= int(m.group(2)) <= 2030:
            out.append((m.group(1), int(m.group(2))))
        elif tk[:2] in ("EP", "CN"):
            out.append((tk[:2], None))
    return out
fam["toks"] = fam.fam.map(fam_feats)
fam = fam[fam.toks.map(len) > 0]
rows = [(a, o, y) for a, L in zip(fam.appl_no, fam.toks) for o, y in L]
FT = pd.DataFrame(rows, columns=["applNo", "office", "yr"]).drop_duplicates()
FT["ay"] = FT.applNo.map(dict(zip(pt.applNo, pt.ap // 10000)))
log("family tokens", len(FT))

# 외부: RoS(미국 패밀리·논문 인용), KPSS, ETSI, NTIS
ros = pd.read_parquet(f"{EXT}/r3ros_rec_sci.parquet", columns=["cc", "appl_no", "n_us", "n_us_sci", "n_paper_cit", "first_issue_year"], filters=[("cc", "=", "KR")])
ros = ros.groupby("appl_no").agg(n_us=("n_us", "max"), n_us_sci=("n_us_sci", "max"), n_paper_cit=("n_paper_cit", "max"), fiy=("first_issue_year", "min")).reset_index().rename(columns={"appl_no": "applNo"})
kp = pd.read_parquet(f"{EXT}/quality_kpss_family_link.parquet", columns=["cc", "appl_no", "issue_date", "xi_real", "cites"], filters=[("cc", "=", "KR")])
kp["iy"] = pd.to_datetime(kp.issue_date, errors="coerce").dt.year
kp = kp.rename(columns={"appl_no": "applNo"})
et = pd.read_parquet(f"{EXT}/quality_etsi_sep_link.parquet"); et = et[et.cc == "KR"].copy()
et["dy"] = pd.to_datetime(et.first_decl, errors="coerce").dt.year; et = et.groupby("appl_no").dy.min().rename("etsi_y").reset_index().rename(columns={"appl_no": "applNo"})
nl = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "Year", "perform_grp"])
nl["Year"] = pd.to_numeric(nl.Year, errors="coerce"); nl = nl.rename(columns={"appl_no": "applNo"})
log("external", len(ros), len(kp), len(et), len(nl))

out = []
for t0 in range(2008, 2021):
    E = t0 * 10000 + 1231; W2 = (t0 + 2) * 10000 + 1231
    assert W2 < SEAL_DATE_I
    P = pt[(pt.rg <= E) & ~(pt.ext_d <= E) & (pt.ex > E)].copy()
    ids = P.rgst_no
    P["age_rg"] = t0 - P.rg // 10000; P["age_ap"] = t0 - P.ap // 10000; P["rem_life"] = (P.ex // 10000) - t0
    pp = py[py.d <= E]
    P["last_annl"] = P.rgst_no.map(pp.groupby("rgst_no").lastAnnl.max().astype(float) if len(pp) else None)
    P["paid_ahead"] = P.last_annl - (t0 - P.rg // 10000 + 1)
    # 결과
    dd = disp[(disp.d > E) & (disp.d <= W2)]
    P["y"] = P.rgst_no.isin(set(dd.rgst_no)).astype(int)
    # M1 이력
    P["own_disp"] = P.rgst_no.map(disp[disp.d <= E].groupby("rgst_no").size()).fillna(0)
    P["own_trial"] = P.rgst_no.map(othr[othr.d <= E].groupby("rgst_no").size()).fillna(0)
    P["tech_eval"] = P.rgst_no.isin(set(teval[teval.d <= E].rgst_no)).astype(int)
    tt = trf[trf.d <= E]
    P["n_trf"] = P.rgst_no.map(tt.groupby("rgst_no").size()).fillna(0)
    P["trf_2y"] = P.rgst_no.map(tt[tt.d > E - 20000].groupby("rgst_no").size()).fillna(0)
    P["merger_trf"] = P.rgst_no.map(tt[tt.cs.isin(["법인합병", "합병"])].groupby("rgst_no").size()).fillna(0)
    P["lic"] = P.rgst_no.map(lc[lc.d <= E].groupby("rgst_no").size()).fillna(0)
    cur = ow[(ow.d.isna()) | (ow.d <= E)].drop_duplicates("rgst_no", keep="last").set_index("rgst_no").on
    P["owner"] = P.rgst_no.map(cur).fillna("")
    live_by_owner = P.groupby("owner").size()
    P["owner_live"] = np.log1p(P.owner.map(live_by_owner).where(P.owner != "", 1))
    # 권리자 과거 5년 분쟁(자기 특허가 분쟁 대상이 된 횟수, 권리자는 t0 현재 기준)
    d5 = disp[(disp.d > E - 50000) & (disp.d <= E)].rgst_no
    P["owner_disp5"] = P.owner.map(P[P.rgst_no.isin(set(d5))].groupby("owner").size()).fillna(0).where(P.owner != "", 0)
    alive5 = pt[(pt.rg <= E) & ~(pt.ext_d <= E - 50000)]
    dis5 = alive5.rgst_no.isin(set(d5))
    for key, col in (("mg", "mg_disp5"), ("ksic", "ksic_disp5")):
        g = pd.DataFrame({"k": alive5[key].values, "d": dis5.values}).groupby("k").d.agg(["sum", "size"])
        rate = (g["sum"] + 0.5) / (g["size"] + 500)
        P[col] = np.log(P[key].map(rate).astype(float))
    # 시점 정합 해외 패밀리
    ok = FT[(FT.yr <= t0) | (FT.yr.isna() & (FT.ay <= t0 - 4))]
    fo = ok.groupby("applNo").office.nunique()
    P["fam_offices"] = P.applNo.map(fo).fillna(0)
    P["fam_us"] = P.applNo.isin(set(ok[ok.office == "US"].applNo)).astype(int)
    # M2 외부
    rr = ros[ros.fiy <= t0].set_index("applNo")
    P["us_granted"] = P.applNo.isin(rr.index).astype(int)
    P["us_sci"] = P.applNo.map(rr.n_us_sci).fillna(0)
    P["us_papercit"] = np.log1p(P.applNo.map(rr.n_paper_cit).fillna(0))
    kk = kp[kp.iy <= t0].groupby("applNo").agg(xi=("xi_real", "max"), kcites=("cites", "max"))
    P["kpss_xi"] = np.log1p(P.applNo.map(kk.xi))
    P["etsi"] = P.applNo.isin(set(et[et.etsi_y <= t0].applNo)).astype(int)
    ng = nl[nl.Year <= t0]
    P["gov"] = P.applNo.isin(set(ng.applNo)).astype(int)
    for gname, col in (("기업", "gov_firm"), ("대학", "gov_univ"), ("출연연", "gov_gri")):
        P[col] = P.applNo.isin(set(ng[ng.perform_grp.astype(str).str.contains(gname)].applNo)).astype(int)
    P["t0"] = t0
    keep = ["rgst_no", "applNo", "t0", "y", "ksic", "owner", "age_rg", "age_ap", "rem_life", "claims", "n_ipc", "n_inv", "utility", "atype", "foreign", "n_appl", "ksic_i", "ipc_sec", "last_annl", "paid_ahead",
            "own_disp", "own_trial", "tech_eval", "n_trf", "trf_2y", "merger_trf", "lic", "owner_live", "owner_disp5", "mg_disp5", "ksic_disp5", "fam_offices", "fam_us",
            "us_granted", "us_sci", "us_papercit", "kpss_xi", "etsi", "gov", "gov_firm", "gov_univ", "gov_gri"]
    out.append(P[keep])
    log(t0, len(P), int(P.y.sum()), round(P.y.mean() * 1000, 3), "‰", "gov", round(P.gov.mean(), 3), "fam", round((P.fam_offices > 0).mean(), 3))
D = pd.concat(out, ignore_index=True)
for cn in D.columns:
    if D[cn].dtype == "float64":
        D[cn] = D[cn].astype("float32")
D.to_parquet(f"{CACHE}/J2_panel.parquet", index=False)
log("saved", len(D))
