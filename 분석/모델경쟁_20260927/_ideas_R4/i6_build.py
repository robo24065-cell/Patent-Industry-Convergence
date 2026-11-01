# -*- coding: utf-8 -*-
"""I6 공공기술 사업화 레이더: 대학·출연연 보유 KR 등록특허(제공 50 KSIC) 중 어떤 특허가 2년 안에 기업에 이전되거나 실시권이 설정되나.
단위 = (특허, t0). t0 말에 살아 있고 그 시점 권리자 묶음에 국내 공공기관(대학·산학협력단·출연연 등)이 있는 특허.
결과 y = (t0말, t0+2년말] 안에 (1) 기업(법인명)이 새 권리자로 들어오는 이전등록 또는 (2) 전용·통상실시권 설정(시작일) 1건 이상.
[봉인] 최종시험 t0=2022 -> 결과 2023-01-01~2024-12-31 (F=2023). 여기서는 2023 이후 사건을 아예 읽지 않는다(캐시가 < 2023-01-01).
피처는 t0 말까지의 정보만: 등록원부(날짜 <= t0말), NTIS(레코드 Year <= t0, 과제연도 <= t0), 제공 KR(공개·등록일 <= t0말)."""
import re, time, numpy as np, pandas as pd, duckdb
from r4_common import *
T = time.time()


def log(*a):
    print(*a, round(time.time() - T), flush=True)


PUB = re.compile(r"대학교|산학협력단|산학협력|과학기술원|대학원|연구원|연구소|연구재단|진흥원|대학|국방과학", re.I)
FIRM = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임|합자회사|합명회사|CO\.|LTD|INC\b|INC\.|CORP|GMBH|주\)|\(유\)|\(株\)|株式会社|삼성|현대|엘지|LG|SK|독립행정법인|中國|중국|일본|유겐|협동조합", re.I)
pt = pd.read_parquet(f"{CACHE}/reg_patent.parquet")
pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce"); pt["ap"] = pd.to_numeric(pt.applDate, errors="coerce")
pt["ex"] = pd.to_numeric(pt.cndrtExptnDate, errors="coerce").fillna(pt.ap + 200000)
pt["claims"] = pd.to_numeric(pt.claimCount, errors="coerce")
ipc1 = pt.ipcCd.fillna("").str.split(",").str[0].str.replace(r"\(.*\)", "", regex=True).str.replace(" ", "")
pt["mg"] = ipc1.str.split("/").str[0]; pt["ipc_sec"] = ipc1.str[:1].map({c: i for i, c in enumerate("ABCDEFGH")})
pt["n_ipc"] = pt.ipcCd.fillna("").str.count(",") + 1
pt["ksic_i"] = pd.to_numeric(pt.ksic, errors="coerce"); pt["ksic2"] = pt.ksic_i // 1000
pt = pt.merge(pd.read_parquet(f"{CACHE}/reg_ninv.parquet"), on="rgst_no", how="left")
ow = pd.read_parquet(f"{CACHE}/reg_owner_lt2023.parquet"); ow["d"] = pd.to_numeric(ow.d, errors="coerce")
assert ow.d.max() < 20230101
nm = ow.ownerName.fillna("")
ow["pub"] = nm.str.contains(PUB) & ~nm.str.contains(FIRM) & ow.ownerNatl.fillna("대한민국").str.contains("대한민국|KR|한국")
ow.loc[nm.str.contains("포항산업과학연구원"), "pub"] = False  # 포스코 계열 재단(기업 부설 성격) — 공공에서 제외(사전 규칙)
ow["firm"] = nm.str.contains(FIRM) & ~ow.pub
ow["key"] = nm.str.replace(r"\s|재단법인|\(재\)", "", regex=True)
rt = pd.read_parquet(f"{CACHE}/reg_right_lt2023.parquet"); rt["d"] = pd.to_numeric(rt.d, errors="coerce"); assert rt.d.max() < 20230101
lic = pd.read_parquet(f"{CACHE}/reg_lic_lt2023.parquet"); lic["d"] = pd.to_numeric(lic.d, errors="coerce"); assert lic.d.max() < 20230101
pay = pd.read_parquet(f"{CACHE}/reg_pay_lt2023.parquet"); pay["d"] = pd.to_numeric(pay.d, errors="coerce"); pay["annl"] = pd.to_numeric(pay.lastAnnl, errors="coerce")
log("load")
# 그룹(특허, 날짜) 요약
g = ow.groupby(["rgst_no", "d"]).agg(pub=("pub", "any"), firm=("firm", "any"), n=("seq", "size"), cs=("cs", "first")).reset_index()
first_d = g.groupby("rgst_no").d.transform("min"); g["is_init"] = g.d == first_d
pkey = ow[ow.pub].sort_values("seq").groupby(["rgst_no", "d"]).key.first()
g = g.merge(pkey.rename("pkey").reset_index(), on=["rgst_no", "d"], how="left")
g = g.sort_values(["rgst_no", "d"]); g["prev_pub"] = g.groupby("rgst_no").pub.shift(); g["prev_pkey"] = g.groupby("rgst_no").pkey.shift()
# 사업화 사건: 기업이 들어오는 이전(초기 묶음 제외) + 실시권 시작
fname = ow[ow.firm].sort_values("seq").groupby(["rgst_no", "d"]).ownerName.first().rename("fname").reset_index()
ev_tr = g[(~g.is_init) & g.firm & (g.prev_pub == True)][["rgst_no", "d", "prev_pkey"]].rename(columns={"prev_pkey": "okey"}); ev_tr["typ"] = "T"
ev_tr = ev_tr.merge(fname, on=["rgst_no", "d"], how="left")
# 대량 거래(같은 날 같은 양수인에게 같은 기관 특허 20건 이상)는 '개별 기술이전'이 아니라 조직 재편·일괄 양도로 보고 표시(사전 규칙)
ev_tr["deal_n"] = ev_tr.groupby(["d", "fname", "okey"]).rgst_no.transform("nunique"); ev_tr["bulk"] = ev_tr.deal_n >= 20
L = lic[["rgst_no", "d", "typ"]].dropna().sort_values("d")
gp = g[g.pub][["rgst_no", "d", "pkey"]].sort_values("d").rename(columns={"d": "gd"})
L = pd.merge_asof(L, gp, left_on="d", right_on="gd", by="rgst_no", direction="backward")
ev_li = L[L.pkey.notna()][["rgst_no", "d", "pkey", "typ"]].rename(columns={"pkey": "okey"}); ev_li["bulk"] = False
EV = pd.concat([ev_tr, ev_li], ignore_index=True); assert EV.d.max() < 20230101
log("events(<2023) T", len(ev_tr), "L", len(ev_li))
EV[["rgst_no", "d", "okey", "typ", "bulk"]].to_parquet(f"{CACHE}/I6_events_lt2023.parquet", index=False)
st = rt[rt.cs.isin(["소멸등록", "회복등록"])][["rgst_no", "d", "cs"]].sort_values("d")
# NTIS
lk = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "rgst_no", "Year", "ProjectID", "ProjectYear", "perform_grp", "MinistryName"])
lk["Year"] = pd.to_numeric(lk.Year, errors="coerce"); lk["ProjectYear"] = pd.to_numeric(lk.ProjectYear, errors="coerce")
a2r = dict(zip(pt.applNo, pt.rgst_no)); lk["rg_no"] = lk.rgst_no.fillna(lk.appl_no.map(a2r))
lk = lk[lk.rg_no.isin(set(pt.rgst_no))]
EXTF = EXT.replace("\\", "/")
PJ = duckdb.sql(f"""select pjt_no, year, dev_phase, gov_funds, sbiz_funds, nongov_funds, total_end
 from read_parquet('{EXTF}/ntis_project_raw.parquet') where year < {SEAL_YEAR}""").df()
PJ["tend"] = pd.to_numeric(PJ.total_end.astype(str).str[:4], errors="coerce")
PJ = PJ.sort_values("year")
lk = lk.merge(PJ[["pjt_no", "year", "dev_phase", "gov_funds", "sbiz_funds", "nongov_funds", "tend"]].drop_duplicates(["pjt_no", "year"]),
              left_on=["ProjectID", "ProjectYear"], right_on=["pjt_no", "year"], how="left")
IND = re.compile(r"산업|지식경제|중소|통상자원|국토|해양|농림|환경|보건|방위|국방")
lk["m_ind"] = lk.MinistryName.fillna("").str.contains(IND).astype(float)
log("ntis", len(lk), "project matched", round(lk.pjt_no.notna().mean(), 3))
# 셀 수요(M2): 제공 KR 출원 중 기업(법인명·비공공) 출원 수
MK = pd.read_parquet(f"{CACHE}/meta_kr.parquet")
an = MK.applicant.fillna("")
MK["firm"] = an.str.contains(FIRM) & ~an.str.contains(PUB)
MK = MK.drop_duplicates(["appl_no", "ksic"]); MK["ksic_i"] = pd.to_numeric(MK.ksic, errors="coerce")
KY = pd.read_parquet(f"{EXT}/ntis_project_ksic_year.parquet")
log("meta")
rows = []
for t0 in range(2011, 2021):
    end = t0 * 10000 + 1231; end2 = (t0 + 2) * 10000 + 1231; assert end2 < 20230101
    P = pt[(pt.rg <= end) & (pt.ex > end)].copy()
    s = st[st.d <= end].groupby("rgst_no").cs.last(); P = P[~P.rgst_no.map(s).eq("소멸등록")]
    gg = g[g.d <= end]; last = gg.groupby("rgst_no").d.transform("max"); cur = gg[gg.d == last].set_index("rgst_no")
    P = P[P.rgst_no.isin(cur.index[cur.pub])].copy()
    c = cur.loc[P.rgst_no]
    P["co_firm"] = c.firm.values.astype(float); P["n_own"] = c.n.values; P["okey"] = c.pkey.values
    P["is_univ"] = P.okey.fillna("").str.contains("대학|산학").astype(float)
    P["age_rg"] = t0 - P.rg // 10000; P["age_ap"] = t0 - P.ap // 10000; P["rem_life"] = (P.ex // 10000) - t0
    past = EV[EV.d <= end]
    P["prior_comm"] = P.rgst_no.isin(set(past.rgst_no)).astype(float)
    osz = P.groupby("okey").size(); P["own_log"] = np.log1p(P.okey.map(osz))
    p3 = past[past.d > end - 30000].drop_duplicates(["rgst_no"])
    oev = p3.groupby("okey").size(); P["own_past_n"] = np.log1p(P.okey.map(oev).fillna(0))
    P["own_past_rate"] = (P.okey.map(oev).fillna(0) + 0.5) / (P.okey.map(osz) + 50)
    mg_ev = p3.merge(pt[["rgst_no", "mg"]], on="rgst_no").groupby("mg").size(); msz = P.groupby("mg").size()
    P["mg_past_rate"] = (P.mg.map(mg_ev).fillna(0) + 0.5) / (P.mg.map(msz) + 50); P["mg_log"] = np.log1p(P.mg.map(msz))
    pp = pay[pay.d <= end]
    P["paid_1y"] = P.rgst_no.map(pp[pp.d > end - 10000].groupby("rgst_no").size()).fillna(0)
    P["last_annl"] = P.rgst_no.map(pp.groupby("rgst_no").annl.max())
    # NTIS (레코드 Year <= t0, 과제연도 <= t0)
    k = lk[(lk.Year <= t0) & (lk.ProjectYear <= t0) & lk.rg_no.isin(set(P.rgst_no))].copy()
    k["is_u"] = k.perform_grp.eq("대학"); k["is_g"] = k.perform_grp.isin(["출연연", "국공립연"]); k["is_f"] = k.perform_grp.eq("기업")
    k["is_dev"] = k.dev_phase.eq("개발연구"); k["is_bas"] = k.dev_phase.eq("기초연구"); k["sb"] = k.sbiz_funds.fillna(0) > 0
    kg = k.groupby("rg_no")
    nt = pd.DataFrame({"g_nproj": kg.ProjectID.nunique(), "g_univ": kg.is_u.max(), "g_gri": kg.is_g.max(), "g_firm": kg.is_f.max(),
                       "g_dev": kg.is_dev.mean(), "g_basic": kg.is_bas.mean(), "g_funds": np.log1p(kg.gov_funds.max()),
                       "g_sbiz": kg.sb.mean(), "g_endgap": t0 - kg.tend.max(), "g_mind": kg.m_ind.mean()})
    P = P.merge(nt, left_on="rgst_no", right_index=True, how="left")
    P["gov"] = P.g_nproj.notna().astype(float); P["g_nproj"] = np.log1p(P.g_nproj.fillna(0))
    for col in ["g_univ", "g_gri", "g_firm"]:
        P[col] = P[col].fillna(False).astype(float)
    nolink = (P.ap < 20050101) & (P.rg < 20070101)
    for col in ["gov", "g_nproj", "g_univ", "g_gri", "g_firm"]:
        P.loc[nolink, col] = np.nan
    # 셀 수요: (KSIC, 메인그룹) 기업 출원 — 알려진 날짜(공개·등록) 기준
    m3 = MK[(MK.known <= end) & (MK.known > end - 30000)]; m6 = MK[(MK.known <= end - 30000) & (MK.known > end - 60000)]
    cf3 = m3[m3.firm].groupby(["ksic_i", "mg"]).size(); ca3 = m3.groupby(["ksic_i", "mg"]).size(); cf6 = m6[m6.firm].groupby(["ksic_i", "mg"]).size()
    ix = pd.MultiIndex.from_arrays([P.ksic_i, P.mg])
    f3 = cf3.reindex(ix).fillna(0).values; a3 = ca3.reindex(ix).fillna(0).values; f6 = cf6.reindex(ix).fillna(0).values
    P["c_firm3"] = np.log1p(f3); P["c_firm_g"] = np.log((f3 + 1) / (f6 + 1)); P["c_firm_sh"] = (f3 + 1) / (a3 + 2)
    # KSIC 수준 NTIS 연구비(a_link_pit, avail_year <= t0) — 교훈 재확인용 M3
    ky = KY[(KY.method == "a_link_pit") & (KY.avail_year <= t0)]
    if len(ky):
        pv = ky.pivot_table(index="ksic", columns="year", values="gov_funds", aggfunc="sum")
        a_ = pv[[y for y in pv.columns if t0 - 2 <= y <= t0 - 1]].sum(1); b_ = pv[[y for y in pv.columns if t0 - 4 <= y <= t0 - 3]].sum(1)
        gr = np.log((a_ + 1) / (b_ + 1)); gr[b_ <= 0] = np.nan
        gr.index = pd.to_numeric(gr.index, errors="coerce"); P["k_rnd_g"] = P.ksic_i.map(gr)
        sh = ky[ky.year == t0 - 1].groupby("ksic").gov_share_of_year.sum(); sh.index = pd.to_numeric(sh.index, errors="coerce"); P["k_rnd_sh"] = P.ksic_i.map(sh)
    else:
        P["k_rnd_g"] = np.nan; P["k_rnd_sh"] = np.nan
    w = EV[(EV.d > end) & (EV.d <= end2)]
    P["y"] = P.rgst_no.isin(set(w[~w.bulk].rgst_no)).astype(int)
    P.loc[P.rgst_no.isin(set(w[w.bulk].rgst_no)) & (P.y == 0), "y"] = -1   # 대량 거래로만 이전된 특허는 평가 제외
    P["y_tr"] = P.rgst_no.isin(set(w[w.typ == "T"].rgst_no)).astype(int)
    P["y_lic"] = P.rgst_no.isin(set(w[w.typ != "T"].rgst_no)).astype(int)
    P["t0"] = t0
    rows.append(P.drop(columns=["ipcCd", "applDate", "cndrtExptnDate", "rgstDate", "claimCount", "applNo", "title"], errors="ignore"))
    log(t0, len(P), "y", int((P.y == 1).sum()), "bulk_excl", int((P.y == -1).sum()), "tr", int(P.y_tr.sum()), "lic", int(P.y_lic.sum()), "gov", round(P.gov.mean(), 3), "inst", P.okey.nunique())
D = pd.concat(rows, ignore_index=True)
D.to_parquet(f"{CACHE}/I6_panel.parquet", index=False)
log("done", D.shape)
