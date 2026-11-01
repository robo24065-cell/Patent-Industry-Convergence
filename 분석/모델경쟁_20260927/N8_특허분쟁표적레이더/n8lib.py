# -*- coding: utf-8 -*-
"""N8 특허 분쟁 표적 레이더 (J2) — 공용 라이브러리.
단위 = (등록특허·실용신안, t0): t0 말까지 등록, t0 말까지 소멸·말소 기록 없음, 존속만료 > t0 말 (제공범위 = registry_full 656,360건).
결과 = (t0, t0+2] 안에 무효(심판)예고등록 또는 권리범위(심판)예고등록 1건 이상.
🔒 최종시험: t0=2022 → 예고등록 2023-01-01~2024-12-31 (F=2023). 학습 t0 2008–2020(결과 ≤2022-12-31).
봉인: dev 모드는 등록원부 사건 날짜 < 2023-01-01 추출본만 읽는다(extract(SEAL)). 최종 결과창 사건은 final 모드의 final_labels()만 읽는다.
파일럿(_ideas_R5/j2_build.py) 피처를 그대로 재현하고(compat), 6라운드 도전자 피처를 같은 패널에 추가 열로 만든다."""
import os, re, time, sqlite3
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
HERE = os.path.join(MC, "N8_특허분쟁표적레이더")
CACHE = os.path.join(HERE, "cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
KOSIS = os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv")
HSMAP = os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv")
HSIMP = os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv")
os.makedirs(CACHE, exist_ok=True)
SEAL = "20230101"          # dev 봉인: 이 날짜 이상 사건은 읽지 않는다
F_YEAR = 2023
DEV = list(range(2012, 2021))
TRAIN_MIN = 2008
DISP = ["무효(심판)예고등록", "권리범위(심판)예고등록"]
OTHER_TRIAL = ["심판예고등록", "정정(심판)예고등록", "취소신청예고등록", "이의신청예고등록", "심결취소소송예고등록", "취소(심판)예고등록", "상고예고등록"]
EXTINCT = ["소멸등록", "권리의 말소등록", "말소등록"]
_t = time.time()


def log(*a):
    print(*a, round(time.time() - _t), flush=True)


def ro():
    return sqlite3.connect(f"file:{REG}?mode=ro", uri=True)


# ------------------------------------------------------------------ 추출
def extract(cut=SEAL):
    """등록원부 추출(사건 날짜 < cut). cut이 SEAL보다 크면 final/future 모드에서만 호출해야 한다."""
    f = f"{CACHE}/reg_lt{cut}.pkl"
    if os.path.exists(f):
        return pd.read_pickle(f)
    c = ro(); R = {}
    R["pt"] = pd.read_sql("select rgst_no, ksic, applDate, applNo, claimCount, cndrtExptnDate, ipcCd, rgstDate from patent", c)
    R["rt"] = pd.read_sql(f"select rgst_no, rgstCsDate d, rgstCsName cs from right where rgstCsDate < '{cut}'", c)
    ow = pd.read_sql(f"""select rgst_no, seq, ownerName, ownerRgstCsDate d, ownerRgstCsName cs from owner
        where ownerRgstCsDate is null or ownerRgstCsDate='' or ownerRgstCsDate < '{cut}'""", c)
    um = pd.read_sql(f"""select rgst_no, seq, utilityModelOwnerName ownerName, utilityModelOwnerRgstCsDate d, utilityModelOwnerRgstCsName cs
        from utilityModelOwner where utilityModelOwnerRgstCsDate is null or utilityModelOwnerRgstCsDate='' or utilityModelOwnerRgstCsDate < '{cut}'""", c)
    R["ow"] = ow; R["um"] = um
    l1 = pd.read_sql(f"select rgst_no, elsLcnsStartDate d, elsLcnsEndDate e, elsLcnsPsnNm nm, 'E' typ from elsLcnsList where elsLcnsStartDate < '{cut}'", c)
    l2 = pd.read_sql(f"select rgst_no, noelsLcnsStartDate d, noelsLcnsEndDate e, noelsLcnsPsnNm nm, 'N' typ from noelsLcnsList where noelsLcnsStartDate < '{cut}'", c)
    R["lc"] = pd.concat([l1, l2], ignore_index=True)
    R["ni"] = pd.read_sql("select rgst_no, count(*) n_inv from inventor group by rgst_no", c)
    R["py"] = pd.read_sql(f"select rgst_no, payDate d, statAnnl, lastAnnl from pay where payDate < '{cut}'", c)
    R["ap"] = pd.read_sql("select rgst_no, seq, applicantCd, applicantName, applicantNatl from applicant", c)
    for k in ("rt", "py", "lc"):
        assert R[k].d.max() < cut, k
    for k in ("ow", "um"):
        assert R[k].d.replace("", np.nan).dropna().max() < cut, k
    pd.to_pickle(R, f)
    return R


def final_labels(lo="20230101", hi="20241231"):
    """🔒 final 모드 전용: 최종 결과창 분쟁 사건. dev 모드에서 호출 금지."""
    c = ro()
    q = f"select rgst_no, rgstCsDate d, rgstCsName cs from right where rgstCsDate >= '{lo}' and rgstCsDate <= '{hi}' and rgstCsName in ('{DISP[0]}','{DISP[1]}')"
    return pd.read_sql(q, c)


# ------------------------------------------------------------------ 보조
def norm_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = re.sub(r"\(주\)|㈜|주식회사|유한회사|\(유\)|유한책임회사|\(사\)|사단법인|재단법인|\(재\)|株式会社|co\.?,?\s*ltd\.?|inc\.?|corp(oration)?\.?|limited", "", s, flags=re.I)
    return re.sub(r"[\s\.,\-·()\[\]]", "", s).lower()


def family_tokens(pt):
    f = f"{CACHE}/fam_tokens.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    import duckdb
    fam = duckdb.sql(f"select appl_no, any_value(family) fam from read_parquet('{META}') where cc='KR' group by appl_no").df()
    rows = []
    for a, s in zip(fam.appl_no, fam.fam):
        if not isinstance(s, str):
            continue
        for tk in set(s.split("|")):
            m = re.match(r"^(WO|US|JP)(\d{4})\d+", tk)
            if m and 1978 <= int(m.group(2)) <= 2030:
                rows.append((a, m.group(1), int(m.group(2))))
            elif tk[:2] in ("EP", "CN"):
                rows.append((a, tk[:2], None))
    FT = pd.DataFrame(rows, columns=["applNo", "office", "yr"]).drop_duplicates()
    FT["yr"] = FT.yr.astype(float)
    FT["ay"] = FT.applNo.map(dict(zip(pt.applNo, pd.to_numeric(pt.applDate, errors="coerce") // 10000)))
    FT.to_parquet(f, index=False)
    return FT


def external_unit():
    ros = pd.read_parquet(f"{EXT}/r3ros_rec_sci.parquet", columns=["cc", "appl_no", "n_us", "n_us_sci", "n_paper_cit", "first_issue_year"], filters=[("cc", "=", "KR")])
    ros = ros.groupby("appl_no").agg(n_us=("n_us", "max"), n_us_sci=("n_us_sci", "max"), n_paper_cit=("n_paper_cit", "max"), fiy=("first_issue_year", "min")).reset_index().rename(columns={"appl_no": "applNo"})
    kp = pd.read_parquet(f"{EXT}/quality_kpss_family_link.parquet", columns=["cc", "appl_no", "issue_date", "xi_real", "cites"], filters=[("cc", "=", "KR")])
    kp["iy"] = pd.to_datetime(kp.issue_date, errors="coerce").dt.year
    kp = kp.rename(columns={"appl_no": "applNo"})
    et = pd.read_parquet(f"{EXT}/quality_etsi_sep_link.parquet"); et = et[et.cc == "KR"].copy()
    et["dy"] = pd.to_datetime(et.first_decl, errors="coerce").dt.year
    et = et.groupby("appl_no").dy.min().rename("etsi_y").reset_index().rename(columns={"appl_no": "applNo"})
    nl = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "Year", "perform_grp"])
    nl["Year"] = pd.to_numeric(nl.Year, errors="coerce"); nl = nl.rename(columns={"appl_no": "applNo"})
    return ros, kp, et, nl


def ksic_industry():
    """KSIC 5자리 외부 산업 데이터(광업제조업조사·HS6 수입). 연도 y 값은 y+1년에 알려진다고 보고 원점 t0에서 ≤ t0-1만 쓴다."""
    k = pd.read_csv(KOSIS, encoding="utf-8-sig")
    k = k[k.항목코드.isin(["T01", "T04"])]
    # 10차(≤2020)와 11차(≥2020) 연결: 2020 비율로 11차를 10차 수준에 맞춘다
    piv = k.pivot_table(index=["세세분류코드", "항목코드", "연도"], columns="기준", values="값").reset_index()
    out = []
    for (kk, it), g in piv.groupby(["세세분류코드", "항목코드"]):
        g = g.set_index("연도")
        r = g.loc[2020, "10차"] / g.loc[2020, "11차"] if (2020 in g.index and g.loc[2020, "11차"] > 0) else 1.0
        s = g["10차"].copy()
        s.loc[g.index > 2020] = g.loc[g.index > 2020, "11차"] * r
        for y, v in s.dropna().items():
            out.append((str(kk), it, int(y), float(v)))
    K = pd.DataFrame(out, columns=["ksic", "item", "year", "v"])
    m = pd.read_csv(HSMAP, dtype={"k5": str, "hs6": str})
    h = pd.read_csv(HSIMP, dtype={"hs6": str})
    h = h.merge(m[["k5", "hs6"]].drop_duplicates(), on="hs6")
    H = h.groupby(["k5", "partner", "year"]).value.sum().unstack("partner").reset_index().rename(columns={"k5": "ksic"})
    return K, H


def ksic_feats(t0, K, H):
    def val(item, y):
        s = K[(K.item == item) & (K.year <= y)].sort_values("year").groupby("ksic").tail(1)
        return s.set_index("ksic").v
    f = pd.DataFrame(index=sorted(K.ksic.unique()))
    s1, s3 = val("T04", t0 - 1), val("T04", t0 - 3)
    e1, e3 = val("T01", t0 - 1), val("T01", t0 - 3)
    f["ks_ship_g2"] = np.log(s1 / s3)
    f["ks_estab_log"] = np.log(e1)
    f["ks_estab_g2"] = np.log(e1 / e3)
    if t0 - 3 < 2008:
        f[["ks_ship_g2", "ks_estab_g2"]] = np.nan
    Hy = H.set_index(["ksic", "year"])
    def imp(y):
        try:
            return Hy.xs(y, level="year")
        except KeyError:
            return None
    a, b = imp(t0 - 1), imp(t0 - 4)
    g = pd.DataFrame(index=f.index)
    if a is not None and b is not None:
        g["imp_g3"] = np.log((a.WLD + 1) / (b.WLD + 1)).reindex(g.index)
        g["imp_cn_share"] = (a.CN / a.WLD).reindex(g.index)
    return f.join(g)


# ------------------------------------------------------------------ 패널
def prep(R):
    pt = R["pt"].copy()
    rt, py, lc = R["rt"].copy(), R["py"].copy(), R["lc"].copy()
    ow, um = R["ow"].copy(), R["um"].copy()
    for df in (rt, ow, um, py, lc):
        df["d"] = pd.to_numeric(df.d, errors="coerce")
    lc["e"] = pd.to_numeric(lc.e, errors="coerce")
    pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce"); pt["ap"] = pd.to_numeric(pt.applDate, errors="coerce")
    pt["ex"] = pd.to_numeric(pt.cndrtExptnDate, errors="coerce"); pt["claims"] = pd.to_numeric(pt.claimCount, errors="coerce")
    pt["n_ipc"] = pt.ipcCd.fillna("").str.count(r"\(").clip(lower=1)
    _m = pt.ipcCd.fillna("").str.split(",").str[0].str.strip().str.extract(r"^([A-H]\d\d[A-Z])\s*(\d+)/")
    pt["mg"] = (_m[0] + _m[1]).fillna("NA")
    pt["ipc_sec"] = pt.ipcCd.fillna("Z").str[0].map(lambda c: "ABCDEFGHZ".find(c))
    pt["utility"] = (pt.rgst_no.str[:2] == "20").astype(int)
    pt["ksic_i"] = pd.to_numeric(pt.ksic, errors="coerce")
    pt = pt.merge(R["ni"], on="rgst_no", how="left")
    ap = R["ap"]
    ap0 = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no")
    nap = ap.groupby("rgst_no").size().rename("n_appl")
    pt = pt.merge(ap0[["rgst_no", "applicantCd", "applicantNatl", "applicantName"]], on="rgst_no", how="left").merge(nap, on="rgst_no", how="left")
    pt["atype"] = pd.to_numeric(pt.applicantCd.astype(str).str[0], errors="coerce")
    pt["foreign"] = (pt.applicantNatl.fillna("대한민국") != "대한민국").astype(int)
    pt["ext_d"] = pt.rgst_no.map(rt[rt.cs.isin(EXTINCT)].groupby("rgst_no").d.min())
    py["la"] = pd.to_numeric(py.lastAnnl, errors="coerce")
    uniq = pd.Series(pd.concat([ow.ownerName, um.ownerName]).dropna().unique())
    nm = dict(zip(uniq, uniq.map(norm_name)))
    ow["on"] = ow.ownerName.map(nm); um["on"] = um.ownerName.map(nm)
    ow = ow.sort_values(["rgst_no", "seq"]); ow2 = pd.concat([ow, um]).sort_values(["rgst_no", "seq"])
    return dict(pt=pt, rt=rt, py=py, lc=lc, ow=ow, ow2=ow2,
                disp=rt[rt.cs.isin(DISP)][["rgst_no", "d"]], othr=rt[rt.cs.isin(OTHER_TRIAL)][["rgst_no", "d", "cs"]],
                teval=rt[rt.cs.str.contains("기술평가", na=False)][["rgst_no", "d"]],
                decided=rt[rt.cs.str.contains("심결확정|심결일부확정", na=False)][["rgst_no", "d"]],
                trf=ow[ow.cs.notna() & (ow.seq > 0)][["rgst_no", "d", "cs"]], trf2=ow2[ow2.cs.notna() & (ow2.seq > 0)][["rgst_no", "d", "cs", "on"]])


def build_t0(t0, X, FT, EXTU, KI, label_disp=None, label_end=None, no_label=False):
    """원점 t0 행 생성. label_disp: 결과 사건표(없으면 X['disp']), label_end: 결과창 끝(YYYYMMDD 정수)."""
    pt, rt, py, lc, ow, ow2 = X["pt"], X["rt"], X["py"], X["lc"], X["ow"], X["ow2"]
    disp, othr, teval, trf, trf2 = X["disp"], X["othr"], X["teval"], X["trf"], X["trf2"]
    E = t0 * 10000 + 1231
    W2 = (t0 + 2) * 10000 + 1231 if label_end is None else label_end
    P = pt[(pt.rg <= E) & ~(pt.ext_d <= E) & (pt.ex > E)].copy()
    P["age_rg"] = t0 - P.rg // 10000; P["age_ap"] = t0 - P.ap // 10000; P["rem_life"] = (P.ex // 10000) - t0
    pp = py[py.d <= E]
    P["last_annl"] = P.rgst_no.map(pp.groupby("rgst_no").lastAnnl.max().astype(float))  # 파일럿 그대로(문자열 max)
    P["paid_ahead"] = P.last_annl - (t0 - P.rg // 10000 + 1)
    P["last_annl_v2"] = P.rgst_no.map(pp.groupby("rgst_no").la.max())                     # 수치 max(보정)
    P["paid_ahead_v2"] = P.last_annl_v2 - (t0 - P.rg // 10000 + 1)
    # 결과
    if no_label:
        P["y"] = np.nan
    else:
        if label_disp is None:
            assert W2 < int(SEAL), "dev 결과창이 봉인선을 넘음"
            dd = disp[(disp.d > E) & (disp.d <= W2)]
        else:
            dd = label_disp[(label_disp.d > E) & (label_disp.d <= W2)]
        P["y"] = P.rgst_no.isin(set(dd.rgst_no)).astype(int)
    # ---- M1 (파일럿)
    dE = disp[disp.d <= E]
    P["own_disp"] = P.rgst_no.map(dE.groupby("rgst_no").size()).fillna(0)
    oE = othr[othr.d <= E]
    P["own_trial"] = P.rgst_no.map(oE.groupby("rgst_no").size()).fillna(0)
    P["tech_eval"] = P.rgst_no.isin(set(teval[teval.d <= E].rgst_no)).astype(int)
    tt = trf[trf.d <= E]
    P["n_trf"] = P.rgst_no.map(tt.groupby("rgst_no").size()).fillna(0)
    P["trf_2y"] = P.rgst_no.map(tt[tt.d > E - 20000].groupby("rgst_no").size()).fillna(0)
    P["merger_trf"] = P.rgst_no.map(tt[tt.cs.isin(["법인합병", "합병"])].groupby("rgst_no").size()).fillna(0)
    lE = lc[lc.d <= E]
    P["lic"] = P.rgst_no.map(lE.groupby("rgst_no").size()).fillna(0)
    cur = ow[(ow.d.isna()) | (ow.d <= E)].drop_duplicates("rgst_no", keep="last").set_index("rgst_no").on
    P["owner"] = P.rgst_no.map(cur).fillna("")
    P["owner_live"] = np.log1p(P.owner.map(P.groupby("owner").size()).where(P.owner != "", 1))
    d5 = set(disp[(disp.d > E - 50000) & (disp.d <= E)].rgst_no)
    P["owner_disp5"] = P.owner.map(P[P.rgst_no.isin(d5)].groupby("owner").size()).fillna(0).where(P.owner != "", 0)
    alive5 = pt[(pt.rg <= E) & ~(pt.ext_d <= E - 50000)]
    dis5 = alive5.rgst_no.isin(d5)
    for key, col in (("mg", "mg_disp5"), ("ksic", "ksic_disp5")):
        g = pd.DataFrame({"k": alive5[key].values, "d": dis5.values}).groupby("k").d.agg(["sum", "size"])
        P[col] = np.log(P[key].map((g["sum"] + 0.5) / (g["size"] + 500)).astype(float))
    ok = FT[(FT.yr <= t0) | (FT.yr.isna() & (FT.ay <= t0 - 4))]
    P["fam_offices"] = P.applNo.map(ok.groupby("applNo").office.nunique()).fillna(0)
    P["fam_us"] = P.applNo.isin(set(ok[ok.office == "US"].applNo)).astype(int)
    # ---- M2 외부 단위(파일럿)
    ros, kp, et, nl = EXTU
    rr = ros[ros.fiy <= t0].set_index("applNo")
    P["us_granted"] = P.applNo.isin(rr.index).astype(int)
    P["us_sci"] = P.applNo.map(rr.n_us_sci).fillna(0)
    P["us_papercit"] = np.log1p(P.applNo.map(rr.n_paper_cit).fillna(0))
    kk = kp[kp.iy <= t0].groupby("applNo").agg(xi=("xi_real", "max"))
    P["kpss_xi"] = np.log1p(P.applNo.map(kk.xi))
    P["etsi"] = P.applNo.isin(set(et[et.etsi_y <= t0].applNo)).astype(int)
    ng = nl[nl.Year <= t0]
    P["gov"] = P.applNo.isin(set(ng.applNo)).astype(int)
    for gname, col in (("기업", "gov_firm"), ("대학", "gov_univ"), ("출연연", "gov_gri")):
        P[col] = P.applNo.isin(set(ng[ng.perform_grp.astype(str).str.contains(gname)].applNo)).astype(int)
    # ================= 6라운드 도전자 피처 =================
    # C02 실용신안 권리자 이력 보정(owner ∪ utilityModelOwner)
    t2 = trf2[trf2.d <= E]
    P["n_trf_v2"] = P.rgst_no.map(t2.groupby("rgst_no").size()).fillna(0)
    P["trf_2y_v2"] = P.rgst_no.map(t2[t2.d > E - 20000].groupby("rgst_no").size()).fillna(0)
    P["merger_trf_v2"] = P.rgst_no.map(t2[t2.cs.isin(["법인합병", "합병"])].groupby("rgst_no").size()).fillna(0)
    cur2 = ow2[(ow2.d.isna()) | (ow2.d <= E)].drop_duplicates("rgst_no", keep="last").set_index("rgst_no").on
    P["owner_v2"] = P.rgst_no.map(cur2).fillna("")
    has = P.owner_v2 != ""
    P["owner_live_v2"] = np.log1p(P.owner_v2.map(P.groupby("owner_v2").size()).where(has, 1))
    P["owner_disp5_v2"] = P.owner_v2.map(P[P.rgst_no.isin(d5)].groupby("owner_v2").size()).fillna(0).where(has, 0)
    # C04 셀 단위 최근 분쟁 당사자 수·최근 2년 분쟁률 (분쟁 특허의 원점 당시 권리자, 소멸 특허 포함)
    curall = ow2[(ow2.d.isna()) | (ow2.d <= E)].drop_duplicates("rgst_no", keep="last").set_index("rgst_no").on
    D5 = pt[pt.rgst_no.isin(d5)][["rgst_no", "mg", "ksic"]].copy(); D5["on"] = D5.rgst_no.map(curall).fillna("?")
    P["mg_parties5"] = np.log1p(P.mg.map(D5.groupby("mg").on.nunique()).fillna(0))
    P["ksic_parties5"] = np.log1p(P.ksic.map(D5.groupby("ksic").on.nunique()).fillna(0))
    d2 = set(disp[(disp.d > E - 20000) & (disp.d <= E)].rgst_no)
    alive2 = pt[(pt.rg <= E) & ~(pt.ext_d <= E - 20000)]
    g = pd.DataFrame({"k": alive2.mg.values, "d": alive2.rgst_no.isin(d2).values}).groupby("k").d.agg(["sum", "size"])
    P["mg_disp2"] = np.log(P.mg.map((g["sum"] + 0.2) / (g["size"] + 500)).astype(float))
    # C05 자기 분쟁 이력 세부(최근성·심결·소송·정정)
    last = dE.groupby("rgst_no").d.max()
    P["yrs_last_disp"] = t0 - P.rgst_no.map(last) // 10000
    dc = X["decided"]
    P["own_decided"] = P.rgst_no.map(dc[dc.d <= E].groupby("rgst_no").size()).fillna(0)
    P["own_appeal"] = P.rgst_no.map(oE[oE.cs.isin(["심결취소소송예고등록", "상고예고등록"])].groupby("rgst_no").size()).fillna(0)
    P["own_corr"] = P.rgst_no.map(oE[oE.cs == "정정(심판)예고등록"].groupby("rgst_no").size()).fillna(0)
    # C06 권리자 심판 경험·매입 집중(NPE 유사)
    o5 = set(othr[(othr.d > E - 50000) & (othr.d <= E)].rgst_no)
    P["owner_trial5"] = P.owner_v2.map(P[P.rgst_no.isin(o5)].groupby("owner_v2").size()).fillna(0).where(has, 0)
    acq = P.n_trf_v2 > 0
    gq = pd.DataFrame({"o": P.owner_v2.values, "a": acq.values}).groupby("o").a.agg(["sum", "mean"])
    P["owner_acq_share"] = P.owner_v2.map(gq["mean"]).where(has, np.nan)
    P["owner_n_acq"] = np.log1p(P.owner_v2.map(gq["sum"]).where(has, 0))
    # C07 실시권 세부
    act = lE[(lE.e.isna()) | (lE.e > E)]
    P["lic_active"] = P.rgst_no.map(act.groupby("rgst_no").size()).fillna(0)
    P["lic_excl"] = P.rgst_no.isin(set(lE[lE.typ == "E"].rgst_no)).astype(int)
    P["lic_2y"] = P.rgst_no.map(lE[lE.d > E - 20000].groupby("rgst_no").size()).fillna(0)
    P["n_licensees"] = P.rgst_no.map(lE.groupby("rgst_no").nm.nunique()).fillna(0)
    # C09·C10 KSIC 외부 산업(광업제조업조사·HS6 수입)
    kf = ksic_feats(t0, *KI)
    for c in ["ks_ship_g2", "ks_estab_log", "ks_estab_g2", "imp_g3", "imp_cn_share"]:
        P[c] = P.ksic.map(kf[c]) if c in kf else np.nan
    P["t0"] = t0
    drop = ["ksic_i_dummy"]
    keep = [c for c in P.columns if c not in ("applDate", "claimCount", "cndrtExptnDate", "ipcCd", "rgstDate", "rg", "ap", "ex", "applicantCd", "applicantNatl", "applicantName", "ext_d") + tuple(drop)]
    return P[keep]


def to32(D):
    for cn in D.columns:
        if D[cn].dtype == "float64":
            D[cn] = D[cn].astype("float32")
    return D


# ------------------------------------------------------------------ 평가
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def auc(y, p, w=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, np.asarray(p), sample_weight=w)


def skill(a, b):
    return (a - b) / (1 - b)


def fit_pred(Xtr, ytr, Xte, model="xgb", seed=0, w=None, params=None):
    if model == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, **(params or XGB_PARAMS))
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr, logisticregression__sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def rank01(p):
    from scipy.stats import rankdata
    return rankdata(p) / len(p)


def topk(y, p, frac=None, k=None):
    y = np.asarray(y); o = np.argsort(-np.asarray(p), kind="stable")
    if k is None:
        k = max(1, int(round(frac * len(y))))
    return float(y[o[:k]].mean()), int(y[o[:k]].sum()), k


def _grp(p):
    u, inv = np.unique(np.asarray(p), return_inverse=True)
    return inv, len(u)


def wauc_g(y, inv, G, w):
    """동점 묶음 기반 가중 AUC(빠름). inv: 점수 순위 묶음 번호."""
    pos = np.bincount(inv, weights=w * (y == 1), minlength=G); neg = np.bincount(inv, weights=w * (y == 0), minlength=G)
    cn = np.cumsum(neg) - neg
    return float((pos * (cn + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def boot_ci(y, pm, pb, groups, n=200, seed=0):
    """군집(권리자) 부트스트랩: 군집 뽑힌 횟수를 행 가중치로. skill·AUC_m·AUC_b의 95% 구간."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    im, Gm = _grp(pm); ib, Gb = _grp(pb)
    codes, uniq = pd.factorize(np.asarray(groups)); G = len(uniq)
    S, A, B = [], [], []
    for _ in range(n):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[codes].astype(float)
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            continue
        a, b = wauc_g(y, im, Gm, w), wauc_g(y, ib, Gb, w)
        S.append(skill(a, b)); A.append(a); B.append(b)
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return q(S), q(A), q(B)


def cluster_key(te):
    """권리자 군집: 원점 당시 권리자(실용신안 보정판), 없으면 특허 자신."""
    o = te["owner_v2"].astype(str).values
    return np.where(o == "", "R" + te.rgst_no.astype(str).values, o)
