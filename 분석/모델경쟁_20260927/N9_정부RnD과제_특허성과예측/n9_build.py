# -*- coding: utf-8 -*-
"""N9 학습표 빌더. 사용: python n9_build.py dev   (final/future 모드는 final_run_r6.py에서만 부른다)
기본 열은 파일럿 _ideas_R5\\j1_build.py와 같은 계산(재현 확인용). 6라운드 도전자 열을 추가로 만든다.
단위 = NTIS 과제고유번호 × 과제연도 t (보안과제 제외). 결과 y = 그 과제 성과로 등록된 제공범위(50 KSIC) KR 특허 중 출원연도 ∈ [t, t+1] 이 1건 이상.
🔒 dev: 성과 연결 출원연도 <= 2021만 읽음, 원점 2010–2020.  final: 원점 2010–2020 + 2022(결과 출원 2022–23).  future: 2010–2024(2023·2024 레이블 없음, final 이후에만).
피처 시점: 과제 속성은 t년 값, 성과 이력은 출원연도 <= t-1(원판) / <= t-2(지연판 _L2), 제공 KR 출원은 t-1년 말까지 공개·등록된 것, KOSIS는 t-2년 값."""
import sys, json
import numpy as np, pandas as pd
from n9_lib import *


def build(mode="dev", save=True):
    assert mode in ("dev", "final", "future")
    if mode == "dev":
        p_max, appl_max, origins = 2020, DEV_MAX_APPL_YEAR, list(range(2010, 2021))
    elif mode == "final":
        p_max, appl_max, origins = FINAL_T, FINAL_T + 1, list(range(2010, 2021)) + [FINAL_T]
    else:
        p_max, appl_max, origins = 2024, 2023, list(range(2010, 2025))   # 2023·2024는 레이블 없음(y=-1)
    hist_appl_max = max(o for o in origins) - 1   # 피처용 이력은 이보다 늦은 출원을 쓰지 않음
    cols = ["pjt_no", "year", "research_agency", "perform_grp", "dev_phase", "continuous", "pre_pjt", "ministry", "sixt_cd",
            "cls_L_canon", "econ_goal_cd", "tech_lifecycle", "apply1_cd", "gov_funds", "total_funds", "sbiz_funds", "nongov_funds",
            "total_start", "total_end", "period_start", "period_end", "secret_flag"]
    RAW = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=cols)
    P = RAW[(RAW.year >= 2007) & (RAW.year <= p_max) & (RAW.secret_flag.astype(float).fillna(0) == 0)].copy()
    P["year"] = P.year.astype(int)
    P["org"] = P.research_agency.map(norm_name)
    km = pd.read_parquet(f"{EXT}/ntis_project_ksic_text_map.parquet")
    P = P.merge(km.drop(columns=["year"]), on="pjt_no", how="left")
    log("projects", len(P), mode)

    # ---- 성과 연결(제공범위 50 KSIC) ----
    L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "ksic", "ProjectID", "appl_year"])
    L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
    L = L[L.appl_year <= appl_max].drop_duplicates(["ProjectID", "appl_no"])
    assert L.appl_year.max() <= appl_max
    if mode == "dev":
        assert L.appl_year.max() <= DEV_MAX_APPL_YEAR, "🔒 dev 모드에서 2022+ 성과를 읽음"
    R = pd.read_parquet(f"{EXT}/ntis_rpatent_raw.parquet", columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber"])
    R["Year"] = pd.to_numeric(R.Year, errors="coerce")
    R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.RegistType == "출원") & (R.Year <= hist_appl_max)].drop_duplicates(["ProjectID", "RegistNumber"])
    log("links", len(L), len(R))
    Lc = L.groupby(["ProjectID", "appl_year"]).size().rename("n").reset_index()
    Rc = R.groupby(["ProjectID", "Year"]).size().rename("n").reset_index().rename(columns={"Year": "appl_year"})
    Lc = Lc.merge(P[["pjt_no", "org", "year"]], left_on="ProjectID", right_on="pjt_no", how="inner")
    Rc = Rc.merge(P[["pjt_no", "org", "year"]], left_on="ProjectID", right_on="pjt_no", how="inner")
    # 피처용 연결(이력)은 출원 <= hist_appl_max 만
    LcH = Lc[Lc.appl_year <= hist_appl_max]
    # 연결 특허 KSIC(기관-KSIC 적합도용)
    Lk = L[L.appl_year <= hist_appl_max].merge(P[["pjt_no", "org", "year"]], left_on="ProjectID", right_on="pjt_no", how="inner")

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
    # ---- 6R 추가 열(과제 자체 정보, t년) ----
    P["b1_rank"] = P.groupby("year").b1_tfidf_score.rank(pct=True)
    P["b2_rank"] = P.groupby("year").b2_dict_score.rank(pct=True)
    P["minper_c"] = pd.factorize(P.ministry.astype(str) + "|" + P.perform_grp.astype(str))[0]
    P["b1_top"] = pd.to_numeric(P.b1_tfidf_top_ksic, errors="coerce"); P["b2_top"] = pd.to_numeric(P.b2_dict_top_ksic, errors="coerce")
    P["top_ksic"] = P.b2_top.where(P.b2_dict_assigned == 1, P.b1_top)
    P["pre1"] = P.pre_pjt.astype(str).str.split(",").str[0].where(P.pre_pjt.notna())
    # 레이블(모든 과제연도, 결과창 [y, y+1] <= appl_max 인 연도만)
    posset = Lc[Lc.appl_year.between(Lc.year, Lc.year + 1)]
    P["y_all"] = np.where(P.year + 1 <= appl_max, P.pjt_no.isin(set(posset.ProjectID)).astype(float), np.nan)

    # ---- 선행과제 사슬(전체 원시 2002~, 보안 포함 — 사슬 연결용 식별자만) ----
    pre_map = RAW[RAW.year <= p_max].assign(pre1=lambda d: d.pre_pjt.astype(str).str.split(",").str[0].where(d.pre_pjt.notna())).set_index("pjt_no").pre1
    pre_map = pre_map[~pre_map.index.duplicated()]
    anc = []
    cur = P[["pjt_no", "year"]].copy(); cur["anc"] = P.pre1.values; depth = 0
    while depth < 15:
        cur = cur[cur.anc.notna() & (cur.anc != cur.pjt_no)]
        if len(cur) == 0:
            break
        depth += 1
        anc.append(cur.assign(d=depth))
        cur = cur.assign(anc=cur.anc.map(pre_map))
    A = pd.concat(anc, ignore_index=True) if anc else pd.DataFrame(columns=["pjt_no", "year", "anc", "d"])
    P["chain_len"] = P.pjt_no.map(A.groupby("pjt_no").d.max()).fillna(0)
    log("chain", len(A), float(P.chain_len.mean()))

    # ---- 제공 KR 출원(메타): 기관명 매칭, 산업 추세 ----
    import duckdb
    kr = duckdb.sql(f"""select appl_no, any_value(applicant) applicant, any_value(appl_year) appl_year, list(distinct ksic) ksics,
                        min(least(coalesce(pub_date,'99999999'), coalesce(rgst_date,'99999999'))) known
                        from read_parquet('{META}') where cc='KR' and appl_year <= {hist_appl_max} group by appl_no""").df()
    kr["known_year"] = pd.to_numeric(kr.known.str[:4], errors="coerce")
    kr["nm"] = kr.applicant.map(norm_name).str.replace("산학협력단", "", regex=False)
    krk = kr[["nm", "appl_year", "known_year", "ksics"]].explode("ksics").rename(columns={"ksics": "ksic"})
    krk["ksic"] = pd.to_numeric(krk.ksic)
    ind = krk.groupby(["ksic", "appl_year"]).size()
    P["org2"] = P.org.str.replace("산학협력단", "", regex=False)
    orgset = set(P.org2)
    kro = kr[kr.nm.isin(orgset)]
    kroc = kro.groupby(["nm", "appl_year", "known_year"]).size().rename("n").reset_index()
    krok = krk[krk.nm.isin(orgset)].groupby(["nm", "ksic", "appl_year", "known_year"]).size().rename("n").reset_index()
    log("kr meta", len(kr), "org matched names", kro.nm.nunique())

    # ---- KOSIS 광업제조업조사(10차, ≤2020) ----
    ko = pd.read_csv(KOSIS); ko = ko[(ko["기준"] == "10차")]
    ship = ko[ko["항목명"] == "출하액"].pivot_table(index="세세분류코드", columns="연도", values="값")
    emp = ko[ko["항목명"] == "종사자수"].pivot_table(index="세세분류코드", columns="연도", values="값")

    def kval(tab, k, yr):
        for yy in (yr, yr - 1):   # 2010 경제총조사 연도는 직전 연도로 대체
            if k in tab.index and yy in tab.columns and pd.notna(tab.at[k, yy]):
                return tab.at[k, yy]
        return np.nan

    out = []
    for v in origins:
        X = P[P.year == v].copy()
        if mode == "dev":
            assert v + 1 <= DEV_MAX_APPL_YEAR
        # 결과
        if v + 1 <= appl_max:
            pos = set(Lc[(Lc.year == v) & Lc.appl_year.between(v, v + 1)].ProjectID)
            X["y"] = X.pjt_no.isin(pos).astype(int)
        else:
            X["y"] = -1
        hist = P[P.year.between(v - 3, v - 1)]
        nproj = hist.groupby("org").size()
        X["org_nproj"] = np.log1p(X.org.map(nproj).fillna(0))
        for suf, amax in (("", v - 1), ("_L2", v - 2)):
            RcW = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= amax)]
            LcW = LcH[LcH.year.between(v - 3, v - 1) & (LcH.appl_year <= amax)]
            ra = RcW.groupby("org").n.sum(); la = LcW.groupby("org").n.sum()
            rp = RcW.groupby("org").pjt_no.nunique(); lp = LcW.groupby("org").pjt_no.nunique()
            X["org_kr_pat" + suf] = np.log1p(X.org.map(ra).fillna(0)); X["org_kr_rate" + suf] = X.org.map(rp).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
            X["org_p50" + suf] = np.log1p(X.org.map(la).fillna(0)); X["org_p50_rate" + suf] = X.org.map(lp).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
            X["org_p50_share" + suf] = X.org.map(la).fillna(0) / (X.org.map(ra).fillna(0) + 1)
            rpre = Rc[Rc.appl_year <= amax].groupby("ProjectID").n.sum(); lpre = LcH[LcH.appl_year <= amax].groupby("ProjectID").n.sum()
            X["pre_kr_pat" + suf] = np.log1p(X.pre_pjt.map(rpre).fillna(0)) * X.has_pre
            X["pre_p50" + suf] = np.log1p(X.pre_pjt.map(lpre).fillna(0)) * X.has_pre
        # 6R: 선행 사슬 누적 성과(지연판, 출원 <= v-2)
        Av = A[A.year == v]
        ch = Av.merge(LcH[LcH.appl_year <= v - 2][["ProjectID", "n"]], left_on="anc", right_on="ProjectID", how="inner").groupby("pjt_no").n.sum()
        chr_ = Av.merge(Rc[Rc.appl_year <= v - 2][["ProjectID", "n"]], left_on="anc", right_on="ProjectID", how="inner").groupby("pjt_no").n.sum()
        X["chain_p50_L2"] = np.log1p(X.pjt_no.map(ch).fillna(0)); X["chain_kr_L2"] = np.log1p(X.pjt_no.map(chr_).fillna(0))
        # 6R: 텍스트 대응 KSIC의 과거 양성률(과제연도 v-5..v-3: 결과창 끝 <= v-2)
        pr = P[P.year.between(v - 5, v - 3) & P.y_all.notna()]
        for col, nm_ in (("b1_top", "ksic_prior_b1"), ("top_ksic", "ksic_prior_top")):
            g = pr.groupby(col).y_all.agg(["sum", "count"])
            rate = (g["sum"] + 1) / (g["count"] + 20)
            X[nm_] = X[col].map(rate)
        # 6R: 기관-KSIC 적합도 — 기관의 과거 제공 연결 특허(출원 <= v-2) 중 텍스트 대응 KSIC 비중
        lk = Lk[Lk.year.between(v - 5, v - 1) & (Lk.appl_year <= v - 2)]
        okk = lk.groupby(["org", "ksic"]).appl_no.nunique()
        okt = lk.groupby("org").appl_no.nunique()
        key = pd.MultiIndex.from_arrays([X.org.values, X.top_ksic.fillna(-1).astype(int).astype(str).values])
        X["org_ksic_fit_L2"] = okk.reindex(key).fillna(0).values / (X.org.map(okt).fillna(0).values + 1)
        # 6R: 제공 KR 출원(기관명 매칭) — 출원 <= v-2, v-1년 말까지 공개·등록
        kc = kroc[(kroc.appl_year <= v - 2) & (kroc.known_year <= v - 1)]
        X["org_krapp"] = np.log1p(X.org2.map(kc.groupby("nm").n.sum()).fillna(0))
        X["org_krapp3"] = np.log1p(X.org2.map(kc[kc.appl_year >= v - 4].groupby("nm").n.sum()).fillna(0))
        kk = krok[(krok.appl_year <= v - 2) & (krok.known_year <= v - 1)].groupby(["nm", "ksic"]).n.sum()
        key2 = pd.MultiIndex.from_arrays([X.org2.values, X.top_ksic.fillna(-1).astype(int).values])
        X["org_krapp_topk"] = np.log1p(kk.reindex(key2).fillna(0).values)
        # 6R: 산업 추세 — 제공 KR 출원 성장(텍스트 대응 KSIC, 출원 v-3·v-4 대 v-5·v-6)
        def icount(k, a, b):
            return sum(ind.get((k, yy), 0) for yy in range(a, b + 1))
        tk = X.top_ksic.values
        uk = {k: np.log((icount(k, v - 4, v - 3) + 1) / (icount(k, v - 6, v - 5) + 1)) for k in pd.unique(tk[~pd.isna(tk)])}
        X["ind_growth"] = pd.Series(tk).map(uk).values
        # 6R: KOSIS 출하액·종사자 성장(t-2 대 t-4)
        uk2 = {k: np.log(kval(ship, int(k), v - 2) / kval(ship, int(k), v - 4)) for k in pd.unique(tk[~pd.isna(tk)])}
        uk3 = {k: np.log(kval(emp, int(k), v - 2) / kval(emp, int(k), v - 4)) for k in pd.unique(tk[~pd.isna(tk)])}
        X["kosis_ship_g"] = pd.Series(tk).map(uk2).values; X["kosis_emp_g"] = pd.Series(tk).map(uk3).values
        X["t0"] = v
        out.append(X)
        log(v, len(X), int((X.y == 1).sum()), round((X.y == 1).mean(), 4))
    D = pd.concat(out, ignore_index=True)
    # 텍스트 중심점 유사도(n9_text.py 산출, 있으면 결합)
    fp = f"{CACHE}/proj_ptsim.parquet"
    if os.path.exists(fp):
        D = D.merge(pd.read_parquet(fp), on="pjt_no", how="left")
    drop = ["research_agency", "dev_phase", "continuous", "ministry", "sixt_cd", "cls_L_canon", "econ_goal_cd", "tech_lifecycle", "apply1_cd", "gov_funds", "total_funds",
            "sbiz_funds", "nongov_funds", "total_start", "total_end", "period_start", "period_end", "secret_flag", "b1_tfidf_top_ksic", "b2_dict_top_ksic", "y_all", "pre1"]
    D = D.drop(columns=[c for c in drop if c in D.columns])
    for cn in D.columns:
        if D[cn].dtype == "float64":
            D[cn] = D[cn].astype("float32")
    if save:
        D.to_parquet(f"{CACHE}/panel_{mode}.parquet", index=False)
    log("saved", len(D), mode)
    return D


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "dev"
    assert m == "dev", "이 스크립트 직접 실행은 dev만 — final·future는 final_run_r6.py"
    build(m)
