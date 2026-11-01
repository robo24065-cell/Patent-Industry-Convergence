# -*- coding: utf-8 -*-
"""N6 패널 빌더. 파일럿 i8_build.py의 M0·M1·M2 피처를 그대로 재현하고 도전자 피처(C01~C10)를 덧붙인다.
사용: python n6_features.py dev   -> cache/panel_dev.parquet (t0 2010–2019; 2010·2011은 목표 인코딩 이력 전용)
final 모드는 final_run_r5.py에서만 호출한다(🔒, 이번 라운드 실행 금지).
시점 규칙
- NTIS 과제: 과제연도 <= t0 (파일럿 규칙 유지). NTIS 성과(범위 밖 IP 포함): 성과연도 <= t0-1 (보고 시차 보수 처리).
- 위험집합: t0 말까지 공개·등록으로 알려진(known_date <= t0-12-31) 제공범위 KR 출원 없음.
- NPS: t0-12 스냅샷(2015-12~), 직전 연말 스냅샷. 산업 침투율: t0-12 NPS 활성 기업 중 t0 말까지 알려진 제공범위 출원 보유 비율.
- 국세청: 폐업일 end_dt <= t0-12-31 만(날짜로 자름). 대상표가 출원인 목록이라 연결 기업에만 값이 있다(미연결=0) — 한계로 기록.
- 과거 전환율 인코딩: 원점 T의 값은 결과창이 T 이전에 끝난 원점(t0 <= T-2)의 레이블만 쓴다.
- 쓰지 않는 것: business_rights(제공범위 보유 법인만 조회한 사후 목록), NTIS series 'after='(미래 과제 번호), 연결표 보유 여부."""
import sys
import numpy as np, pandas as pd
from n6_lib import *

SIXT = ["IT", "BT", "NT", "ET", "ST", "CT"]
CLS = ["기계", "전기전자", "정보통신", "화공", "재료", "보건의료", "농림수산", "생명과학"]
MIN_GRP = {"중소기업청": "sme", "중소벤처기업부": "sme", "산업통상자원부": "motie", "지식경제부": "motie", "과학기술정보통신부": "msit",
           "미래창조과학부": "msit", "교육과학기술부": "msit", "농림축산식품부": "agri", "농촌진흥청": "agri", "농림수산식품부": "agri",
           "환경부": "env", "보건복지부": "health", "해양수산부": "ocean", "국토교통부": "land"}


def apply_grp(s):
    s = str(s or "")
    if s.startswith("제조업"):
        return "mfg"
    if "정보" in s or "통신" in s or "출판" in s:
        return "ict"
    if "보건" in s or "의료" in s:
        return "health"
    if "농" in s or "어업" in s or "임업" in s:
        return "agri"
    if "에너지" in s or "전기,가스" in s:
        return "energy"
    if "건설" in s:
        return "constr"
    if "전문" in s or "과학" in s or "기술서비스" in s:
        return "prosvc"
    return "other"


def build(mode="dev", t0_list=None):
    if mode == "dev":
        t0_list = t0_list or T0_ALL_DEV
        assert max(t0_list) + 2 <= F_YEAR - 1, "dev: 결과창이 F 이후와 겹치는 원점 금지"
        year_hi = 2019
    elif mode == "final":   # 🔒 final_run_r5.py --mode final 에서만
        t0_list = t0_list or (T0_ALL_DEV + [FINAL_T0])
        assert max(t0_list) == FINAL_T0
        year_hi = FINAL_T0
    else:                   # future: 최신 원점(레이블 없음), final 이후에만
        assert mode == "future"
        year_hi = max(t0_list)
    NT = load_ntis(2008, year_hi)
    NT["mgrp"] = NT.ministry.map(MIN_GRP).fillna("other")
    NT["agrp"] = NT.apply1.map(apply_grp)
    NT["end_y"] = pd.to_numeric(NT.total_end.astype(str).str[:4], errors="coerce")
    log("NTIS firm projects", len(NT), NT.bz.nunique())
    firm = firm_link(NT)
    log("firms", len(firm), "linked", round(firm.linked.mean(), 3))
    AF = load_apps(firm, mode)
    first_known = AF.groupby("bz").known_date.min()
    k2 = dict(zip(firm.bz, firm.key))
    # NPS
    NPS = {}
    for y in range(2014, max(t0_list) + 1):
        fp = f"{R3I}/cache/nps_firm_{y}-12.parquet"
        if os.path.exists(fp):
            n = pd.read_parquet(fp, columns=["key", "n_emp", "active", "adpt", "n_new", "n_loss", "amt", "n_wkpl", "indcd"])
            NPS[y] = n[n.active].drop_duplicates("key").set_index("key")
    # 산업 침투율용: NPS 기업키 -> 최초 알려진 제공범위 출원일
    afk = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["key", "appl_date", "known_date"]).dropna(subset=["key"])
    afk = afk[afk.appl_date < {"dev": DEV_SEAL_APPL, "final": 20240101}.get(mode, 99999999)]
    key_first = afk.groupby("key").known_date.min()
    xw = pd.read_csv(f"{EXT}/firm_nps_indcd_ksic50_map.csv", dtype=str)
    k50_ind = set(xw.indcd)
    # NTIS 성과 IP(범위 밖 포함) -> 사업자번호
    RP = duckdb.sql(f"""select cast(r.Year as int) yr, try_cast(r.ProjectYear as int) pyr, r.IprType ipr, r.RegistType rt, r.is_kr, regexp_replace(p.corp_rgst_no,'[^0-9]','','g') bz
        from read_parquet('{EXTF}/ntis_rpatent_raw.parquet') r join (select distinct pjt_no, corp_rgst_no from read_parquet('{EXTF}/ntis_project_raw.parquet')
        where perform_grp in ('중소기업','중견기업','대기업')) p on r.ProjectID = p.pjt_no where cast(r.Year as int) <= {max(t0_list) - 1}""").df()
    log("NTIS outcome IP rows (firm)", len(RP))
    # C16: NTIS 성과 특허 중 제공 KR(50 KSIC) 연결분 -> 사업자번호
    LP = duckdb.sql(f"""select distinct cast(l.Year as int) yr, l.appl_no, regexp_replace(p.corp_rgst_no,'[^0-9]','','g') bz
        from read_parquet('{EXTF}/ntis_patent_link.parquet') l join (select distinct pjt_no, corp_rgst_no from read_parquet('{EXTF}/ntis_project_raw.parquet')
        where perform_grp in ('중소기업','중견기업','대기업')) p on l.ProjectID = p.pjt_no where l.cc = 'KR' and cast(l.Year as int) <= {max(t0_list) - 1}""").df()
    log("NTIS outcome provided-KR links (firm)", len(LP))
    # 국세청 폐업일(연결 출원인 목록 기준)
    ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["bizno", "end_dt"])
    ns["end_dt"] = pd.to_numeric(ns.end_dt.astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    bz_end = ns.dropna(subset=["end_dt"]).groupby("bizno").end_dt.min()

    # 행 단위 지표(벡터화 — groupby 평균이 파일럿의 apply(lambda) 결과와 같다; panel_dev_v1_slow와 대조 확인)
    NT["i_cont"] = (pd.to_numeric(NT.continuous, errors="coerce") == 1).astype(float)
    NT["i_sme"] = (NT.perform_grp == "중소기업").astype(float)
    NT["ph_dev"] = (NT.dev_phase == "개발연구").astype(float); NT["ph_app"] = (NT.dev_phase == "응용연구").astype(float)
    for s6 in SIXT:   # 파일럿 6T 인코딩(값에 'IT' 등 문자열이 없어 항상 0 — 재현을 위해 유지, C10에서 코드 기반으로 교정)
        NT["t6_" + s6] = NT.sixt.fillna("").str.contains(s6).astype(float)
    for cl in CLS:
        NT["cl_" + cl] = (NT.cls_L_canon == cl).astype(float)
    NT["k50a"] = NT.b2_dict_assigned.fillna(False).astype(float); NT["b1a"] = NT.b1_tfidf_assigned.fillna(False).astype(float)
    APG = ["mfg", "ict", "health", "agri", "energy", "prosvc"]; MNG = ["sme", "motie", "msit", "agri", "env", "health"]
    for gname in APG:
        NT["ap_" + gname] = (NT.agrp == gname).astype(float)
    for gname in MNG:
        NT["mn_" + gname] = (NT.mgrp == gname).astype(float)
    S6C = ["01", "02", "03", "04", "05", "06"]
    NT["s6c"] = NT.sixt_cd.astype(str).str[:2]
    for c6 in S6C:
        NT["s6_" + c6] = (NT.s6c == c6).astype(float)
    LC = {"도입기": "intro", "성장기": "growth", "성숙기": "mature"}
    for kk, vv in LC.items():
        NT["lc_" + vv] = (NT.tech_lifecycle == kk).astype(float)
    pts = f"{CACHE}/proj_ptsim.parquet"
    if os.path.exists(pts):
        NT = NT.merge(pd.read_parquet(pts, columns=["pjt_no", "year", "pt_sim_max", "pt_margin", "pt_sim_all"]), on=["pjt_no", "year"], how="left")
    else:
        NT["pt_sim_max"] = np.nan; NT["pt_margin"] = np.nan; NT["pt_sim_all"] = np.nan
    MEANC = ["ph_dev", "ph_app"] + ["t6_" + s for s in SIXT] + ["cl_" + c for c in CLS] + ["ap_" + a for a in APG] + ["mn_" + a for a in MNG] \
        + ["s6_" + c for c in S6C] + ["lc_" + v for v in LC.values()]
    NT10 = NT[NT.year >= 2010]   # 파일럿 재현: 경력·누적은 2010년부터 센다
    rows, pairs = [], []
    for t0 in t0_list:
        end = t0 * 10000 + 1231; lo = (t0 + 1) * 10000 + 101; hi = (t0 + 2) * 10000 + 1231
        if mode == "dev":
            assert hi < DEV_SEAL_APPL
        w = NT[NT.year.between(t0 - 2, t0)].copy()
        w["i_end"] = w.end_y.between(t0, t0 + 1).astype(float)
        g = w.groupby("bz")
        f = pd.DataFrame({"n_proj": np.log1p(g.pjt_no.nunique()), "funds": np.log1p(g.gov_funds.sum()), "n_yrs": g.year.nunique(),
                          "rec": t0 - g.year.max(), "sbiz_sh": g.sbiz_funds.sum() / (g.total_funds.sum() + 1),
                          "cont_sh": g.i_cont.mean(), "sme": g.i_sme.mean()})
        # [R5 감사 수정 2026-10-04] PROTOCOL_R4 §3 '연구비는 이듬해부터'(NTIS 조사·분석 이듬해 공개, avail_year=year+1):
        # 연구비 피처(funds, sbiz_sh)는 과제연도 t0-2..t0-1 값만 쓴다. t0년 과제만 있는 기업은 0(알려진 연구비 없음). 과제 레코드 자체(건수·연수·최근성)는 '레코드 연도 <= 원점' 규칙대로 유지.
        gl = w[w.year <= t0 - 1].groupby("bz")
        f["funds"] = np.log1p(gl.gov_funds.sum().reindex(f.index).fillna(0))
        f["sbiz_sh"] = (gl.sbiz_funds.sum() / (gl.total_funds.sum() + 1)).reindex(f.index).fillna(0)
        allp = NT10[NT10.year <= t0].groupby("bz")
        f["career"] = t0 - allp.year.min().reindex(f.index); f["n_proj_all"] = np.log1p(allp.pjt_no.nunique().reindex(f.index))
        f["corp_type"] = pd.to_numeric(pd.Series(f.index, index=f.index).str[3:5], errors="coerce")
        mm = g[MEANC + ["k50a", "b1a", "i_end", "pt_sim_max", "pt_margin", "pt_sim_all"]].mean()
        for c in ["ph_dev", "ph_app"] + ["t6_" + s for s in SIXT] + ["cl_" + c for c in CLS]:
            f[c] = mm[c]
        f["k50_assigned"] = mm["k50a"]
        f["k50_score"] = g.b2_dict_score.mean(); f["b1_score"] = g.b1_tfidf_score.mean()
        # 위험집합
        f = f[~(f.index.map(first_known) <= end)].copy()
        idx = f.index
        mm = mm.reindex(idx)
        # M2 NPS (파일럿)
        if t0 in NPS and t0 >= 2015:
            cur = NPS[t0]; prv = NPS.get(t0 - 1); prv2 = NPS.get(t0 - 2)
            keys = idx.map(k2)
            get = lambda tab, col: pd.Series([tab[col].get(k, np.nan) for k in keys], index=idx)
            emp = get(cur, "n_emp")
            f["nps_present"] = emp.notna().astype(float); f["log_emp"] = np.log1p(emp)
            f["g_emp1"] = np.log((emp + 1) / (get(prv, "n_emp") + 1)) if prv is not None else np.nan
            f["nps_age"] = t0 - pd.to_numeric(pd.Series([cur.adpt.get(k, None) for k in keys], index=idx).astype(str).str[:4], errors="coerce")
            # C06 NPS 고도화
            f["hire_rate"] = get(cur, "n_new") / (emp + 1); f["loss_rate"] = get(cur, "n_loss") / (emp + 1)
            f["wage"] = np.log1p(get(cur, "amt") / (emp + 1)); f["n_wkpl"] = get(cur, "n_wkpl")
            f["g_emp2"] = np.log((emp + 1) / (get(prv2, "n_emp") + 1)) if (prv2 is not None and t0 - 2 >= 2015) else np.nan
            # C07 산업 침투율 (NPS 업종코드)
            ind = get(cur, "indcd").astype(object)
            c = cur[["indcd"]].copy(); c["pat"] = (c.index.map(key_first) <= end).astype(float)
            c["i3"] = c.indcd.astype(str).str[:3]
            pen6 = c.groupby("indcd").pat.agg(["mean", "size"]); pen3 = c.groupby("i3").pat.agg(["mean", "size"])
            p0 = c.pat.mean()
            sm = lambda t, m=20: (t["mean"] * t["size"] + m * p0) / (t["size"] + m)
            pen6s = sm(pen6); pen3s = sm(pen3)
            f["ind_pen6"] = ind.map(pen6s).astype(float); f["ind_pen3"] = ind.astype(str).str[:3].map(pen3s).astype(float)
            f.loc[ind.isna(), "ind_pen3"] = np.nan
            f["ind_k50"] = np.where(ind.isna(), np.nan, ind.astype(str).isin(k50_ind).astype(float))
        else:
            for col in ["nps_present", "log_emp", "g_emp1", "nps_age", "hire_rate", "loss_rate", "wage", "n_wkpl", "g_emp2", "ind_pen6", "ind_pen3", "ind_k50"]:
                f[col] = np.nan
        # C01 텍스트 대응 고도화 A
        f["k50_max"] = g.b2_dict_score.max().reindex(idx); f["k50_n"] = np.log1p(g.k50a.sum().reindex(idx))
        f["b1_assigned"] = mm["b1a"]; f["b1_max"] = g.b1_tfidf_score.max().reindex(idx)
        f["k50_rank"] = f.k50_score.rank(pct=True); f["b1_rank"] = f.b1_score.rank(pct=True)
        wt = w[w.year == t0].groupby("bz")
        f["k50_cur"] = wt.b2_dict_score.mean().reindex(idx)
        # C02 특허 제목 중심 대응
        f["pt_mean"] = mm["pt_sim_max"]; f["pt_max"] = g.pt_sim_max.max().reindex(idx); f["pt_margin"] = mm["pt_margin"]; f["pt_all"] = mm["pt_sim_all"]
        # C03 적용분야·부처 비중
        for c in ["ap_" + a for a in APG] + ["mn_" + a for a in MNG]:
            f[c] = mm[c]
        # C10 6T 코드(교정)·기술수명주기
        for c in ["s6_" + c6 for c6 in S6C] + ["lc_" + v for v in LC.values()]:
            f[c] = mm[c]
        # C05 NTIS 성과 IP (성과연도 <= t0-1)
        r = RP[RP.yr <= t0 - 1]; r = r[r.bz.isin(idx)]
        def cnt(mask):
            return np.log1p(r[mask].groupby("bz").size().reindex(idx).fillna(0))
        f["o_pat"] = cnt(r.is_kr & (r.ipr == "특허")); f["o_pat_rg"] = cnt(r.is_kr & (r.ipr == "특허") & (r.rt == "등록"))
        f["o_des"] = cnt(r.is_kr & (r.ipr == "디자인")); f["o_util"] = cnt(r.is_kr & (r.ipr == "실용신안"))
        f["o_tm"] = cnt(r.is_kr & (r.ipr == "상표")); f["o_frgn"] = cnt(~r.is_kr)
        r2 = r[(r.yr >= t0 - 2)]
        f["o_pat_rec"] = np.log1p(r2[r2.is_kr & (r2.ipr == "특허")].groupby("bz").size().reindex(idx).fillna(0))
        # 감사용 엄격판(_s): 성과가 귀속된 과제연도도 <= t0-1 (원점 뒤 과제가 과거 특허를 성과로 끌어온 경우 제외)
        rs = r[r.pyr <= t0 - 1]
        def cnts(mask):
            return np.log1p(rs[mask].groupby("bz").size().reindex(idx).fillna(0))
        f["o_pat_s"] = cnts(rs.is_kr & (rs.ipr == "특허")); f["o_pat_rg_s"] = cnts(rs.is_kr & (rs.ipr == "특허") & (rs.rt == "등록"))
        f["o_des_s"] = cnts(rs.is_kr & (rs.ipr == "디자인")); f["o_util_s"] = cnts(rs.is_kr & (rs.ipr == "실용신안"))
        f["o_tm_s"] = cnts(rs.is_kr & (rs.ipr == "상표")); f["o_frgn_s"] = cnts(~rs.is_kr)
        rs2 = rs[rs.yr >= t0 - 2]
        f["o_pat_rec_s"] = np.log1p(rs2[rs2.is_kr & (rs2.ipr == "특허")].groupby("bz").size().reindex(idx).fillna(0))
        lp = LP[(LP.yr <= t0 - 1) & LP.bz.isin(idx)]
        f["o_prov"] = np.log1p(lp.groupby("bz").appl_no.nunique().reindex(idx).fillna(0)); f["o_prov_any"] = (f.o_prov > 0).astype(float)
        # C08 국세청 폐업(원점까지)
        f["nts_closed"] = (idx.map(bz_end) <= end).astype(float)
        # C09 과제 시점 구조
        f["end_soon"] = mm["i_end"]
        f["n_new_t0"] = np.log1p(w[(w.year == t0) & (w.i_cont == 1)].groupby("bz").pjt_no.nunique().reindex(idx).fillna(0))
        f["fund_t0_sh"] = (wt.gov_funds.sum().reindex(idx).fillna(0) / (g.gov_funds.sum().reindex(idx) + 1))
        f["n_prog"] = g.business_name.nunique().reindex(idx); f["max_fund"] = np.log1p(g.gov_funds.max().reindex(idx))
        f["tot_funds"] = np.log1p(g.total_funds.sum().reindex(idx))
        # 결과
        o = AF[(AF.appl_date >= lo) & (AF.appl_date <= hi)]
        f["y"] = idx.isin(set(o.bz)).astype(int); f["t0"] = t0
        f["linked"] = idx.map(dict(zip(firm.bz, firm.linked))).astype(float)   # 점검용(피처 금지)
        rows.append(f.reset_index().rename(columns={"index": "bz"}))
        ww = w[w.bz.isin(idx)]
        pairs.append(pd.concat([ww[["bz", "business_name"]].rename(columns={"business_name": "cat"}).assign(kind="prog"),
                                ww[["bz", "cls_M"]].rename(columns={"cls_M": "cat"}).assign(kind="clsM")]).dropna().drop_duplicates().assign(t0=t0))
        log(t0, len(f), "y", int(f.y.sum()), round(f.y.mean(), 4))
    D = pd.concat(rows, ignore_index=True)
    P = pd.concat(pairs, ignore_index=True)
    D = add_target_enc(D, P)
    return D


def add_target_enc(D, P, m=30):
    """C04 과거 전환율 인코딩: 원점 T 값 = 원점 t0 <= T-2 단위들의 (범주별) 전환율, 강도 m으로 전체율에 수축."""
    P = P.merge(D[["bz", "t0", "y"]], on=["bz", "t0"])
    out = []
    for T in sorted(D.t0.unique()):
        H = P[P.t0 <= T - 2]; cur = P[P.t0 == T]
        res = pd.DataFrame(index=D[D.t0 == T].bz.values)
        for kind in ["prog", "clsM"]:
            if len(H) == 0:
                res[f"te_{kind}_max"] = np.nan; res[f"te_{kind}_mean"] = np.nan; continue
            h = H[H.kind == kind]
            base = D[(D.t0 <= T - 2)].y.mean()
            st = h.groupby("cat").y.agg(["sum", "size"])
            te = (st["sum"] + m * base) / (st["size"] + m)
            c = cur[cur.kind == kind].copy(); c["te"] = c.cat.map(te).fillna(base)
            agg = c.groupby("bz").te.agg(["max", "mean"])
            res[f"te_{kind}_max"] = agg["max"].reindex(res.index); res[f"te_{kind}_mean"] = agg["mean"].reindex(res.index)
        res["t0"] = T
        out.append(res.reset_index().rename(columns={"index": "bz"}))
    return D.merge(pd.concat(out, ignore_index=True), on=["bz", "t0"], how="left")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    assert mode == "dev", "이번 라운드는 dev만 실행한다(final·future 금지)"
    D = build("dev")
    D.to_parquet(f"{CACHE}/panel_dev.parquet", index=False)
    log("saved", D.shape)
