# -*- coding: utf-8 -*-
r"""N7 도전자 묶음 피처(PLAN.md에 사전 선언). 모든 피처는 원점 t0 연말(end=t0*10000+1231)까지 알 수 있던 값만 쓴다.
각 함수: (B: 패널 한 원점분, t0, ctx) → B.index에 맞춘 DataFrame. ctx = load_ctx(mode)."""
import os, re
import numpy as np, pandas as pd
from n7lib import *

BUNDLES = {
    "C01_pay_delay": ["late_share_2y", "delay_mean_2y", "overdue_share", "rec_2y"],
    "C02_disposal": ["trf_out_2y", "trf_out_1y", "merge_2y", "surrender_2y"],
    "C03_portfolio_spend": ["live_age_mean", "hi_annl_share", "due_next_share", "spend_1y", "spend_trend"],
    "C04_coapplicant": ["co_pub_any", "co_pub_n", "co_pub_share", "co_pub_3y", "co_firm_share", "co_ntis_any"],
    "C05_ksic_firm_rnd": ["k_firm_sh", "k_firm_g", "k_nproj_g"],
    "C06_peer_entry_ksic": ["peer_entry_k", "peer_entry_k_rel"],
    "C07_design_utility": ["rl_log_des", "rl_des_2y", "rl_log_um", "rl_um_2y", "rl_du_trend"],
    "C08_foreign_family": ["fam_n_3y", "fam_share", "fam_prio_2y"],
    "C09_peer_entry_region": ["peer_entry_r", "peer_entry_r_rel", "sido"],
    "C10_nps_ext": ["g_emp2", "log_wkpl", "g_wage1", "d_hire_rate", "nps_gone"],
    "C11_ksic_mix": ["ksic_top", "ksic_top2", "ksic_top_share", "ksic_entropy"],
    "V05_ksic_firm_rnd_text": ["kt_firm_sh", "kt_firm_g", "kt_nproj_g"],
    "D01_inventor_size": ["inv_n_all", "inv_n_3y", "inv_per_pat", "inv_growth"],
    "D02_inventor_gov_exp": ["inv_gov_any", "inv_gov_share", "inv_pub_any", "inv_pub_share"],
}
PUB_RE = re.compile(r"대학|산학협력|과학기술원|KAIST|POSTECH|UNIST|DGIST|GIST|연구원|테크노파크|진흥원|국립|연구재단")
CORP_RE = re.compile(r"주식회사|\(주\)|㈜|유한회사|CO\.|LTD|INC|CORP", re.I)


def load_ctx(mode):
    ctx = {"mode": mode}
    ctx["R"] = load_registry()
    ctx["own"], ctx["sur"] = load_registry_extra()
    ctx["NPS"] = load_nps(mode)
    AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
    ctx["AF"] = AF
    ctx["ns"] = load_nts()
    NT = load_ntis(mode)
    ctx["first"] = ntis_first_year(NT)
    KY = pd.read_parquet(os.path.join(EXT, "ntis_project_ksic_year.parquet"))
    ctx["KYa"] = KY[KY.method == "a_link_pit"]; ctx["KYb"] = KY[KY.method == "b2_dict_soft"]
    # 공동출원인: 제공 KR 출원 출원인명(파이프 구분)
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원인명"])
    k.columns = ["appl_no", "names"]
    k["appl_no"] = k.appl_no.astype(str)
    nm = k.names.fillna("").str.split("|")
    k["n_names"] = nm.map(len)
    k["has_pub"] = nm.map(lambda L: any(PUB_RE.search(x) and not CORP_RE.search(x) for x in L))
    k["has_corp2"] = nm.map(lambda L: sum(bool(CORP_RE.search(x)) for x in L) >= 2)
    # 공동출원인 중 NTIS 수행기관(이름 정규화 일치)이 있는지: 원점별로 판정하려면 이름→첫 수행연도가 필요
    k["norm_names"] = nm.map(lambda L: [norm(x) for x in L])
    ctx["coapp"] = k.drop_duplicates("appl_no").set_index("appl_no")
    ntn = load_ntis(mode, cols=("year", "research_agency"))
    ntn = ntn[ntn.year >= 2010].dropna(subset=["research_agency"])
    ntn["n"] = ntn.research_agency.map(norm)
    ctx["name_first"] = ntn.groupby("n").year.min()
    ctx["fev"] = pd.read_parquet(os.path.join(CACHE, "foreign_events.parquet"))
    return ctx


def _add_years(d, n):
    d = d.astype("Int64")
    y = d // 10000; md = d % 10000
    md = md.where(md != 229, 228)
    return (y + n) * 10000 + md


def _to_dt(x):
    return pd.to_datetime(x.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def f_C01(B, t0, ctx):
    """납부 지연(N4 C04와 같은 정의)"""
    R = ctx["R"]; end = t0 * 10000 + 1231; y2 = end - 20000
    py = R["py"]; py = py[(py.d <= end) & py.applicantCd.isin(set(B.applicantCd))]
    w = py[(py.d > y2) & (py.statAnnl >= 4) & py.rgstDate.notna()].copy()
    w["due"] = _add_years(w.rgstDate, (w.statAnnl - 1).astype(int))
    w["delay"] = (_to_dt(w.d) - _to_dt(w.due)).dt.days
    w["late"] = (w.delay > 0).astype(float)
    g = w.groupby("applicantCd")
    o = pd.DataFrame(index=B.index); a = B["applicantCd"]
    o["late_share_2y"] = a.map(g.late.mean()).astype(float)
    o["delay_mean_2y"] = a.map(g.delay.apply(lambda s: s.clip(lower=0).mean())).astype(float)
    pt = R["pt"]; live = pt[(pt.rgstDate <= end) & ~(pt.ext_d <= end) & pt.applicantCd.isin(set(B.applicantCd))]
    paid = py.groupby("rgst_no").lastAnnl.max()
    lv = live[["rgst_no", "applicantCd", "rgstDate"]].copy()
    lv["la"] = lv.rgst_no.map(paid)
    lv = lv[lv.la.notna()]
    lv["pthru"] = _add_years(lv.rgstDate, lv.la.astype(int))
    lv["over"] = (lv.pthru <= end).astype(float)
    o["overdue_share"] = a.map(lv.groupby("applicantCd").over.mean()).astype(float)
    rec = R["rec"]; rec = rec[(rec.d > y2) & (rec.d <= end)]
    o["rec_2y"] = a.map(rec.groupby("applicantCd").rgst_no.nunique()).fillna(0).astype(float)
    return o


def f_C02(B, t0, ctx):
    """권리 처분(N4 C11): owner 테이블 권리자 변경·합병/분할, 포기 말소 — 사건일 ≤ t0"""
    R = ctx["R"]; end = t0 * 10000 + 1231; y1 = end - 10000; y2 = end - 20000
    pt = R["pt"]; a = B["applicantCd"]
    p = pt[pt.rgstDate <= y1]
    npv = a.map(p[~(p.ext_d <= y1)].groupby("applicantCd").rgst_no.nunique()).fillna(0).astype(float)
    ow, sur = ctx["own"], ctx["sur"]
    w2 = ow[(ow.d > y2) & (ow.d <= end)]
    o = pd.DataFrame(index=B.index)
    o["trf_out_2y"] = a.map(w2[w2.kind == "assign"].groupby("applicantCd").rgst_no.nunique()).fillna(0).astype(float) / (npv + 1)
    o["trf_out_1y"] = a.map(w2[(w2.kind == "assign") & (w2.d > y1)].groupby("applicantCd").rgst_no.nunique()).fillna(0).astype(float) / (npv + 1)
    o["merge_2y"] = (a.map(w2[w2.kind == "merge"].groupby("applicantCd").rgst_no.nunique()).fillna(0) > 0).astype(float)
    s2 = sur[(sur.d > y2) & (sur.d <= end)]
    o["surrender_2y"] = a.map(s2.groupby("applicantCd").rgst_no.nunique()).fillna(0).astype(float) / (npv + 1)
    return o


def f_C03(B, t0, ctx):
    """포트폴리오 연령·연차료 지출(N4 C05)"""
    R = ctx["R"]; end = t0 * 10000 + 1231; y1 = end - 10000; y2 = end - 20000
    pt = R["pt"]; aset = set(B.applicantCd)
    live = pt[(pt.rgstDate <= end) & ~(pt.ext_d <= end) & pt.applicantCd.isin(aset)][["rgst_no", "applicantCd", "rgstDate", "applDate"]].copy()
    py = R["py"]; py = py[(py.d <= end) & py.applicantCd.isin(aset)]
    paid = py.groupby("rgst_no").lastAnnl.max()
    live["age"] = t0 - live.applDate // 10000
    live["la"] = live.rgst_no.map(paid)
    live["hi"] = (live.la + 1 >= 7).astype(float)
    pth = _add_years(live.rgstDate, live.la.fillna(3).astype(int))
    live["due_next"] = ((pth > end) & (pth <= end + 10000)).astype(float)
    g = live.groupby("applicantCd")
    o = pd.DataFrame(index=B.index); a = B["applicantCd"]
    o["live_age_mean"] = a.map(g.age.mean()).astype(float)
    o["hi_annl_share"] = a.map(g.hi.mean()).astype(float)
    o["due_next_share"] = a.map(g.due_next.mean()).astype(float)
    S1 = py[py.d > y1].groupby("applicantCd").payAmount.sum()
    S2 = py[(py.d > y2) & (py.d <= y1)].groupby("applicantCd").payAmount.sum()
    s1 = a.map(S1).fillna(0).astype(float); s2 = a.map(S2).fillna(0).astype(float)
    o["spend_1y"] = np.log1p(s1 / 1000)
    o["spend_trend"] = np.log((s1 / 1000 + 1) / (s2 / 1000 + 1))
    return o


def f_C04(B, t0, ctx):
    """산학연 공동출원: t0까지 알려진 자사 제공 KR 출원 중 대학·공공연구기관 공동출원, 타 기업 공동출원,
    공동출원인 중 t0까지 NTIS 수행 이력이 있는 기관(이름 정규화 일치, 과제연도 ≤ t0)"""
    end = t0 * 10000 + 1231
    AF = ctx["AF"]; co = ctx["coapp"]
    kn = AF[(AF.known_date <= end) & AF.applicantCd.isin(set(B.applicantCd))][["applicantCd", "appl_no", "known_date"]].drop_duplicates(["applicantCd", "appl_no"]).copy()
    kn["appl_no"] = kn.appl_no.astype(str)
    j = kn.join(co[["n_names", "has_pub", "has_corp2", "norm_names"]], on="appl_no")
    j["has_pub"] = j.has_pub.fillna(False).astype(float); j["has_corp2"] = j.has_corp2.fillna(False).astype(float)
    nf = ctx["name_first"]; okn = set(nf[nf <= t0].index)
    j["co_ntis"] = j.norm_names.map(lambda L: float(isinstance(L, list) and len(L) >= 2 and any(x in okn for x in L)))
    g = j.groupby("applicantCd")
    o = pd.DataFrame(index=B.index); a = B["applicantCd"]
    npub = a.map(g.has_pub.sum()).fillna(0).astype(float); nall = a.map(g.size()).fillna(1).astype(float)
    o["co_pub_any"] = (npub > 0).astype(float)
    o["co_pub_n"] = np.log1p(npub)
    o["co_pub_share"] = npub / nall
    o["co_pub_3y"] = np.log1p(a.map(j[j.known_date > end - 30000].groupby("applicantCd").has_pub.sum()).fillna(0).astype(float))
    o["co_firm_share"] = a.map(g.has_corp2.mean()).astype(float)
    o["co_ntis_any"] = (a.map(g.co_ntis.sum()).fillna(0) > 0).astype(float)
    return o


def _ksic_rnd(B, t0, ctx, KY, pre):
    ky = KY[KY.avail_year <= t0]
    pv_f = ky.pivot_table(index="ksic", columns="year", values="gov_funds_firm", aggfunc="sum")
    pv_n = ky.pivot_table(index="ksic", columns="year", values="n_proj", aggfunc="sum")
    pv_g = ky.pivot_table(index="ksic", columns="year", values="gov_funds", aggfunc="sum")
    def grow(pv):
        a_ = pv[[y for y in pv.columns if t0 - 2 <= y <= t0 - 1]].sum(1); b_ = pv[[y for y in pv.columns if t0 - 4 <= y <= t0 - 3]].sum(1)
        g = np.log((a_ + 1) / (b_ + 1)); g[b_ <= 0] = np.nan
        return g
    ly = t0 - 1
    sh = (pv_f[ly] / pv_g[ly].replace(0, np.nan)) if ly in pv_f.columns else pd.Series(dtype=float)
    W = ksic_weights(B, t0, ctx["AF"])
    o = pd.DataFrame(index=B.index)
    o[f"{pre}_firm_sh"] = wavg(W, sh, B); o[f"{pre}_firm_g"] = wavg(W, grow(pv_f), B); o[f"{pre}_nproj_g"] = wavg(W, grow(pv_n), B)
    return o


def f_C05(B, t0, ctx):
    """KSIC별 '기업 수행' 정부연구비 비중·성장, 배분 과제 수 성장(a_link_pit: 연구비 ≤ t0−1, 배분표 성과 ≤ 연도−2), 자사 KSIC 구성 가중"""
    return _ksic_rnd(B, t0, ctx, ctx["KYa"], "k")


def f_V05(B, t0, ctx):
    """C05의 KSIC 연결을 공개 정의문 텍스트 대응(b2_dict_soft)으로 바꾼 변형(avail_year ≤ t0)"""
    return _ksic_rnd(B, t0, ctx, ctx["KYb"], "kt")


def _eligible_at(s, ctx):
    """원점 s의 '특허 법인·미폐업·NTIS 수행 이력 없음' 집합(국민연금 불필요한 축약 정의)과 (s, s+2] 첫 수행 여부.
    s+2 ≤ t0일 때만 호출(t0 시점에 관측 완료)."""
    end = s * 10000 + 1231
    AF, R, ns = ctx["AF"], ctx["R"], ctx["ns"]
    reg = set(R["pt"][R["pt"].rgstDate <= end].applicantCd)
    kn = AF[(AF.known_date <= end) & AF.applicantCd.isin(reg)]
    e = ns.set_index("applicantCd")
    codes = pd.Index(kn.applicantCd.unique()).intersection(e.index)
    e = e.loc[codes]
    e = e[~(e.end <= end)]
    fy = e.bizno.str.replace(r"[^0-9]", "", regex=True).map(ctx["first"])
    e = e[~(fy <= s)]
    fy = fy.loc[e.index]
    yv = ((fy > s) & (fy <= s + 2)).astype(float)
    return e.index, yv, kn


def f_C06(B, t0, ctx):
    """같은 KSIC 동종 특허기업의 최근 NTIS 진입률: 원점 t0−2 무이력 특허 법인 중 (t0−2, t0]에 첫 수행한 비율(KSIC 구성 가중, 평활 강도 20)"""
    s = t0 - 2; end_s = s * 10000 + 1231
    idx, yv, kn = _eligible_at(s, ctx)
    kn = kn[kn.applicantCd.isin(idx)]
    W = kn.groupby(["applicantCd", "ksic"]).appl_no.nunique().unstack(fill_value=0)
    W = W.div(W.sum(1), axis=0)
    yy = yv.reindex(W.index).fillna(0)
    m = yy.mean()
    rate = (W.mul(yy, axis=0).sum() + m * 20) / (W.sum() + 20)
    Wc = ksic_weights(B, t0, ctx["AF"])
    o = pd.DataFrame(index=B.index)
    o["peer_entry_k"] = wavg(Wc, rate, B)
    o["peer_entry_k_rel"] = o.peer_entry_k / m
    return o


def nps_sido(snap):
    fc = os.path.join(CACHE, f"nps_sido_{snap}.parquet")
    if os.path.exists(fc):
        return pd.read_parquet(fc).set_index("key").sido
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{snap.replace('-', '')}.parquet"), columns=["name", "biz6", "ldong", "n_emp", "stat"])
    d = d[d.stat == "1"]
    d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm)
    d = d.sort_values("n_emp", ascending=False).drop_duplicates("key")
    d["sido"] = pd.to_numeric(d.ldong.astype(str).str[:2], errors="coerce")
    d[["key", "sido"]].to_parquet(fc, index=False)
    return d.set_index("key").sido


def f_C09(B, t0, ctx):
    """지역(시도) 동종 진입률: 국민연금 t0-12 스냅샷 최대 사업장 법정동 시도. (t0−2, t0] 첫 수행률(평활 20)"""
    sd = nps_sido(f"{t0}-12")
    A2K = ctx.setdefault("A2K", a2key())
    s = t0 - 2
    idx, yv, _ = _eligible_at(s, ctx)
    reg = pd.Series([sd.get(A2K.get(c), np.nan) if A2K.get(c) else np.nan for c in idx], index=idx)
    df = pd.DataFrame({"sido": reg, "y": yv.reindex(idx).fillna(0)}).dropna()
    m = df.y.mean()
    g = df.groupby("sido").y.agg(["sum", "size"])
    rate = (g["sum"] + m * 20) / (g["size"] + 20)
    o = pd.DataFrame(index=B.index)
    sid = B["key"].map(lambda k: sd.get(k, np.nan) if isinstance(k, str) else np.nan).astype(float)
    o["sido"] = sid
    o["peer_entry_r"] = sid.map(rate).astype(float)
    o["peer_entry_r_rel"] = o.peer_entry_r / m
    return o


def f_C07(B, t0, ctx):
    """디자인권·실용신안(business_rights 권리목록, 등록일 ≤ t0). 조회 성공 출원인만 값, 나머지 NaN. 시점 정합 점검 대상(PLAN §5)."""
    if "rl" not in ctx:
        c = ro(BRS)
        ok = set(pd.read_sql("select applicant_cd from applicant_query where status='OK'", c).applicant_cd)
        rl = pd.read_sql("select applicant_cd, substr(rgst_no,1,2) p, rgst_no, rgst_date from right_list where substr(rgst_no,1,2) in ('10','20','30')", c)
        rl["d"] = pd.to_numeric(rl.rgst_date.str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
        ctx["rl"] = rl; ctx["rl_ok"] = ok
    rl, ok = ctx["rl"], ctx["rl_ok"]
    end = t0 * 10000 + 1231
    a = B["applicantCd"]; okm = a.isin(ok)
    o = pd.DataFrame(index=B.index)
    def cnt(p, lo=None, hi=end):
        r = rl[(rl.p == p) & (rl.d <= hi)]
        if lo is not None:
            r = r[r.d > lo]
        return a.map(r.groupby("applicant_cd").size()).fillna(0).astype(float).where(okm)
    td, rd, pd2 = cnt("30"), cnt("30", end - 20000), cnt("30", end - 40000, end - 20000)
    tu, ru, pu = cnt("20"), cnt("20", end - 20000), cnt("20", end - 40000, end - 20000)
    o["rl_log_des"] = np.log1p(td); o["rl_des_2y"] = np.log1p(rd)
    o["rl_log_um"] = np.log1p(tu); o["rl_um_2y"] = np.log1p(ru)
    o["rl_du_trend"] = np.log((rd + ru + 1) / (pd2 + pu + 1))
    return o


def f_C08(B, t0, ctx):
    """5개국 해외 패밀리(N4 C14): US·EP·JP·CN 공개일 ≤ t0"""
    end = t0 * 10000 + 1231
    ev = ctx["fev"]; ev = ev[(ev.known <= end) & ev.applicantCd.isin(set(B.applicantCd))]
    a = B["applicantCd"]
    o = pd.DataFrame(index=B.index)
    o["fam_n_3y"] = np.log1p(a.map(ev[ev.known > end - 30000].drop_duplicates(["applicantCd", "fkey"]).groupby("applicantCd").size()).fillna(0).astype(float))
    o["fam_share"] = (a.map(ev.groupby("applicantCd").kr_appl.nunique()).fillna(0) / B["n_app"]).astype(float)
    o["fam_prio_2y"] = np.log1p(a.map(ev[ev.prio > end - 20000].drop_duplicates(["applicantCd", "fkey"]).groupby("applicantCd").size()).fillna(0).astype(float))
    return o


def f_C10(B, t0, ctx):
    """국민연금 확장: 2년 고용 증감, 활성 사업장 수, 1인당 고지액 1년 변화, 취득률 변화, 과거 활성→현재 비활성"""
    NPS = ctx["NPS"]; keys = B["key"]
    cur = NPS[f"{t0}-12"]; prv = NPS.get(f"{t0 - 1}-12"); p2 = NPS.get(f"{t0 - 2}-12")
    o = pd.DataFrame(index=B.index)
    lk = lambda s: keys.map(lambda k: s.get(k, np.nan) if isinstance(k, str) else np.nan).astype(float)
    emp = np.expm1(B["log_emp"])
    if p2 is not None:
        o["g_emp2"] = np.log((emp + 1) / (lk(p2[p2.active].set_index("key").n_emp) + 1))
    else:
        o["g_emp2"] = np.nan
    ca = cur[cur.active].set_index("key")
    o["log_wkpl"] = np.log1p(lk(ca.n_act_wkpl))
    def per(d, col):
        d = d[d.active].set_index("key"); e = d.n_emp.replace(0, np.nan)
        return lk(d[col] / e)
    if prv is not None:
        o["g_wage1"] = np.log((per(cur, "amt") + 1) / (per(prv, "amt") + 1))
        o["d_hire_rate"] = per(cur, "n_new") - per(prv, "n_new")
    else:
        o["g_wage1"] = np.nan; o["d_hire_rate"] = np.nan
    act = keys.map(lambda k: bool(ca.active.get(k, False)) if isinstance(k, str) else False).astype(bool)
    past = set()
    for s, d in NPS.items():
        if s < f"{t0}-12":
            past |= set(d.key[d.active])
    o["nps_gone"] = (keys.map(lambda k: k in past if isinstance(k, str) else False).astype(bool) & ~act).astype(float)
    return o


def f_C11(B, t0, ctx):
    """KSIC 구성 직접(제공 데이터 KSIC 라벨): 주력 KSIC 5자리·2자리 코드, 주력 비중, KSIC 엔트로피"""
    W = ksic_weights(B, t0, ctx["AF"])
    W = W.sort_values(["applicantCd", "w", "ksic"], ascending=[True, False, True])
    top = W.drop_duplicates("applicantCd").set_index("applicantCd")
    ent = W.assign(e=-W.w * np.log(W.w)).groupby("applicantCd").e.sum()
    a = B["applicantCd"]
    o = pd.DataFrame(index=B.index)
    o["ksic_top"] = pd.to_numeric(a.map(top.ksic), errors="coerce").astype(float)
    o["ksic_top2"] = (o.ksic_top // 1000).astype(float)
    o["ksic_top_share"] = a.map(top.w).astype(float)
    o["ksic_entropy"] = a.map(ent).astype(float)
    return o


def _inv_ctx(ctx):
    """등록원부 발명자 코드 × 등록특허(등록일), NTIS 성과 연결 등록특허(첫 레코드 연도), 대학·공공연구기관 권리자 특허"""
    if "inv" in ctx:
        return
    fc = os.path.join(CACHE, "reg_inv.parquet")
    if os.path.exists(fc):
        inv = pd.read_parquet(fc)
    else:
        c = ro(REG)
        inv = pd.read_sql("select rgst_no, inventorCd, inventorName, inventorAddr from inventor", c)
        # 발명자 코드는 83%가 비어 있다 → 키 = 코드(있으면) 또는 '이름|주소 앞 두 토큰(시도 시군구)'
        ad = inv.inventorAddr.fillna("").str.split().map(lambda L: " ".join(L[:2]))
        key = inv.inventorName.fillna("").str.replace(r"\s", "", regex=True) + "|" + ad
        has = inv.inventorCd.fillna("").str.len() > 0
        inv["inventorCd"] = np.where(has, "C" + inv.inventorCd.fillna(""), "N" + key)
        inv = inv[inv.inventorName.fillna("").str.len() > 0][["rgst_no", "inventorCd"]].drop_duplicates()
        inv.to_parquet(fc, index=False)
    fp = os.path.join(CACHE, "reg_pubapp.parquet")
    if os.path.exists(fp):
        pubapp = pd.read_parquet(fp)
    else:
        c = ro(REG)
        a = pd.read_sql("select rgst_no, applicantCd, applicantName from applicant", c)
        a["pub"] = a.applicantName.fillna("").map(lambda x: bool(PUB_RE.search(x)) and not CORP_RE.search(x))
        pubapp = a[a.pub][["rgst_no", "applicantCd"]].drop_duplicates()
        pubapp.to_parquet(fp, index=False)
    pt = ctx["R"]["pt"][["rgst_no", "rgstDate", "applicantCd"]].drop_duplicates()
    lk = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["rgst_no", "Year"]).dropna(subset=["rgst_no"])
    lk["Year"] = pd.to_numeric(lk.Year, errors="coerce")
    gy = lk.groupby("rgst_no").Year.min()
    rd = pt.drop_duplicates("rgst_no").set_index("rgst_no").rgstDate
    ctx["inv"] = inv.merge(pt, on="rgst_no")          # (rgst_no, inventorCd, rgstDate, applicantCd)
    ctx["gov_rg"] = gy
    ctx["pub_rg"] = pubapp.assign(rgstDate=pubapp.rgst_no.map(rd))


def f_D01(B, t0, ctx):
    """발명자 규모·증가: 자사 등록특허(등록일 ≤ t0)의 고유 발명자 코드 수"""
    _inv_ctx(ctx)
    end = t0 * 10000 + 1231
    iv = ctx["inv"]; iv = iv[(iv.rgstDate <= end) & iv.applicantCd.isin(set(B.applicantCd))]
    a = B["applicantCd"]
    n_all = a.map(iv.groupby("applicantCd").inventorCd.nunique()).fillna(0).astype(float)
    r3 = iv[iv.rgstDate > end - 30000]; p3 = iv[(iv.rgstDate > end - 60000) & (iv.rgstDate <= end - 30000)]
    n3 = a.map(r3.groupby("applicantCd").inventorCd.nunique()).fillna(0).astype(float)
    np3 = a.map(p3.groupby("applicantCd").inventorCd.nunique()).fillna(0).astype(float)
    npat = a.map(iv.groupby("applicantCd").rgst_no.nunique()).fillna(0).astype(float)
    o = pd.DataFrame(index=B.index)
    o["inv_n_all"] = np.log1p(n_all); o["inv_n_3y"] = np.log1p(n3)
    o["inv_per_pat"] = n_all / npat.replace(0, np.nan)
    o["inv_growth"] = np.log((n3 + 1) / (np3 + 1))
    return o


def f_D02(B, t0, ctx):
    """발명자의 정부R&D·공공연구 경험: 자사 발명자(등록일 ≤ t0 자사 특허)가 다른 출원인의
    NTIS 성과 등록특허(첫 성과 레코드 Year ≤ t0, 등록일 ≤ t0) 또는 대학·공공연구기관 권리 등록특허(등록일 ≤ t0)에 등장"""
    _inv_ctx(ctx)
    end = t0 * 10000 + 1231
    iv = ctx["inv"]; iv = iv[iv.rgstDate <= end]
    fi = iv[iv.applicantCd.isin(set(B.applicantCd))][["applicantCd", "inventorCd"]].drop_duplicates()
    gy = ctx["gov_rg"]; gset = set(gy[gy <= t0].index)
    gv = iv[iv.rgst_no.isin(gset)][["inventorCd", "applicantCd"]].drop_duplicates().rename(columns={"applicantCd": "other"})
    pr = ctx["pub_rg"]; pr = pr[pr.rgstDate <= end]
    pv = iv[iv.rgst_no.isin(set(pr.rgst_no))][["inventorCd", "applicantCd"]].drop_duplicates().rename(columns={"applicantCd": "other"})
    a = B["applicantCd"]; ntot = a.map(fi.groupby("applicantCd").size()).fillna(0).astype(float)
    o = pd.DataFrame(index=B.index)
    for src, nm in [(gv, "gov"), (pv, "pub")]:
        m = fi.merge(src, on="inventorCd"); m = m[m.other != m.applicantCd]
        k = a.map(m.groupby("applicantCd").inventorCd.nunique()).fillna(0).astype(float)
        o[f"inv_{nm}_any"] = (k > 0).astype(float)
        o[f"inv_{nm}_share"] = k / ntot.replace(0, np.nan)
    return o


FUNCS = {"C01_pay_delay": f_C01, "C02_disposal": f_C02, "C03_portfolio_spend": f_C03, "C04_coapplicant": f_C04,
         "C05_ksic_firm_rnd": f_C05, "C06_peer_entry_ksic": f_C06, "C07_design_utility": f_C07, "C08_foreign_family": f_C08,
         "C09_peer_entry_region": f_C09, "C10_nps_ext": f_C10, "C11_ksic_mix": f_C11, "V05_ksic_firm_rnd_text": f_V05,
         "D01_inventor_size": f_D01, "D02_inventor_gov_exp": f_D02}


def build_features(D, ctx, which=None):
    which = which or list(FUNCS)
    parts = []
    for t0, B in D.groupby("t0", sort=True):
        fs = [FUNCS[w](B, t0, ctx) for w in which]
        parts.append(pd.concat(fs, axis=1))
        log("feat", t0, len(B))
    return pd.concat(parts).loc[D.index]
