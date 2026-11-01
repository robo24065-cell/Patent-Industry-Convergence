# -*- coding: utf-8 -*-
r"""N4 도전자 묶음 피처(PLAN.md에 사전 선언). 모든 피처는 원점 t0 연말(end=t0*10000+1231)까지 알 수 있던 값만 쓴다.
각 함수: (B: 기본 패널 한 원점분 DataFrame, t0, ctx) → B.index에 맞춘 DataFrame.
ctx는 load_ctx(mode)로 한 번만 만든다."""
import os
import numpy as np, pandas as pd
from n4lib import *

BUNDLES = {
    "C01_nps_exit": ["nps_withdrawn", "nps_exit1y", "g_emp2", "log_wkpl", "nps_gone"],
    "C02_ntis_project": ["ntis_any5", "ntis_n3", "ntis_funds", "ntis_active", "ntis_trend", "ntis_yrs_since"],
    "C03_ntis_patent": ["gov_pat_n", "gov_pat_share", "gov_pat_recent"],
    "C04_pay_delay": ["late_share_2y", "delay_mean_2y", "overdue_share", "rec_2y"],
    "C05_portfolio_age": ["live_age_mean", "hi_annl_share", "due_next_share", "spend_1y", "spend_trend"],
    "C06_rl_patent_oos": ["rl_log_pat", "rl_oos_share", "rl_pat_2y", "rl_pat_trend"],
    "C07_design_utility": ["rl_log_des", "rl_des_2y", "rl_log_um", "rl_um_2y", "rl_du_trend"],
    "C08_trademark": ["rl_log_tm", "rl_tm_2y", "rl_tm_trend"],
    "C09_dart_fin": ["dart_has", "dart_lev", "dart_opm", "dart_roa", "dart_neg_eq", "dart_impair", "dart_g_rev", "dart_log_assets", "dart_curr"],
    "C10_ksic_peer_closure": ["x_close_ksic", "x_close_ksic_rel"],
    "C11_disposal": ["trf_out_2y", "merge_2y", "surrender_2y", "trf_out_1y"],
    "C13_nps_flow": ["g_wage1", "d_loss_rate", "d_hire_rate"],
    "C14_foreign_family": ["fam_n_3y", "fam_share", "fam_prio_2y"],
}


def load_ctx(mode, pop_codes):
    ctx = {"mode": mode}
    ctx["R"] = load_registry()
    ctx["NPS"] = load_nps(mode)
    # NTIS 과제: 수행기관 사업자번호(2010~ 거의 100%) → 연도별 과제 수·정부연구비
    pj = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["year", "corp_rgst_no", "gov_funds"])
    pj["bizno"] = pj.corp_rgst_no.fillna("").str.replace(r"\D", "", regex=True)
    pj = pj[pj.bizno.str.len() == 10]
    ctx["pj"] = pj.groupby(["bizno", "year"]).agg(n=("gov_funds", "size"), funds=("gov_funds", "sum")).reset_index()
    # NTIS 성과 특허(제공 KR 연결)
    nl = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["appl_no", "Year"])
    nl["Year"] = pd.to_numeric(nl.Year, errors="coerce")
    ctx["nl"] = nl.groupby("appl_no").Year.min().astype(float)  # 해당 출원의 가장 이른 성과 레코드 연도
    # 권리목록(2026 스냅샷이지만 rgst_date ≤ t0 인 권리만 씀). 조회 성공(OK) 출원인만 값, 나머지 NaN.
    c = ro(BRS)
    ok = pd.read_sql("select applicant_cd from applicant_query where status='OK'", c).applicant_cd
    ok = set(ok) & set(pop_codes)
    rl = pd.read_sql("select applicant_cd, substr(rgst_no,1,2) p, rgst_no, rgst_date from right_list", c)
    rl = rl[rl.applicant_cd.isin(ok)]
    rl["d"] = pd.to_numeric(rl.rgst_date.str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    rl["typ"] = np.select([rl.p == "10", rl.p == "20", rl.p == "30", rl.p.isin(["40", "41", "42", "43", "44", "45"])], ["pat", "um", "des", "tm"], "oth")
    prov = set(ctx["R"]["pt"].rgst_no)
    rl["in_prov"] = rl.rgst_no.isin(prov)
    ctx["rl"] = rl[["applicant_cd", "typ", "d", "in_prov"]]
    ctx["rl_ok"] = ok
    # DART 재무
    dd = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"))
    dd["avail"] = pd.to_numeric(dd.avail_date.dt.strftime("%Y%m%d"), errors="coerce")
    dd["ofs"] = (dd.fs == "OFS").astype(int)
    ctx["dart"] = dd
    ctx["own"], ctx["sur"] = load_registry_extra()
    AFd = load_app_firm(); AFd = AFd[(AFd.applicantCd.astype(str).str[0] == "1") & AFd.known_date.notna()]
    ctx["fev"] = load_foreign_events(AFd)
    return ctx


def _cnt(df, col, idx):
    return df.groupby(col).size().reindex(idx).fillna(0)


def f_C01(B, t0, ctx):
    NPS = ctx["NPS"]
    cur = NPS[f"{t0}-12"].set_index("key")
    prv = NPS.get(f"{t0 - 1}-12")
    p2 = NPS.get(f"{t0 - 2}-12")
    keys = B["key"]
    o = pd.DataFrame(index=B.index)
    inc = keys.map(lambda k: k in cur.index if isinstance(k, str) else False)
    act = keys.map(lambda k: bool(cur.active.get(k, False)) if isinstance(k, str) else False)
    inc = inc.astype(bool); act = act.astype(bool)
    o["nps_withdrawn"] = (inc & ~act).astype(float)
    if prv is not None:
        pa = prv.set_index("key").active
        was = keys.map(lambda k: bool(pa.get(k, False)) if isinstance(k, str) else False)
        o["nps_exit1y"] = (was.astype(bool) & ~act).astype(float)
    else:
        o["nps_exit1y"] = np.nan
    # 과거 스냅샷(≤t0)에 활성으로 있었지만 지금 활성이 아님
    past = set()
    for s, d in NPS.items():
        if s < f"{t0}-12":
            past |= set(d.key[d.active])
    o["nps_gone"] = (keys.map(lambda k: k in past if isinstance(k, str) else False).astype(bool) & ~act).astype(float)
    past3 = set()
    for s_, d in NPS.items():
        if f"{t0 - 3}-12" <= s_ < f"{t0}-12":
            past3 |= set(d.key[d.active])
    o["nps_gone3"] = (keys.map(lambda k: k in past3 if isinstance(k, str) else False).astype(bool) & ~act).astype(float)
    if p2 is not None:
        e2 = p2[p2.active].set_index("key").n_emp
        emp = np.expm1(B["log_emp"])
        e0 = keys.map(lambda k: e2.get(k, np.nan) if isinstance(k, str) else np.nan).astype(float)
        o["g_emp2"] = np.log((emp + 1) / (e0 + 1))
    else:
        o["g_emp2"] = np.nan
    ca = cur[cur.active].n_act_wkpl
    o["log_wkpl"] = np.log1p(keys.map(lambda k: ca.get(k, np.nan) if isinstance(k, str) else np.nan).astype(float))
    return o


def f_C02(B, t0, ctx):
    pj = ctx["pj"]; pj = pj[pj.year <= t0]
    o = pd.DataFrame(index=B.index)
    bz = B["bizno"]
    def agg(lo, hi, col):
        s = pj[pj.year.between(lo, hi)].groupby("bizno")[col].sum()
        return bz.map(s).fillna(0).astype(float)
    o["ntis_any5"] = (agg(t0 - 4, t0, "n") > 0).astype(float)
    o["ntis_n3"] = np.log1p(agg(t0 - 2, t0, "n"))
    o["ntis_funds"] = np.log1p(agg(t0 - 3, t0 - 1, "funds") / 1e6)  # 연구비는 이듬해부터(y ≤ t0-1)
    o["ntis_active"] = (agg(t0, t0, "n") > 0).astype(float)
    o["ntis_trend"] = np.log((agg(t0 - 1, t0, "n") + 1) / (agg(t0 - 3, t0 - 2, "n") + 1))
    last = pj[pj.year >= 2010].groupby("bizno").year.max()
    o["ntis_yrs_since"] = (t0 - bz.map(last)).astype(float)
    return o


def f_C03(B, t0, ctx, AF):
    end = t0 * 10000 + 1231
    kn = AF[(AF.known_date <= end) & AF.applicantCd.isin(set(B.applicantCd))][["applicantCd", "appl_no", "known_date"]].drop_duplicates(["applicantCd", "appl_no"])
    yr = kn.appl_no.map(ctx["nl"]).astype(float)
    kn = kn.assign(gov=(yr <= t0).astype(int))
    g = kn.groupby("applicantCd")
    ng = g.gov.sum(); na = g.size()
    rec = kn[kn.known_date > end - 30000].groupby("applicantCd").gov.sum()
    o = pd.DataFrame(index=B.index)
    a = B["applicantCd"]
    o["gov_pat_n"] = np.log1p(a.map(ng).fillna(0).astype(float))
    o["gov_pat_share"] = (a.map(ng).fillna(0) / a.map(na).fillna(1)).astype(float)
    o["gov_pat_recent"] = np.log1p(a.map(rec).fillna(0).astype(float))
    return o


def _add_years(d, n):
    """YYYYMMDD 정수에 n년 더하기(2월 29일은 28일로)"""
    d = d.astype("Int64")
    y = d // 10000; md = d % 10000
    md = md.where(md != 229, 228)
    return (y + n) * 10000 + md


def _to_dt(x):
    return pd.to_datetime(x.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def f_C04(B, t0, ctx):
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
    # 원점 시점 미납 초과(납부 기한 경과, 소멸등록 전): 살아있는 등록특허 중 납부완료연차 기한 < end
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


def f_C05(B, t0, ctx):
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


def _rl_counts(B, t0, ctx, typ):
    end = t0 * 10000 + 1231
    rl = ctx["rl"]; rl = rl[(rl.typ.isin(typ)) & (rl.d <= end)]
    a = B["applicantCd"]; okm = a.isin(ctx["rl_ok"])
    tot = a.map(rl.groupby("applicant_cd").size()).fillna(0).astype(float)
    r2 = a.map(rl[rl.d > end - 20000].groupby("applicant_cd").size()).fillna(0).astype(float)
    p2 = a.map(rl[(rl.d > end - 40000) & (rl.d <= end - 20000)].groupby("applicant_cd").size()).fillna(0).astype(float)
    return tot.where(okm), r2.where(okm), p2.where(okm), rl, okm


def f_C06(B, t0, ctx):
    tot, r2, p2, rl, okm = _rl_counts(B, t0, ctx, ["pat"])
    a = B["applicantCd"]
    oos = a.map(rl[~rl.in_prov].groupby("applicant_cd").size()).fillna(0).astype(float).where(okm)
    o = pd.DataFrame(index=B.index)
    o["rl_log_pat"] = np.log1p(tot); o["rl_oos_share"] = oos / tot.replace(0, np.nan)
    o["rl_pat_2y"] = np.log1p(r2); o["rl_pat_trend"] = np.log((r2 + 1) / (p2 + 1))
    return o


def f_C07(B, t0, ctx):
    td, rd, pd_, _, _ = _rl_counts(B, t0, ctx, ["des"])
    tu, ru, pu, _, _ = _rl_counts(B, t0, ctx, ["um"])
    o = pd.DataFrame(index=B.index)
    o["rl_log_des"] = np.log1p(td); o["rl_des_2y"] = np.log1p(rd)
    o["rl_log_um"] = np.log1p(tu); o["rl_um_2y"] = np.log1p(ru)
    o["rl_du_trend"] = np.log((rd + ru + 1) / (pd_ + pu + 1))
    return o


def f_C08(B, t0, ctx):
    t, r, p, _, _ = _rl_counts(B, t0, ctx, ["tm"])
    o = pd.DataFrame(index=B.index)
    o["rl_log_tm"] = np.log1p(t); o["rl_tm_2y"] = np.log1p(r); o["rl_tm_trend"] = np.log((r + 1) / (p + 1))
    return o


def f_C09(B, t0, ctx):
    end = t0 * 10000 + 1231
    dd = ctx["dart"]; dd = dd[(dd.avail <= end) & (dd.fy >= t0 - 2) & dd.bizr_no.isin(set(B.bizno))]
    dd = dd.sort_values(["bizr_no", "fy", "ofs", "avail"]).drop_duplicates("bizr_no", keep="last").set_index("bizr_no")
    o = pd.DataFrame(index=B.index); bz = B["bizno"]
    m = lambda s: bz.map(s).astype(float)
    o["dart_has"] = bz.isin(dd.index).astype(float)
    o["dart_lev"] = m(dd.lev); o["dart_opm"] = m(dd.opm)
    o["dart_roa"] = m(dd.ni / dd.assets.replace(0, np.nan))
    o["dart_neg_eq"] = m((dd.equity <= 0).astype(float))
    o["dart_impair"] = m((dd.equity < dd.capital).astype(float))
    o["dart_g_rev"] = m(dd.g_rev); o["dart_log_assets"] = m(np.log1p(dd.assets.clip(lower=0) / 1e6))
    o["dart_curr"] = m(dd.cur_assets / dd.cur_liab.replace(0, np.nan))
    return o


def f_C10(B, t0, ctx, AF, ns):
    """특허 KSIC 라벨로 본 동종 특허기업의 직전 2년 폐업률(원점 t0에 이미 관측된 폐업: (t0-2, t0])."""
    s = t0 - 2; send = s * 10000 + 1231; end = t0 * 10000 + 1231
    pt = ctx["R"]["pt"]
    reg = set(pt[pt.rgstDate <= send].applicantCd)
    kn = AF[(AF.known_date <= send) & AF.applicantCd.isin(reg)]
    e = ns.set_index("applicantCd").end
    kn = kn[kn.applicantCd.isin(e.index)]
    kn = kn[~(kn.applicantCd.map(e) <= send)]
    W = kn.groupby(["applicantCd", "ksic"]).appl_no.nunique().unstack(fill_value=0)
    W = W.div(W.sum(1), axis=0)
    ind = pd.Series(W.index.map(e), index=W.index).astype(float)
    yv = ((ind > send) & (ind <= end)).astype(float)
    rate = (W.mul(yv, axis=0).sum() + 0.5 * yv.mean() * 20) / (W.sum() + 20)   # 평활(강도 20)
    kc = AF[(AF.known_date <= end) & AF.applicantCd.isin(set(B.applicantCd))]
    Wc = kc.groupby(["applicantCd", "ksic"]).appl_no.nunique().unstack(fill_value=0)
    Wc = Wc.div(Wc.sum(1), axis=0).reindex(columns=rate.index, fill_value=0)
    x = pd.Series(Wc.values @ rate.values, index=Wc.index)
    o = pd.DataFrame(index=B.index)
    o["x_close_ksic"] = B["applicantCd"].map(x).astype(float)
    o["x_close_ksic_rel"] = o.x_close_ksic / yv.mean()
    return o


def f_C11(B, t0, ctx):
    """권리 처분 행태: 원점 직전 2년 보유 특허의 양도(권리자 변경)·합병/분할·포기(말소) — 모두 사건일 ≤ t0"""
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


def f_C13(B, t0, ctx):
    """NPS 흐름: 1인당 고지액(임금 대리) 1년 변화, 상실률·취득률 1년 변화(t0-12 vs t0-1-12 스냅샷)"""
    NPS = ctx["NPS"]; keys = B["key"]
    o = pd.DataFrame(index=B.index, columns=BUNDLES["C13_nps_flow"], dtype=float)
    prv = NPS.get(f"{t0 - 1}-12")
    if prv is None:
        return o
    def v(d, col):
        d = d[d.active].set_index("key")
        e = d.n_emp.replace(0, np.nan)
        s = (d[col] / e) if col != "n_emp" else d.n_emp
        return keys.map(lambda k: s.get(k, np.nan) if isinstance(k, str) else np.nan).astype(float)
    cur = NPS[f"{t0}-12"]
    o["g_wage1"] = np.log((v(cur, "amt") + 1) / (v(prv, "amt") + 1))
    o["d_loss_rate"] = v(cur, "n_loss") - v(prv, "n_loss")
    o["d_hire_rate"] = v(cur, "n_new") - v(prv, "n_new")
    return o


def f_C14(B, t0, ctx, AF):
    end = t0 * 10000 + 1231
    ev = ctx["fev"]; ev = ev[(ev.known <= end) & ev.applicantCd.isin(set(B.applicantCd))]
    a = B["applicantCd"]
    o = pd.DataFrame(index=B.index)
    o["fam_n_3y"] = np.log1p(a.map(ev[ev.known > end - 30000].drop_duplicates(["applicantCd", "fkey"]).groupby("applicantCd").size()).fillna(0).astype(float))
    o["fam_share"] = (a.map(ev.groupby("applicantCd").kr_appl.nunique()).fillna(0) / B["n_app"]).astype(float)
    o["fam_prio_2y"] = np.log1p(a.map(ev[ev.prio > end - 20000].drop_duplicates(["applicantCd", "fkey"]).groupby("applicantCd").size()).fillna(0).astype(float))
    return o


RL_COLS = BUNDLES["C06_rl_patent_oos"] + BUNDLES["C07_design_utility"] + BUNDLES["C08_trademark"]


def hotdeck_rl(B, X, t0):
    """권리목록 조회 실패(RC:001) 출원인은 폐업 기업에 몰려 있다(2026 스냅샷의 사후 선택 → 결측 자체가 누수).
    결측 행에는 같은 원점·같은 셀(출원 규모 5분위 × NPS 존재)의 조회 성공 행 벡터를 무작위로 통째 복사한다(시드 = t0).
    이렇게 하면 결측 여부가 결과와 무관해진다(조건: 원점 시점 정보만)."""
    rng = np.random.default_rng(int(t0))
    miss = X["rl_log_pat"].isna().values
    if miss.sum() == 0:
        return X
    cell = pd.qcut(B["log_app"].rank(method="first"), 5, labels=False).astype(str).values + "_" + B["nps_present"].astype(int).astype(str).values
    Xr = X[RL_COLS].copy()
    V = Xr.values
    for cval in np.unique(cell[miss]):
        tgt = np.where(miss & (cell == cval))[0]
        don = np.where(~miss & (cell == cval))[0]
        if len(don) == 0:
            don = np.where(~miss)[0]
        V[tgt] = V[rng.choice(don, len(tgt))]
    X = X.copy()
    X[RL_COLS] = V
    X["rl_imputed"] = miss.astype(float)   # 기록용(모형 피처 아님)
    return X


def build_all(B, t0, ctx, AF, ns):
    parts = [f_C01(B, t0, ctx), f_C02(B, t0, ctx), f_C03(B, t0, ctx, AF), f_C04(B, t0, ctx), f_C05(B, t0, ctx),
             f_C06(B, t0, ctx), f_C07(B, t0, ctx), f_C08(B, t0, ctx), f_C09(B, t0, ctx), f_C10(B, t0, ctx, AF, ns), f_C11(B, t0, ctx), f_C13(B, t0, ctx), f_C14(B, t0, ctx, AF)]
    X = pd.concat(parts, axis=1)
    return hotdeck_rl(B, X, t0)
