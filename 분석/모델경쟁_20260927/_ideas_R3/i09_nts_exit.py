# -*- coding: utf-8 -*-
"""I2' 특허기업 폐업 조기경보 — 결과를 국세청 폐업일(r3nts_applicant_status.end_dt)로 바꾼 재검정(3인 미만·NPS 미가입 기업 포함).
단위 = 국내법인 출원인코드(1…), t0까지 제공범위 KR 등록특허 1건 이상(국세청 조회 대상 tier1과 정합), 국세청 상태 연결, t0 말까지 폐업 아님.
결과 = 폐업일 ∈ (t0, t0+2]. 🔒 최종시험 t0=2023(폐업 2024–25, F=2024) → 개발 t0 ≤ 2021(≤2023)만.
M0 = 외부(NPS 속성+NPS 미등록 여부+사업자번호 유형) 기준선, M1 = +특허 행태(등록료 불납·납부·출원 추세), M2 = +KSIC 노출(특허 모멘텀), M3 = +외부 산업(논문·물가)."""
import os, sqlite3, time
import pandas as pd, numpy as np
from i00_common import *

t = time.time()
def log(*a):
    print(*a, round(time.time() - t), flush=True)

SEAL = 2024
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "b_stt_cd", "end_dt", "registered"])
ns = ns[ns.registered.fillna(False).astype(bool)].drop_duplicates("applicantCd")
ns["end"] = pd.to_numeric(ns.end_dt.dt.strftime("%Y%m%d"), errors="coerce")
ns["corp_type"] = pd.to_numeric(ns.bizno.str[3:5], errors="coerce")
AF = pd.read_parquet(os.path.join(CACHE, "app_firm.parquet"))
AF = AF[AF.applicantCd.astype(str).str[0] == "1"]
AF = AF[AF.known_date.notna()]
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, applicantCd from applicant", c).drop_duplicates()
pt = pd.read_sql("select rgst_no, rgstDate from patent", c); pt["rgstDate"] = pd.to_numeric(pt.rgstDate, errors="coerce")
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right", c); rt["d"] = pd.to_numeric(rt.rgstCsDate, errors="coerce")
ext_first = rt[rt.rgstCsName == "소멸등록"].groupby("rgst_no").d.min()
lapnp = rt[(rt.rgstCsName == "소멸등록") & rt.rgstCsReason.isin(["등록료불납", "연차료불납"])][["rgst_no", "d"]].merge(ap, on="rgst_no")
trf = rt[rt.rgstCsName.fillna("").str.contains("이전")][["rgst_no", "d"]].merge(ap, on="rgst_no")
py = pd.read_sql("select rgst_no, payDate from pay", c); py["d"] = pd.to_numeric(py.payDate, errors="coerce"); py = py.merge(ap, on="rgst_no")
pt = pt.merge(ap, on="rgst_no"); pt["ext_d"] = pt.rgst_no.map(ext_first)
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]; lk["key"] = lk.biz6 + "|" + lk.nname
a2k = dict(zip(lk.applicantCd, lk.key))
# KSIC 신호는 i03과 같은 정의
import importlib.util
spec = importlib.util.spec_from_file_location("sig", os.path.join(OUT, "i03_feats.py"))
log("load")


def ksic_signals_lite():
    kp = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "세세분류코드"])
    kp.columns = ["appl_no", "ay", "ksic"]; kp["ksic"] = kp.ksic.astype(str)
    KR = kp.drop_duplicates(["appl_no", "ksic"]).groupby(["ksic", "ay"]).size().unstack(fill_value=0)
    FC = pd.read_parquet(os.path.join(EXT, "science_patent_counts_ksic_cc_year.parquet"))
    FR = FC[FC.cc != "KR"].groupby(["ksic", "year"]).n_pat.sum().unstack(fill_value=0)
    OA = pd.read_parquet(os.path.join(EXT, "science_openalex_ksic_extra_year.parquet"))
    OA = OA[OA.variant == "company_ALL"].pivot(index="ksic", columns="year", values="n_works").fillna(0)
    PP = pd.read_parquet(os.path.join(EXT, "demand_ksic_price_panel_1996_2021.parquet"))
    PP = PP[PP.series == "ppi_basic"].pivot(index="ksic", columns="year", values="dl")
    return KR, FR, OA, PP


KR, FR, OA, PP = ksic_signals_lite(); KS = sorted(KR.index)
ws = lambda M, ys: M[[y for y in ys if y in M.columns]].sum(axis=1)


def sig(t0):
    s = pd.DataFrame(index=KS)
    s["s_kr"] = np.log((ws(KR, [t0 - 3, t0 - 2]) + 1) / (ws(KR, [t0 - 5, t0 - 4]) + 1)).reindex(KS)
    s["s_fr"] = np.log((ws(FR, [t0 - 3, t0 - 2]) + 1) / (ws(FR, [t0 - 5, t0 - 4]) + 1)).reindex(KS)
    s["s_sci"] = np.log((ws(OA, [t0 - 3, t0 - 2, t0 - 1]) + 1) / (ws(OA, [t0 - 6, t0 - 5, t0 - 4]) + 1)).reindex(KS)
    cols = [y for y in [t0 - 3, t0 - 2, t0 - 1] if y in PP.columns]
    s["s_ppi"] = PP[cols].sum(axis=1, min_count=len(cols)).reindex(KS)
    return s


NPS = {s: pd.read_parquet(os.path.join(CACHE, f"nps_firm_{s}.parquet")) for s in SNAPS_OK if s <= "2021-12"}
rows = []
for t0 in range(2015, 2022):
    assert t0 + 2 < SEAL
    end = t0 * 10000 + 1231; y1 = end - 10000; y2 = end - 20000
    kn = AF[AF.known_date <= end]
    g = kn.groupby("applicantCd")
    f = pd.DataFrame({"n_app": g.appl_no.nunique(), "n_ksic": g.ksic.nunique(), "last_known": g.known_date.max()})
    f = f.join(ns.set_index("applicantCd")[["end", "corp_type"]], how="inner")
    f = f[~(f.end <= end)]
    f["log_app"] = np.log1p(f.n_app); f["yrs_since_last"] = t0 - f.last_known // 10000
    a_new = kn[kn.ay.between(t0 - 3, t0 - 2)].groupby("applicantCd").appl_no.nunique()
    a_old = kn[kn.ay.between(t0 - 5, t0 - 4)].groupby("applicantCd").appl_no.nunique()
    f["app_trend"] = np.log((a_new.reindex(f.index).fillna(0) + 1) / (a_old.reindex(f.index).fillna(0) + 1))
    f["known_1y"] = kn[kn.known_date > y1].groupby("applicantCd").appl_no.nunique().reindex(f.index).fillna(0)
    p = pt[pt.rgstDate <= end]
    # 국세청 조회 대상(tier1)이 '제공범위 등록특허 보유 출원인'이므로, t0까지 등록 1건 이상인 기업만 남겨 선택을 시점 정합으로 만든다
    f = f[f.index.isin(set(p.applicantCd))].copy()
    f["n_live"] = p[~(p.ext_d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    npv = p[(p.rgstDate <= y1) & ~(p.ext_d <= y1)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    f["lapse_1y"] = lapnp[(lapnp.d > y1) & (lapnp.d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    f["lapse_2y"] = lapnp[(lapnp.d > y2) & (lapnp.d <= y1)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    P1 = py[(py.d > y1) & (py.d <= end)].groupby("applicantCd").size().reindex(f.index).fillna(0)
    P2 = py[(py.d > y2) & (py.d <= y1)].groupby("applicantCd").size().reindex(f.index).fillna(0)
    f["pay_1y"] = P1 / (npv + 1); f["pay_trend"] = np.log((P1 + 1) / (P2 + 1))
    f["trf_2y"] = trf[(trf.d > y2) & (trf.d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    f["log_live"] = np.log1p(f.n_live); f["live_ratio"] = f.n_live / (f.n_app + 1)
    # NPS
    cur = NPS[f"{t0}-12"]; cur = cur[cur.active].set_index("key")
    prv = NPS.get(f"{t0 - 1}-12"); prv = prv[prv.active].set_index("key") if prv is not None else None
    keys = f.index.map(lambda a: a2k.get(a))
    emp = pd.Series([cur.n_emp.get(k, np.nan) if k else np.nan for k in keys], index=f.index)
    f["nps_present"] = emp.notna().astype(float)
    f["log_emp"] = np.log1p(emp)
    if prv is not None:
        e0 = pd.Series([prv.n_emp.get(k, np.nan) if k else np.nan for k in keys], index=f.index)
        f["g_emp1"] = np.log((emp + 1) / (e0 + 1))
    else:
        f["g_emp1"] = np.nan
    ad = pd.Series([cur.adpt.get(k, None) if k else None for k in keys], index=f.index)
    f["age"] = t0 - pd.to_numeric(ad.str[:4], errors="coerce")
    for col, src in [("hire_rate", "n_new"), ("loss_rate", "n_loss")]:
        v = pd.Series([cur[src].get(k, np.nan) if k else np.nan for k in keys], index=f.index)
        f[col] = v / emp
    am = pd.Series([cur.amt.get(k, np.nan) if k else np.nan for k in keys], index=f.index)
    f["wage"] = np.log1p(am / emp)
    W = kn[kn.applicantCd.isin(f.index)].groupby(["applicantCd", "ksic"]).appl_no.nunique().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
    W = W.div(W.sum(1), axis=0); S = sig(t0)
    for cn in S.columns:
        v = S[cn]; m = v.notna().values; ww = W.values[:, m]; den = ww.sum(1)
        f.loc[W.index, "x_" + cn] = np.where(den > 0, ww @ v.values[m] / np.where(den > 0, den, 1), np.nan)
    f["y"] = ((f.end > end) & (f.end <= (t0 + 2) * 10000 + 1231)).astype(int)
    f["t0"] = t0
    rows.append(f.reset_index().rename(columns={"index": "applicantCd"}))
    log(t0, len(f), int(f.y.sum()), round(f.y.mean(), 4), "NPS 연결", round(f.nps_present.mean(), 3))
D = pd.concat(rows, ignore_index=True)
D.to_parquet(os.path.join(CACHE, "I2b_panel.parquet"), index=False)
