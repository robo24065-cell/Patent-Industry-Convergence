# -*- coding: utf-8 -*-
"""I1(잠재 첫 출원 기업)·I2(특허기업 소멸)·I3(특허기업 고용 고성장) 원점별 학습표 생성 — 개발 원점만.
I1: t0 ∈ 2015..2019, 결과 = 50 KSIC 제공범위 KR 출원(출원일) t0+1..t0+2  (최대 2021 < 봉인 2022)
I2·I3: t0 ∈ 2015..2021, 결과 = NPS t0+2 스냅샷 (최대 2023-12 < 봉인 2024-12)
피처는 모두 t0년 12월 31일(또는 t0 스냅샷)까지 알 수 있던 값만 쓴다."""
import os, sqlite3, time, sys
import pandas as pd, numpy as np
from i00_common import *

t = time.time()
def log(*a):
    print(*a, round(time.time() - t), flush=True)

NPS = {s: pd.read_parquet(os.path.join(CACHE, f"nps_firm_{s}.parquet")) for s in SNAPS_OK}
AF = pd.read_parquet(os.path.join(CACHE, "app_firm.parquet"))
AFk = AF.dropna(subset=["key"]).copy()
AFk = AFk[AFk.known_date.notna()]
log("load")

# ---------------- KSIC 수준 신호 (t0 시점 계산) ----------------
kp = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "세세분류코드"])
kp.columns = ["appl_no", "ay", "ksic"]; kp["ksic"] = kp.ksic.astype(str)
KR = kp.drop_duplicates(["appl_no", "ksic"]).groupby(["ksic", "ay"]).size().unstack(fill_value=0)
FC = pd.read_parquet(os.path.join(EXT, "science_patent_counts_ksic_cc_year.parquet"))
FR = FC[FC.cc != "KR"].groupby(["ksic", "year"]).n_pat.sum().unstack(fill_value=0)
OA = pd.read_parquet(os.path.join(EXT, "science_openalex_ksic_extra_year.parquet"))
OA = OA[OA.variant == "company_ALL"].pivot(index="ksic", columns="year", values="n_works").fillna(0)
PP = pd.read_parquet(os.path.join(EXT, "demand_ksic_price_panel_1996_2021.parquet"))
PP = PP[PP.series == "ppi_basic"].pivot(index="ksic", columns="year", values="dl")
KS = sorted(KR.index)


def wsum(M, ys):
    cols = [y for y in ys if y in M.columns]
    return M[cols].sum(axis=1) if cols else pd.Series(0.0, index=M.index)


def ksic_signals(t0):
    s = pd.DataFrame(index=KS)
    s["s_kr"] = np.log((wsum(KR, [t0 - 3, t0 - 2]) + 1) / (wsum(KR, [t0 - 5, t0 - 4]) + 1)).reindex(KS)
    s["s_fr"] = np.log((wsum(FR, [t0 - 3, t0 - 2]) + 1) / (wsum(FR, [t0 - 5, t0 - 4]) + 1)).reindex(KS)
    s["s_sci"] = np.log((wsum(OA, [t0 - 3, t0 - 2, t0 - 1]) + 1) / (wsum(OA, [t0 - 6, t0 - 5, t0 - 4]) + 1)).reindex(KS)
    cols = [y for y in [t0 - 3, t0 - 2, t0 - 1] if y in PP.columns]   # 연평균 물가는 이듬해 1월 공표 → t0-1까지
    s["s_ppi"] = PP[cols].sum(axis=1, min_count=len(cols)).reindex(KS) if cols else np.nan
    return s


def expose(W, sig):
    """W: (단위 × ksic) 비중행렬, sig: ksic×신호 → 단위×신호 가중평균(결측 신호는 재정규화)"""
    out = {}
    for c in sig.columns:
        v = sig[c].reindex(W.columns)
        m = v.notna().values
        ww = W.values[:, m]
        den = ww.sum(1)
        out["x_" + c] = np.where(den > 0, ww @ v.values[m] / np.where(den > 0, den, 1), np.nan)
    return pd.DataFrame(out, index=W.index)


def nps_feats(t0):
    cur = NPS[f"{t0}-12"]
    cur = cur[cur.active & (cur.n_emp > 0)].copy()
    prev = NPS.get(f"{t0 - 1}-12")
    if prev is not None:
        cur = cur.merge(prev[["key", "n_emp"]].rename(columns={"n_emp": "emp_prev"}), on="key", how="left")
        cur["g_emp1"] = np.log((cur.n_emp + 1) / (cur.emp_prev + 1))
    else:
        cur["g_emp1"] = np.nan
    cur["log_emp"] = np.log1p(cur.n_emp)
    cur["age"] = t0 - pd.to_numeric(cur.adpt.str[:4], errors="coerce")
    cur["hire_rate"] = cur.n_new / cur.n_emp
    cur["loss_rate"] = cur.n_loss / cur.n_emp
    cur["wage"] = np.log1p(cur.amt / cur.n_emp)
    cur["div2"] = pd.to_numeric(cur.indcd.fillna("").str[:2], errors="coerce")
    cur["sido"] = pd.to_numeric(cur.sgg.str[:2], errors="coerce")
    return cur


F_NPS = ["log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "n_wkpl", "div2", "sido"]

# ================= I1 =================
def build_I1(t0):
    end = t0 * 10000 + 1231
    cur = nps_feats(t0)
    cur = cur[cur.mfg].copy()
    known = AFk[AFk.known_date <= end]
    filers = set(known.key)
    cur["filer"] = cur.key.isin(filers)
    first_known = known.groupby("key").known_date.min()
    cur["first_known"] = cur.key.map(first_known)
    cur["recent_first"] = (cur.first_known > (t0 - 2) * 10000 + 1231) & cur.filer
    # 업종·지역 침투율(평활)
    g0 = cur.filer.mean(); r0 = cur.recent_first.mean()
    for name, grp in [("c6", cur.indcd.fillna("NA")), ("c4", cur.indcd.fillna("NA").str[:4]), ("sgg", cur.sgg), ("sggd", cur.sgg + "_" + cur.div2.astype(str))]:
        a = cur.groupby(grp).agg(n=("filer", "size"), f=("filer", "sum"), r=("recent_first", "sum"))
        pen = (a.f + 20 * g0) / (a.n + 20); ent = (a.r + 20 * r0) / (a.n + 20)
        cur["pen_" + name] = grp.map(pen).values
        cur["ent_" + name] = grp.map(ent).values
    # 업종코드 → KSIC 노출(해당 업종 기존 출원기업들의 KSIC 분포, 전역분포로 수축)
    kk = known.merge(cur[["key", "indcd"]], on="key")
    Wc = kk.groupby(["indcd", "ksic"]).appl_no.nunique().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
    glob = Wc.sum(0) / Wc.values.sum()
    Wc = (Wc.add(20 * glob, axis=1)).div(Wc.sum(1) + 20, axis=0)
    allc = cur.indcd.fillna("NA").unique()
    Wc = Wc.reindex(allc)
    Wc = Wc.fillna(pd.Series(glob.values, index=KS))
    X = expose(Wc, ksic_signals(t0))
    cur = cur.merge(X, left_on=cur.indcd.fillna("NA"), right_index=True, how="left").drop(columns=["key_0"], errors="ignore")
    risk = cur[~cur.filer].copy()
    # 결과: 출원일 연도 t0+1..t0+2
    assert t0 + 2 < SEAL_I1_OUTCOME_YEAR
    pos = set(AF.loc[AF.ay.between(t0 + 1, t0 + 2) & AF.key.notna(), "key"])
    risk["y"] = risk.key.isin(pos).astype(int)
    risk["t0"] = t0
    return risk


# ================= I2·I3 =================
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
ap = pd.read_sql("select rgst_no, applicantCd from applicant", c)
pt = pd.read_sql("select rgst_no, rgstDate, ksic from patent", c)
pt["rgstDate"] = pd.to_numeric(pt.rgstDate, errors="coerce")
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right", c)
rt["d"] = pd.to_numeric(rt.rgstCsDate, errors="coerce")
lap = rt[(rt.rgstCsName == "소멸등록")][["rgst_no", "d", "rgstCsReason"]]
lapnp = lap[lap.rgstCsReason.isin(["등록료불납", "연차료불납"])]
ext_first = lap.groupby("rgst_no").d.min()
trf = rt[rt.rgstCsName.fillna("").str.contains("이전")][["rgst_no", "d"]]
py = pd.read_sql("select rgst_no, payDate from pay", c)
py["d"] = pd.to_numeric(py.payDate, errors="coerce")
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
ap = ap.merge(lk[["applicantCd", "key"]], on="applicantCd").drop_duplicates(["rgst_no", "key"])
pt = pt.merge(ap[["rgst_no", "key"]], on="rgst_no")
pt["ext_d"] = pt.rgst_no.map(ext_first)
log("registry", len(pt))


def alive_keys(s):
    """s 스냅샷에서 살아 있는 기업키 + (biz6, 적용일) — 상호변경 흡수용"""
    d = NPS[s]; d = d[d.active]
    return set(d.key), set(zip(d.biz6, d.adpt))


def build_I23(t0):
    end = t0 * 10000 + 1231; y1 = (t0 - 1) * 10000 + 1231; y2 = (t0 - 2) * 10000 + 1231
    cur = nps_feats(t0)
    known = AFk[AFk.known_date <= end]
    cur = cur[cur.key.isin(set(known.key))].copy()
    # 제공범위 출원 행태
    g = known.groupby("key")
    f = pd.DataFrame({"n_app": g.appl_no.nunique(), "n_ksic": g.ksic.nunique(), "last_known": g.known_date.max()})
    f["log_app"] = np.log1p(f.n_app)
    f["yrs_since_last"] = t0 - f.last_known // 10000
    a_new = known[known.ay.between(t0 - 3, t0 - 2)].groupby("key").appl_no.nunique()
    a_old = known[known.ay.between(t0 - 5, t0 - 4)].groupby("key").appl_no.nunique()
    f["app_trend"] = np.log((a_new.reindex(f.index).fillna(0) + 1) / (a_old.reindex(f.index).fillna(0) + 1))
    f["known_1y"] = known[known.known_date > y1].groupby("key").appl_no.nunique().reindex(f.index).fillna(0)
    # 등록원부 행태 (t0 시점)
    p = pt[pt.rgstDate <= end]
    live = p[~(p.ext_d <= end)]
    live_prev = p[(p.rgstDate <= y1) & ~(p.ext_d <= y1)]
    f["n_live"] = live.groupby("key").rgst_no.nunique().reindex(f.index).fillna(0)
    f["n_live_prev"] = live_prev.groupby("key").rgst_no.nunique().reindex(f.index).fillna(0)
    L1 = lapnp[(lapnp.d > y1) & (lapnp.d <= end)].merge(ap[["rgst_no", "key"]], on="rgst_no").groupby("key").rgst_no.nunique()
    L2 = lapnp[(lapnp.d > y2) & (lapnp.d <= y1)].merge(ap[["rgst_no", "key"]], on="rgst_no").groupby("key").rgst_no.nunique()
    f["lapse_1y"] = L1.reindex(f.index).fillna(0) / (f.n_live_prev + 1)
    f["lapse_2y"] = L2.reindex(f.index).fillna(0) / (f.n_live_prev + 1)
    P1 = py[(py.d > y1) & (py.d <= end)].merge(ap[["rgst_no", "key"]], on="rgst_no").groupby("key").size()
    P2 = py[(py.d > y2) & (py.d <= y1)].merge(ap[["rgst_no", "key"]], on="rgst_no").groupby("key").size()
    f["pay_1y"] = P1.reindex(f.index).fillna(0) / (f.n_live_prev + 1)
    f["pay_trend"] = np.log((P1.reindex(f.index).fillna(0) + 1) / (P2.reindex(f.index).fillna(0) + 1))
    T1 = trf[(trf.d > y2) & (trf.d <= end)].merge(ap[["rgst_no", "key"]], on="rgst_no").groupby("key").rgst_no.nunique()
    f["trf_2y"] = T1.reindex(f.index).fillna(0) / (f.n_live_prev + 1)
    f["live_ratio"] = f.n_live / (f.n_app + 1)
    f["log_live"] = np.log1p(f.n_live)
    cur = cur.merge(f.drop(columns=["last_known", "n_app"]), left_on="key", right_index=True, how="left")
    # KSIC 노출(기업 자신의 출원 KSIC 분포)
    W = known.groupby(["key", "ksic"]).appl_no.nunique().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
    W = W.div(W.sum(1), axis=0)
    X = expose(W, ksic_signals(t0))
    cur = cur.merge(X, left_on="key", right_index=True, how="left")
    # 결과
    s_out = f"{t0 + 2}-12"
    assert s_out < SEAL_I23_SNAP
    ak, ab = alive_keys(s_out)
    alive = cur.key.isin(ak) | pd.Series(list(zip(cur.biz6, cur.adpt)), index=cur.index).isin(ab)
    cur["y_exit"] = (~alive).astype(int)
    o = NPS[s_out]; o = o[o.active].groupby("key").n_emp.sum()
    emp2 = cur.key.map(o).fillna(0)
    cur["y_hgf"] = ((cur.n_emp >= 10) & (emp2 >= 1.44 * cur.n_emp)).astype(int)
    cur.loc[cur.n_emp < 10, "y_hgf"] = -1   # HGF 대상 아님
    cur["t0"] = t0
    return cur


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "I1"):
        D = pd.concat([build_I1(t0) for t0 in range(2015, 2020)], ignore_index=True)
        D.to_parquet(os.path.join(CACHE, "I1_panel.parquet"), index=False)
        log("I1", D.groupby("t0").agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean")).to_string())
    if which in ("all", "I23"):
        D = pd.concat([build_I23(t0) for t0 in range(2015, 2022)], ignore_index=True)
        D.to_parquet(os.path.join(CACHE, "I23_panel.parquet"), index=False)
        log("I23", D.groupby("t0").agg(n=("y_exit", "size"), exit=("y_exit", "mean"), hgf_n=("y_hgf", lambda s: (s >= 0).sum()),
                                        hgf=("y_hgf", lambda s: s[s >= 0].mean())).to_string())
