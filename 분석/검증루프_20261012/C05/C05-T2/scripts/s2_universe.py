# -*- coding: utf-8 -*-
"""s2: 후보 전체 U(t0) 구성 + 기준일 관측 변수 X + 미래 진단 변수(M, 마지막 활동연도) + 층 표지.
트랙: N4(t0=2023), N7(t0=2022). 출력 cache/U_{track}.parquet. 레이블·예측은 붙이지 않는다(s3에서)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *

AF = pd.read_parquet(os.path.join(N4, "cache", "app_firm.parquet"))
AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()].copy()
PT = pd.read_parquet(os.path.join(N4, "cache", "reg_pt.parquet"))
LAP = pd.read_parquet(os.path.join(N4, "cache", "reg_lapnp.parquet"))
ADDR = pd.read_parquet(os.path.join(CACHE, "reg_addr.parquet"))
NS = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "src", "registered", "end_dt"])
NSR = NS[NS.registered.fillna(False).astype(bool)].drop_duplicates("applicantCd").set_index("applicantCd")
HASB = set(NS.applicantCd)  # 사업자번호(10자리) 있는 출원인코드(등록 여부 무관)
# 출처: CA 또는 권리목록만
src = NS.groupby("applicantCd").src.agg(lambda s: "CA" if (s == "CA").any() else "RL")
# as-of 확인 자료
nt = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["year", "corp_rgst_no"])
nt["bz"] = nt.corp_rgst_no.fillna("").str.replace(r"\D", "", regex=True)
nt = nt[nt.bz.str.len() == 10]
ntis_first = nt[nt.year >= 2010].groupby("bz").year.min()
dart = pd.read_parquet(os.path.join(EXT, "firm_dart_fin_panel.parquet"), columns=["bizr_no", "avail_date"])
dart["bz"] = dart.bizr_no.fillna("").astype(str).str.replace(r"\D", "", regex=True)
dart["av"] = pd.to_numeric(pd.to_datetime(dart.avail_date, errors="coerce").dt.strftime("%Y%m%d"), errors="coerce")
dart_first = dart.dropna(subset=["av"]).groupby("bz").av.min()
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
lk = lk[lk.nps_hit]; A2K = dict(zip(lk.applicantCd, lk.biz6 + "|" + lk.nname))


def nps_active_keys(t0):
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{t0}12.parquet"), columns=["name", "biz6", "stat"])
    d = d[d.stat == "1"]
    return set(d.biz6.fillna("") + "|" + d.name.map(norm))


def build(t0):
    end = t0 * 10000 + 1231; y1 = end - 10000; y2 = end - 20000
    m_end = (t0 + 2) * 10000 + 831
    kn = AF[AF.known_date <= end]
    p = PT[PT.rgstDate <= end]
    firms = sorted(set(kn.applicantCd) & set(p.applicantCd))
    kn = kn[kn.applicantCd.isin(firms)]; p = p[p.applicantCd.isin(firms)]
    g = kn.groupby("applicantCd")
    f = pd.DataFrame({"n_app": g.appl_no.nunique(), "last_known": g.known_date.max(), "first_ay": g.ay.min().astype(float)})
    f["ksic2"] = kn.assign(k2=kn.ksic.astype(str).str[:2]).groupby("applicantCd").k2.agg(lambda s: s.value_counts().index[0])
    f["log_app"] = np.log1p(f.n_app)
    f["yrs_since_last"] = t0 - f.last_known // 10000
    f["n_rgst"] = p.groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    f["n_live"] = p[~(p.ext_d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    f["has_live"] = (f.n_live > 0).astype(float); f["log_live"] = np.log1p(f.n_live); f["live_ratio"] = f.n_live / (f.n_app + 1)
    npv = p[(p.rgstDate <= y2) & ~(p.ext_d <= y2)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    f["lapse_2y"] = LAP[(LAP.d > y2) & (LAP.d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    a = ADDR[(ADDR.rgstDate <= end) & ADDR.applicantCd.isin(f.index)].sort_values("rgstDate")
    f["sido"] = a.groupby("applicantCd").sido.last().reindex(f.index).fillna("미상")
    # 미래(진단 전용)
    fut = AF[(AF.known_date > end) & AF.applicantCd.isin(f.index)]
    f["M"] = f.index.isin(set(fut[fut.known_date <= m_end].applicantCd)).astype(int)
    f["M_all"] = f.index.isin(set(fut.applicantCd)).astype(int)
    f["M_appl"] = f.index.isin(set(AF[(AF.appl_date > end) & AF.applicantCd.isin(f.index)].applicantCd)).astype(int)
    lastall = AF[AF.applicantCd.isin(f.index)].groupby("applicantCd").known_date.max()
    f["last_year"] = (lastall.reindex(f.index) // 10000).astype(float)
    # 연결
    f["has_bizno"] = f.index.isin(HASB).astype(int)
    f["L"] = f.index.isin(NSR.index).astype(int)
    f["bizno"] = NSR.bizno.reindex(f.index)
    f["src"] = src.reindex(f.index)
    f["closed_le_t0"] = (pd.to_numeric(NSR.end_dt.reindex(f.index).dt.strftime("%Y%m%d"), errors="coerce") <= end).astype(int)
    # as-of 확인(S2)
    act = nps_active_keys(t0)
    keys = pd.Series([A2K.get(x) for x in f.index], index=f.index)
    f["asof_nps"] = keys.map(lambda k: k in act if isinstance(k, str) else False).astype(int)
    f["asof_ntis"] = (f.bizno.map(ntis_first) <= t0).fillna(False).astype(int)
    f["asof_dart"] = (f.bizno.map(dart_first) <= end).fillna(False).astype(int)
    f["S2"] = ((f.asof_nps + f.asof_ntis + f.asof_dart) > 0).astype(int)
    f["t0"] = t0
    return f.reset_index().rename(columns={"index": "applicantCd"})


for name, t0 in [("N4", 2023), ("N7", 2022)]:
    U = build(t0)
    U.to_parquet(os.path.join(CACHE, f"U_{name}.parquet"), index=False)
    log(name, t0, "U", len(U), "L", int(U.L.sum()), "has_bizno", int(U.has_bizno.sum()), "M mean", round(U.M.mean(), 3),
        "M|L=1", round(U[U.L == 1].M.mean(), 3), "M|L=0", round(U[U.L == 0].M.mean(), 3))
