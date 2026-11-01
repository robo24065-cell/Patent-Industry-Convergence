# -*- coding: utf-8 -*-
"""I4 '첫 해외출원(글로벌 진출) 기업 예측' 학습표 — 개발 원점 t0=2008..2018만.
기업 = 국내법인 출원인코드(1로 시작). 해외출원 사건 = 제공 데이터의 US·EP·JP 출원 중 우선권국 KR이고
패밀리 토큰(KR 등록·공개·출원번호)이 그 기업의 KR 출원과 이어지는 것. 사건 시각 = 가장 이른 우선일, 알려진 시각 = 해외 공개일.
결과: t0에 해외출원이 '알려지지 않은' 기업이 우선일 t0+1..t0+2 해외출원을 하는가. 🔒 최종시험 t0=2020(우선일 2021–22, F=2021) → 여기선 t0≤2018(우선일 ≤2020)만."""
import os, time, sqlite3
import duckdb, pandas as pd, numpy as np
from i00_common import *

t = time.time()
def log(*a):
    print(*a, round(time.time() - t), flush=True)

SEAL_I4_PRIO_YEAR = 2021
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "등록번호", "공개번호"])
k.columns = ["appl_no", "rg", "pb"]
tok = []
for c in ["appl_no", "rg", "pb"]:
    s = k[c].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    if c == "rg":
        v = "KR" + s.str[:9]
    elif c == "pb":
        v = "KR" + s.str[2:]
    else:
        v = "KR" + s
    tok.append(pd.DataFrame({"tok": v, "appl_no": k.appl_no.astype(str)})[s.str.len() >= 9])
tok = pd.concat(tok).drop_duplicates("tok")
tmap = dict(zip(tok.tok, tok.appl_no))
log("KR 토큰", len(tmap))

meta = os.path.join(MC, "_shared", "meta", "*.parquet").replace("\\", "/")
fo = duckdb.sql(f"""select distinct cc, appl_no, appl_date, pub_date, prio_date, family, ksic from read_parquet('{meta}')
                 where cc in ('US','EP','JP') and prio_cc like '%KR%'""").df()
log("해외 KR우선 기록", len(fo))
fo["fkey"] = fo.cc + fo.appl_no.astype(str)
fo["prio"] = fo.prio_date.fillna("").str.split("|").map(lambda L: min([int(x) for x in L if x.isdigit() and len(x) == 8], default=np.nan))
fo["known"] = pd.to_numeric(fo.pub_date, errors="coerce")
fo["tok"] = fo.family.fillna("").str.split("|")
ex = fo[["fkey", "prio", "known", "tok"]].explode("tok")
ex = ex[ex.tok.str.startswith("KR", na=False)]
ex["kr_appl"] = ex.tok.map(tmap)
ex = ex.dropna(subset=["kr_appl"]).drop_duplicates(["fkey", "kr_appl"])
log("해외기록-KR출원 연결", len(ex), "해외기록 연결률", round(ex.fkey.nunique() / fo.fkey.nunique(), 3))

AF = pd.read_parquet(os.path.join(CACHE, "app_firm.parquet"))
AF = AF[AF.applicantCd.astype(str).str[0] == "1"].copy()
AF["appl_no"] = AF.appl_no.astype(str)
ev = ex.merge(AF[["appl_no", "applicantCd"]].drop_duplicates(), left_on="kr_appl", right_on="appl_no")
ev = ev.groupby(["applicantCd", "fkey"]).agg(prio=("prio", "min"), known=("known", "min")).reset_index()
ev["py"] = ev.prio // 10000
ev.to_parquet(os.path.join(CACHE, "I4_foreign_events.parquet"), index=False)
# KR 출원 단위 '해외 가족 알려진 시각'(업종 관행 피처용)
kr_abroad = ex.merge(fo[["fkey"]].drop_duplicates(), on="fkey").groupby("kr_appl").known.min()
log("기업 해외사건", len(ev), "기업", ev.applicantCd.nunique())

# 레지스트리 청구항(등록건)
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
cl = pd.read_sql("select applNo appl_no, claimCount, rgstDate from patent", c)
cl["claimCount"] = pd.to_numeric(cl.claimCount, errors="coerce")
cl["rgstDate"] = pd.to_numeric(cl.rgstDate, errors="coerce")
AF = AF.merge(cl.drop_duplicates("appl_no"), on="appl_no", how="left")
AF["abroad_known"] = AF.appl_no.map(kr_abroad)

# KSIC 수준 신호
kp = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "세세분류코드"])
kp.columns = ["appl_no", "ay", "ksic"]; kp["ksic"] = kp.ksic.astype(str); kp["appl_no"] = kp.appl_no.astype(str)
kp["abroad_known"] = kp.appl_no.map(kr_abroad)
KR = kp.drop_duplicates(["appl_no", "ksic"]).groupby(["ksic", "ay"]).size().unstack(fill_value=0)
KS = sorted(KR.index)
FC = pd.read_parquet(os.path.join(EXT, "science_patent_counts_ksic_cc_year.parquet"))
FR = FC[FC.cc != "KR"].groupby(["ksic", "year"]).n_pat.sum().unstack(fill_value=0)
OA = pd.read_parquet(os.path.join(EXT, "science_openalex_ksic_extra_year.parquet"))
OA = OA[OA.variant == "company_ALL"].pivot(index="ksic", columns="year", values="n_works").fillna(0)
PP = pd.read_parquet(os.path.join(EXT, "demand_ksic_price_panel_1996_2021.parquet"))
PP = PP[PP.series == "ppi_basic"].pivot(index="ksic", columns="year", values="dl")
hm = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str)
xe = pd.read_csv(os.path.join(EXT, "trade_kr_export_hs6_uv_2000_2021.csv"), dtype={"hs6": str})
xe = xe.merge(hm[["k5", "hs6"]].drop_duplicates(), on="hs6")
XP = xe.groupby(["k5", "year"]).export_usd.sum().unstack(fill_value=0)


def wsum(M, ys):
    cols = [y for y in ys if y in M.columns]
    return M[cols].sum(axis=1) if cols else pd.Series(0.0, index=M.index)


def sig(t0):
    s = pd.DataFrame(index=KS)
    s["s_kr"] = np.log((wsum(KR, [t0 - 3, t0 - 2]) + 1) / (wsum(KR, [t0 - 5, t0 - 4]) + 1)).reindex(KS)
    s["s_fr"] = np.log((wsum(FR, [t0 - 3, t0 - 2]) + 1) / (wsum(FR, [t0 - 5, t0 - 4]) + 1)).reindex(KS)
    w = kp[kp.ay.between(t0 - 4, t0 - 2)]
    s["s_abroad_rate"] = (w.assign(a=w.abroad_known <= t0 * 10000 + 1231).groupby("ksic").a.mean()).reindex(KS)
    s["s_sci"] = np.log((wsum(OA, [t0 - 3, t0 - 2, t0 - 1]) + 1) / (wsum(OA, [t0 - 6, t0 - 5, t0 - 4]) + 1)).reindex(KS)
    cols = [y for y in [t0 - 3, t0 - 2, t0 - 1] if y in PP.columns]
    s["s_ppi"] = PP[cols].sum(axis=1, min_count=len(cols)).reindex(KS)
    s["s_exp"] = np.log((wsum(XP, [t0 - 3, t0 - 2, t0 - 1]) + 1) / (wsum(XP, [t0 - 6, t0 - 5, t0 - 4]) + 1)).reindex(KS)
    s["s_explvl"] = np.log1p(wsum(XP, [t0 - 1])).reindex(KS)
    return s


# NPS(2015~만)
NPS = {}
for s in SNAPS_OK:
    NPS[s] = pd.read_parquet(os.path.join(CACHE, f"nps_firm_{s}.parquet"), columns=["key", "n_emp", "active"])
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
a2k = dict(zip(lk.applicantCd, lk.key))

rows = []
for t0 in range(2008, 2019):
    assert t0 + 2 < SEAL_I4_PRIO_YEAR
    end = t0 * 10000 + 1231
    kn = AF[AF.known_date <= end].copy()
    kn["reg_t0"] = kn.rgstDate <= end                       # t0까지 등록된 것만(사후 등록 여부 누수 방지)
    kn["cl_t0"] = kn.claimCount.where(kn.reg_t0)
    g = kn.groupby("applicantCd")
    f = pd.DataFrame({"n_app": g.appl_no.nunique(), "first_known": g.known_date.min(), "last_known": g.known_date.max(),
                      "n_ksic": g.ksic.nunique(), "claims": g.cl_t0.mean(), "reg_share": g.reg_t0.mean()})
    f = f[f.last_known > (t0 - 5) * 10000 + 1231]           # 최근 5년 활동 기업
    known_abroad = set(ev.loc[ev.known <= end, "applicantCd"])
    f = f[~f.index.isin(known_abroad)].copy()
    f["log_app"] = np.log1p(f.n_app)
    f["pat_age"] = t0 - f.first_known // 10000
    f["yrs_since_last"] = t0 - f.last_known // 10000
    a_new = kn[kn.ay.between(t0 - 3, t0 - 2)].groupby("applicantCd").appl_no.nunique()
    a_old = kn[kn.ay.between(t0 - 5, t0 - 4)].groupby("applicantCd").appl_no.nunique()
    f["app_trend"] = np.log((a_new.reindex(f.index).fillna(0) + 1) / (a_old.reindex(f.index).fillna(0) + 1))
    f["known_2y"] = kn[kn.known_date > (t0 - 2) * 10000 + 1231].groupby("applicantCd").appl_no.nunique().reindex(f.index).fillna(0)
    # KR 출원 중 (이미 알려진) 해외가족 비율은 정의상 0 → 대신 '자사 KR 출원 중 PCT/WO 표기' 없음. 기업 KSIC 노출
    W = kn[kn.applicantCd.isin(f.index)].groupby(["applicantCd", "ksic"]).appl_no.nunique().unstack(fill_value=0).reindex(columns=KS, fill_value=0)
    W = W.div(W.sum(1), axis=0)
    S = sig(t0)
    for cname in S.columns:
        v = S[cname].reindex(KS); m = v.notna().values
        ww = W.values[:, m]; den = ww.sum(1)
        f.loc[W.index, "x_" + cname] = np.where(den > 0, ww @ v.values[m] / np.where(den > 0, den, 1), np.nan)
    # 동료 효과: 같은 주력 KSIC 기업 중 최근 2년 '첫 해외' 드러난 비율
    main = W.idxmax(axis=1)
    fe = ev.groupby("applicantCd").known.min()
    recent = fe[(fe > (t0 - 2) * 10000 + 1231) & (fe <= end)].index
    allact = kn.groupby("applicantCd").ksic.agg(lambda s: s.mode().iat[0])
    peer = pd.Series(allact.index.isin(recent), index=allact.index).groupby(allact).mean()
    f["x_peer_first_abroad"] = main.map(peer).reindex(f.index)
    # NPS (t0≥2015)
    snap = f"{t0}-12"
    if snap in NPS:
        d = NPS[snap]; d = d[d.active].set_index("key").n_emp
        emp = f.index.map(lambda a: d.get(a2k.get(a), np.nan))
        f["nps_log_emp"] = np.log1p(np.asarray(emp, dtype=float))
    else:
        f["nps_log_emp"] = np.nan
    # 결과
    pos = set(ev.loc[ev.py.between(t0 + 1, t0 + 2), "applicantCd"])
    f["y"] = f.index.isin(pos).astype(int)
    f["t0"] = t0
    rows.append(f.reset_index())
    log(t0, len(f), int(f.y.sum()), round(f.y.mean(), 4))
D = pd.concat(rows, ignore_index=True)
D.to_parquet(os.path.join(CACHE, "I4_panel.parquet"), index=False)
