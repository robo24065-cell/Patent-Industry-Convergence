# -*- coding: utf-8 -*-
"""C15-T1 s2: 등록원부(mode=ro) 이탈 재고-유량 vs 연보 항등식 → 포착률 r (PREREG §1), 분포·결측·중복·절단(§2), 내부 누락(§3), 시차·소급(§4).
입력(읽기만): registry_full.sqlite, cache/official_latest.csv(s1), C13-T1 cache/reg_alive_by_year.csv, C14-T3 official_compare_patent.csv(교차 확인)
출력: capture_rate_T4.csv, cache/s2_yearly.csv, cache/s2_out.json, cache/s2_np_events.parquet, cache/s2_dist_*.csv"""
import os, sys, json, time, sqlite3
import numpy as np, pandas as pd
t0 = time.time()
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP = os.path.dirname(os.path.dirname(W))
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
OUT = {}
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)


def pdt(s):
    return pd.to_datetime(pd.Series(s, dtype="object").astype(str).str.strip().str[:8], format="%Y%m%d", errors="coerce")


pt = pd.read_sql("select rgst_no, ksic, applDate, rgstDate, cndrtExptnDate from patent", c)
OUT["patent_rows"] = int(len(pt)); OUT["patent_by_prefix"] = pt.rgst_no.str[:2].value_counts().to_dict()
OUT["patent_dup_rgst_no"] = int(pt.rgst_no.duplicated().sum())
pt["rd"] = pdt(pt.rgstDate); pt["ex"] = pdt(pt.cndrtExptnDate); pt["ad"] = pdt(pt.applDate)
OUT["rd_missing"] = int(pt.rd.isna().sum()); OUT["ex_missing"] = int(pt.ex.isna().sum()); OUT["ksic_missing"] = int(pt.ksic.isna().sum())
m = pt.ex.isna() & pt.ad.notna()
pt.loc[m, "ex"] = pt.loc[m, "ad"] + pd.DateOffset(years=20)
P = pt[(pt.rgst_no.str[:2] == "10") & pt.rd.notna()].reset_index(drop=True)
P["pid"] = np.arange(len(P)); idx = pd.Series(P.pid.values, index=P.rgst_no.values)
OUT["n_patent10"] = int(len(P))
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant", c)
a1 = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no").set_index("rgst_no").applicantCd
P["acd"] = P.rgst_no.map(a1).fillna("NA")
P["atype"] = P.acd.str[0].map({"1": "corp", "2": "univ", "3": "inst", "4": "indiv", "5": "fcorp", "6": "findiv"}).fillna("other")
P["foreign"] = P.atype.isin(["fcorp", "findiv"])
psize = P[P.acd != "NA"].groupby("acd").size()
P["port"] = P.acd.map(psize).fillna(1)
P["size"] = pd.cut(P.port, [0, 1, 10, 50, 100, 500, 1e9], labels=["1", "2-10", "11-50", "51-100", "101-500", ">500"])

rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right", c)
rt["d"] = pdt(rt.rgstCsDate)
nm = rt.rgstCsName.fillna(""); rs = rt.rgstCsReason.fillna("").str.replace(" ", "")
typ = pd.Series(None, index=rt.index, dtype="object")
typ[(nm == "소멸등록") & rs.str.contains("불납")] = "NP"
typ[(nm == "소멸등록") & rs.str.contains("기간만료|존속기간")] = "EXP"
typ[(nm == "소멸등록") & (rs == "")] = "EXT_UNK"
typ[nm.isin(["권리의 말소등록", "말소등록", "직권말소등록"])] = "ABN"
inv = nm.str.contains("확정") & ~nm.str.contains("기술평가|이의신청") & rs.str.contains("무효") & ~rs.str.contains("정정")
typ[inv] = "INV"
typ[nm.str.startswith("회복등록")] = "REC"
rt["typ"] = typ
E = rt[rt.typ.notna()].copy()
E["pre"] = E.rgst_no.str[:2]
OUT["event_counts_by_prefix"] = E.groupby(["pre", "typ"]).size().unstack(fill_value=0).to_dict()
OUT["event_date_missing"] = E[E.d.isna()].groupby(["pre", "typ"]).size().to_dict().__repr__()
E = E[E.pre == "10"].copy()
E["pid"] = E.rgst_no.map(idx)
OUT["events10_without_patent"] = int(E.pid.isna().sum())
E = E[E.pid.notna() & E.d.notna()].copy(); E["pid"] = E.pid.astype(int)
E["ord"] = np.where(E.typ == "REC", 1, 0)            # 같은 날 회복은 소멸 뒤
E = E.sort_values(["pid", "d", "ord", "seq"]).reset_index(drop=True)
SNAP = rt.d.max(); OUT["right_max_date"] = str(SNAP.date())


def alive_at(status_date, reg_cut, use_exp=False):
    ok = (P.rd <= reg_cut).values
    e = E[E.d <= status_date]
    last = e.groupby("pid").typ.last()
    dead = np.zeros(len(P), bool)
    dead[last.index.values] = (last.values != "REC")
    a = ok & ~dead
    if use_exp:
        a &= (P.ex > reg_cut).values
    return a


years = list(range(2000, 2026))
rows = []
G = {}
for y in years:
    yend = pd.Timestamp(f"{y}-12-31"); st = pd.Timestamp(f"{y+1}-05-31")
    a = alive_at(st, yend); a_exp = alive_at(st, yend, True); a_dec = alive_at(yend, yend)
    G[y] = a
    rows.append(dict(year=y, A=int(a.sum()), A_exp=int(a_exp.sum()), A_dec=int(a_dec.sum()),
                     A_dom=int((a & ~P.foreign.values).sum()), A_for=int((a & P.foreign.values).sum()),
                     Rp=int((P.rd.dt.year == y).sum()), Rp_dom=int(((P.rd.dt.year == y) & ~P.foreign).sum()), Rp_for=int(((P.rd.dt.year == y) & P.foreign).sum())))
Y = pd.DataFrame(rows).set_index("year")
OFF = pd.read_csv(os.path.join(W, "cache", "official_latest.csv"), index_col=0)
Y = Y.join(OFF[["I", "R"]])
# 공식 내·외국인(4.1.11 존속: 2018–2025, 4.1.3 등록: 2021–2025) — cache/p2_dump.txt, p2_dump_2022.txt 원표에서 옮김
I_dom = {2018: 752309, 2019: 791438, 2020: 833953, 2021: 882651, 2022: 930224, 2023: 975170, 2024: 1007339, 2025: 1049771}
I_for = {2018: 248854, 2019: 256641, 2020: 262768, 2021: 270669, 2022: 283922, 2023: 296589, 2024: 304955, 2025: 316307}
R_dom = {2021: 110351, 2022: 99202, 2023: 99315, 2024: 95165, 2025: 117232}
R_for = {2021: 35531, 2022: 35978, 2023: 35419, 2024: 32641, 2025: 38917}
Y["I_dom"] = pd.Series(I_dom); Y["I_for"] = pd.Series(I_for); Y["R_dom"] = pd.Series(R_dom); Y["R_for"] = pd.Series(R_for)
assert all(Y.loc[y, "I_dom"] + Y.loc[y, "I_for"] == Y.loc[y, "I"] for y in range(2018, 2026))
assert all(Y.loc[y, "R_dom"] + Y.loc[y, "R_for"] == Y.loc[y, "R"] for y in range(2021, 2026))
Y["X"] = Y.I.shift(1) + Y.R - Y.I
Y["Xp"] = Y.A.shift(1) + Y.Rp - Y.A
Y["Xp_exp"] = Y.A_exp.shift(1) + Y.Rp - Y.A_exp
Y["Xp_dec"] = Y.A_dec.shift(1) + Y.Rp - Y.A_dec
for g in ["dom", "for"]:
    Y[f"X_{g}"] = Y[f"I_{g}"].shift(1) + Y[f"R_{g}"] - Y[f"I_{g}"]
    Y[f"Xp_{g}"] = Y[f"A_{g}"].shift(1) + Y[f"Rp_{g}"] - Y[f"A_{g}"]
Y["s"] = Y.A / Y.I
Y["s3"] = (Y.A - Y.Rp.rolling(3).sum()) / (Y.I - Y.R.rolling(3).sum())
Y["s_exp"] = Y.A_exp / Y.I; Y["s_dec"] = Y.A_dec / Y.I
for g in ["dom", "for"]:
    Y[f"s_{g}"] = Y[f"A_{g}"] / Y[f"I_{g}"]
C13 = pd.read_csv(os.path.join(LOOP, "C13", "C13-T1", "cache", "reg_alive_by_year.csv"))
Y["s_C13"] = C13[C13.right == "특허"].set_index("year").s
Y["q"] = Y.Rp / Y.R
# 이탈 사건(창 (y-05-31, y+1-05-31]) 사유 분해: 그 창에서 처음 '소멸 상태'로 바뀐 사건의 종류
tr = []
for y in years[1:]:
    w0, w1 = pd.Timestamp(f"{y}-05-31"), pd.Timestamp(f"{y+1}-05-31")
    e = E[(E.d > w0) & (E.d <= w1) & (E.typ != "REC")]
    cnt = e.groupby("typ").pid.nunique()
    tr.append(dict(year=y, **{f"ev_{k}": int(v) for k, v in cnt.items()}))
Y = Y.join(pd.DataFrame(tr).set_index("year"))
avg = lambda col: (Y[col].shift(1) + Y[col]) / 2
Y["exp_s3"] = Y.X * avg("s3"); Y["r"] = Y.Xp / Y.exp_s3
Y["r_s"] = Y.Xp / (Y.X * avg("s")); Y["r_s_prev"] = Y.Xp / (Y.X * Y.s.shift(1)); Y["r_C13"] = Y.Xp / (Y.X * avg("s_C13"))
Y["r_dec"] = Y.Xp_dec / (Y.X * avg("s_dec")); Y["r_exp"] = Y.Xp_exp / (Y.X * avg("s_exp"))
Y["r_np_lb"] = Y.ev_NP / Y.exp_s3
for g in ["dom", "for"]:
    Y[f"r_{g}"] = Y[f"Xp_{g}"] / (Y[f"X_{g}"] * avg(f"s_{g}"))
Y.to_csv(os.path.join(W, "cache", "s2_yearly.csv"), encoding="utf-8-sig")


def win(ys, num="Xp", den="exp_s3"):
    d = Y.loc[ys]
    return float(d[num].sum() / d[den].sum())


F = [2022, 2023, 2024, 2025]
OUT["r_F"] = win(F); OUT["r_F_np_lb"] = win(F, "ev_NP")
OUT["r_2017_2021"] = win(range(2017, 2022))
OUT["r_F_over_2017_21"] = OUT["r_F"] / OUT["r_2017_2021"]
seg = Y.loc[2017:2025, "r"]; med = float(seg.median())
OUT["r_2017_2025_median"] = med; OUT["r_2017_2025_maxdev"] = float((seg / med - 1).abs().max())
OUT["r_y"] = {int(k): round(float(v), 4) for k, v in Y.r.dropna().items()}
for v in ["r_s", "r_s_prev", "r_C13", "r_dec", "r_exp"]:
    num = "Xp_dec" if v == "r_dec" else "Xp_exp" if v == "r_exp" else "Xp"
    sc = {"r_s": "s", "r_s_prev": None, "r_C13": "s_C13", "r_dec": "s_dec", "r_exp": "s_exp"}[v]
    d = Y.loc[F]
    den = (d.X * (Y.s.shift(1).loc[F])) if sc is None else (d.X * avg(sc).loc[F])
    OUT[f"{v}_F"] = float(d[num].sum() / den.sum())
OUT["r_F_dom"] = float(Y.loc[F, "Xp_dom"].sum() / (Y.loc[F, "X_dom"] * avg("s_dom").loc[F]).sum())
OUT["r_F_for"] = float(Y.loc[F, "Xp_for"].sum() / (Y.loc[F, "X_for"] * avg("s_for").loc[F]).sum())
# 코호트 누적 검정
for ye in (2023, 2025):
    yy = range(2002, ye + 1)
    Rp, R = Y.loc[yy, "Rp"].sum(), Y.loc[yy, "R"].sum()
    # 제공 누적 이탈: 2002 이후 등록분만(제공 원부 2002 이전 등록 205건은 빼고 셈)
    a_post = int((G[ye] & (P.rd.dt.year >= 2002).values).sum())
    q = Rp / R
    OUT[f"r_cum_{ye}"] = float((Rp - a_post) / ((R - Y.loc[ye, "I"]) * q))
    OUT[f"cum_{ye}"] = dict(Rp=int(Rp), R=int(R), A_post=a_post, I=int(Y.loc[ye, "I"]), q=float(q),
                           exit_frac_prov=float((Rp - a_post) / Rp), exit_frac_off=float((R - Y.loc[ye, "I"]) / R))
# 연령 구성 가중 누적(보고): 등록연도별 제공 생존률을 공식 등록 가중으로 합친 기대 존속
surv = {t: float((G[2023] & (P.rd.dt.year == t).values).sum() / max(1, (P.rd.dt.year == t).sum())) for t in range(2002, 2024)}
exp_alive_off = sum(surv[t] * Y.loc[t, "R"] for t in range(2002, 2024))
OUT["cohort_weighted_expected_I2023_from_prov_survival"] = float(exp_alive_off)
OUT["cohort_weighted_ratio_exits"] = float((Y.loc[2002:2023, "R"].sum() - exp_alive_off) / (Y.loc[2002:2023, "R"].sum() - Y.loc[2023, "I"]))
OUT["prov_registered_before_2002"] = int((P.rd.dt.year < 2002).sum())

# ---------------- NP 사건: E(실효일), 시차, 분포
pay = pd.read_sql("select rgst_no, lastAnnl, payDate from pay where substr(rgst_no,1,2)='10'", c)
pay["pid"] = pay.rgst_no.map(idx); pay["l"] = pd.to_numeric(pay.lastAnnl, errors="coerce"); pay["pd"] = pdt(pay.payDate)
OUT["pay_rows10"] = int(len(pay)); OUT["pay_bad"] = int((pay.pid.isna() | pay.l.isna() | pay.pd.isna()).sum())
pay = pay[pay.pid.notna() & pay.l.notna() & pay.pd.notna()].copy(); pay["pid"] = pay.pid.astype(int)
NP = E[E.typ == "NP"][["pid", "d", "seq"]].reset_index(drop=True)
OUT["np_rows"] = int(len(NP)); OUT["np_patents"] = int(NP.pid.nunique())
OUT["np_same_day_dup"] = int(NP.duplicated(["pid", "d"]).sum())
OUT["np_multi_per_patent"] = int((NP.groupby("pid").size() > 1).sum())
mm = NP.reset_index().merge(pay[["pid", "pd", "l"]], on="pid", how="left")
lmax = mm[mm.pd < mm.d].groupby("index").l.max()
NP["lmax"] = NP.index.map(lmax)
OUT["np_without_prior_pay"] = int(NP.lmax.isna().sum())
lmax_all = mm[mm.pd <= mm.d].groupby("index").l.max()
NP["lmax_le"] = NP.index.map(lmax_all)
rdv = P.rd.values[NP.pid.values]
ok = NP.lmax.notna()
Ed = pd.Series(pd.NaT, index=NP.index, dtype="datetime64[ns]")
Ed[ok] = [pd.Timestamp(r) + pd.DateOffset(years=int(k)) for r, k in zip(rdv[ok.values], NP.lmax[ok].values)]
NP["E"] = Ed
NP["lag"] = (NP.d - NP.E).dt.days
NP["cy"] = NP.d.dt.year; NP["ey"] = NP.E.dt.year
OUT["np_lag_negative"] = int((NP.lag < 0).sum()); OUT["np_lag_gt_730"] = int((NP.lag > 730).sum())
lag = NP.dropna(subset=["lag"]).groupby("cy").lag.agg(["count", "median", lambda s: s.quantile(0.97)])
lag.columns = ["n", "median", "p97"]
OUT["lag_by_cd_year"] = {int(k): [int(v.n), float(v["median"]), float(v.p97)] for k, v in lag.iterrows()}
# 같은 날 소멸·회복 짝, 소멸 뒤 소급 납부(납부일 ≤ 소멸등록일인데 원부에서 E 이후 연차를 덮는 행)
rec = E[E.typ == "REC"][["pid", "d"]]
same = NP.merge(rec, on=["pid", "d"], how="inner")
OUT["np_with_same_day_recovery"] = int(same.drop_duplicates(["pid", "d"]).shape[0])
OUT["np_cover_contradiction"] = int((NP.lmax_le > NP.lmax).sum())   # 소멸등록 당일 납부가 연차를 늘림
NP["ksic"] = P.ksic.values[NP.pid.values]; NP["atype"] = P.atype.values[NP.pid.values]; NP["size"] = P["size"].astype(str).values[NP.pid.values]
NP["annl"] = NP.lmax + 1
NP[["pid", "d", "E", "lag", "lmax", "ksic", "atype", "size"]].to_parquet(os.path.join(W, "cache", "s2_np_events.parquet"), index=False)
for col in ["cy", "ey", "ksic", "atype", "size", "annl"]:
    t = NP.groupby(col).size().rename("n_np").to_frame()
    if col in ("ksic", "atype", "size"):
        base = P.groupby(P[col].astype(str)).size().rename("n_patent")
        t.index = t.index.astype(str); t = t.join(base); t["np_per_patent"] = t.n_np / t.n_patent
    t.to_csv(os.path.join(W, "cache", f"s2_dist_{col}.csv"), encoding="utf-8-sig")
OUT["np_by_atype_share"] = (NP.atype.value_counts(normalize=True).round(4)).to_dict()
OUT["patent_by_atype_share"] = (P.atype.value_counts(normalize=True).round(4)).to_dict()
# 스냅샷 끝 절단: 실효월별 NP 수
em = NP[NP.E >= "2023-01-01"].groupby(NP.E.dt.to_period("M")).size()
OUT["np_by_E_month_2023_on"] = {str(k): int(v) for k, v in em.items()}
# ---------------- 내부 누락: 납부완료일+183+400 < 스냅샷, 만료 전, 소멸 미등록
lm = pay.groupby("pid").l.max()
P["lmax_all"] = P.pid.map(lm)
P["paid_thru"] = [pd.Timestamp(r) + pd.DateOffset(years=int(k)) if k == k else pd.NaT for r, k in zip(P.rd.values, P.lmax_all.values)]
PAYEND = pay.pd.max(); OUT["pay_max_date"] = str(PAYEND.date())
due = P.paid_thru.notna() & (P.paid_thru + pd.Timedelta(days=583) < PAYEND) & (P.paid_thru < P.ex)
last_ev = E.groupby("pid").typ.last()
P["last_typ"] = P.pid.map(last_ev)
has_np = P.pid.isin(NP.pid)
other_exit = P.last_typ.isin(["ABN", "INV", "EXP"])
miss = due & ~has_np & ~other_exit
OUT["internal"] = dict(due=int(due.sum()), due_with_np=int((due & has_np).sum()), due_other_exit=int((due & ~has_np & other_exit).sum()),
                       due_missing=int(miss.sum()), missing_rate=float(miss.sum() / max(1, (due & (has_np | miss)).sum())),
                       no_pay_rows=int(P.lmax_all.isna().sum()))
mb = P[miss].groupby(P.paid_thru.dt.year).size()
OUT["internal_missing_by_paidthru_year"] = {int(k): int(v) for k, v in mb.items()}
OUT["internal_missing_sample_lmax"] = P[miss].lmax_all.value_counts().head(10).to_dict()
# ---------------- 규모 대표성(4.1.8, 2022 등록): 설정등록인 = 1순위 출원인코드
r22 = P[P.rd.dt.year == 2022]
per = r22.groupby(r22.acd.where(r22.acd != "NA", "P" + r22.pid.astype(str))).size()
cls = pd.cut(per, [0, 1, 10, 50, 100, 500, 1e9], labels=["1", "2-10", "11-50", "51-100", "101-500", ">500"])
prov = per.groupby(cls, observed=False).sum() / per.sum()
off = pd.Series([27312, 36629, 15719, 9191, 23277, 23052], index=["1", "2-10", "11-50", "51-100", "101-500", ">500"]) / 135180
OUT["size_2022_share"] = dict(prov={k: round(float(v), 4) for k, v in prov.items()}, official={k: round(float(v), 4) for k, v in off.items()})
OUT["size_2022_tvd"] = float(0.5 * (prov.values - off.values).__abs__().sum())
# 개인·법인·내외(4.1.2, 2022 등록)
o22 = dict(indiv_dom=16376, indiv_for=495, corp_dom=82826, corp_for=35483)
p22 = dict(indiv_dom=int((r22.atype == "indiv").sum()), indiv_for=int((r22.atype == "findiv").sum()),
           corp_dom=int(r22.atype.isin(["corp", "univ", "inst", "other"]).sum()), corp_for=int((r22.atype == "fcorp").sum()))
OUT["type_2022"] = dict(prov={k: round(v / len(r22), 4) for k, v in p22.items()}, official={k: round(v / 135180, 4) for k, v in o22.items()})
OUT["secs"] = round(time.time() - t0, 1)
cols = ["A", "Rp", "I", "R", "X", "Xp", "s", "s3", "s_C13", "q", "ev_NP", "ev_EXP", "ev_ABN", "ev_INV", "exp_s3", "r", "r_s", "r_C13", "r_dec", "r_np_lb", "r_dom", "r_for"]
CR = Y.loc[2001:2025, [x for x in cols if x in Y.columns]].copy()
CR["window"] = ["F(결과창)" if y in F else ("T-보고(연령 혼입)" if 2006 <= y <= 2020 else "") for y in CR.index]
CR.to_csv(os.path.join(W, "capture_rate_T4.csv"), encoding="utf-8-sig")
json.dump(OUT, open(os.path.join(W, "cache", "s2_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 250)
print(CR.round(4).to_string())
print(json.dumps({k: v for k, v in OUT.items() if k not in ("lag_by_cd_year", "np_by_E_month_2023_on", "r_y")}, ensure_ascii=False, indent=1, default=str))
