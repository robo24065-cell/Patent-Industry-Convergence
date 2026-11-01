# -*- coding: utf-8 -*-
"""C04-T2 T4 레이블 독립 재구성 (t4lib/r2lib import 없음). registry_full.sqlite mode=ro.
위험집합 L=2022-01-01, 레이블 = 납부기록 기준 3년(2022-01-01, 2025-01-01] 불납 실효, DATA_END 2025-08-22, 유예 213일.
출력: cache/t4_labels_indep.parquet, cache/t4_s1_out.json"""
import os, sys, json, time, sqlite3, hashlib
import numpy as np, pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
TR = os.path.join(BASE, r"분석\모델경쟁_20260927\T4_특허포기예측")
FD = os.path.join(TR, r"R2\runs_final\F22")
L = np.datetime64("2022-01-01"); END = np.datetime64("2025-01-01"); DE = np.datetime64("2025-08-22"); GR = np.timedelta64(213, "D")


def pdate(s):
    s = pd.Series(s, dtype="object").astype(str).str.strip()
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce").values.astype("datetime64[D]")


def plus_years(d, k):
    """기념일 + k년 (2/29 → 2/28). d: datetime64[D] 배열, k: 정수 배열"""
    d = pd.DatetimeIndex(d)
    y = d.year.values.astype(float) + np.asarray(k, float)
    mo = d.month.values.astype(float); dd = d.day.values.astype(float)
    dd = np.where((mo == 2) & (dd == 29), 28, dd)
    ok = np.isfinite(y) & np.isfinite(mo)
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[D]")
    if ok.any():
        s = pd.Series(y[ok].astype(int)).astype(str).str.zfill(4) + pd.Series(mo[ok].astype(int)).astype(str).str.zfill(2) + pd.Series(dd[ok].astype(int)).astype(str).str.zfill(2)
        out[ok] = pd.to_datetime(s, format="%Y%m%d", errors="coerce").values.astype("datetime64[D]")
    return out


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pt = pd.read_sql("select rowid as rid, rgst_no, ksic, applDate, rgstDate, cndrtExptnDate from patent order by rowid", con)
n_all = len(pt)
pt["rd"] = pdate(pt.rgstDate); pt["ad"] = pdate(pt.applDate); pt["exp"] = pdate(pt.cndrtExptnDate)
pt = pt[(~pd.isna(pt.rd)) & pt.rgst_no.str.startswith("10")].reset_index(drop=True)
pt["pid"] = np.arange(len(pt))
miss_exp = pd.isna(pt.exp).values
pt.loc[miss_exp, "exp"] = plus_years(pt.loc[miss_exp, "ad"].values, np.full(miss_exp.sum(), 20))
idx = pd.Series(pt.pid.values, index=pt.rgst_no.values)
pay = pd.read_sql("select rgst_no, lastAnnl, payDate from pay", con)
n_pay = len(pay)
pay["pid"] = pay.rgst_no.map(idx)
pay["l"] = pd.to_numeric(pay.lastAnnl, errors="coerce"); pay["pd"] = pdate(pay.payDate)
pay = pay[pay.pid.notna() & pay.l.notna() & ~pd.isna(pay.pd)]
pay["pid"] = pay.pid.astype(int)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right", con)
con.close()
rt["pid"] = rt.rgst_no.map(idx); rt = rt[rt.pid.notna()].copy(); rt["pid"] = rt.pid.astype(int); rt["cd"] = pdate(rt.rgstCsDate)
print("load", round(time.time() - t0), len(pt), len(pay), len(rt), flush=True)


def paid_thru(asof):
    p = pay[pay.pd <= asof]
    lm = p.groupby("pid").l.max()
    lv = np.zeros(len(pt)); lv[lm.index.values] = lm.values
    out = np.full(len(pt), np.datetime64("NaT"), dtype="datetime64[D]")
    ok = lv > 0
    out[ok] = plus_years(pt.rd.values[ok], lv[ok].astype(int))
    return out


malso = rt[rt.rgstCsName.isin(["권리의 말소등록", "말소등록", "직권말소등록"])]
pthL = paid_thru(L)
rd = pt.rd.values; ex = pt.exp.values
risk = (rd < L) & (pthL > L) & (ex > L)
risk[malso[malso.cd <= L].pid.unique()] = False
rp = np.where(risk)[0]
pthE = paid_thru(DE)[rp]; exr = ex[rp]
mz = np.isin(rp, malso[(malso.cd > L) & (malso.cd <= END)].pid.unique())
zero = (pthE > END) | ((exr <= END) & (pthE >= exr)) | mz
one = (pthE > L) & (pthE <= END) & (pthE < exr) & (pthE + GR <= DE) & ~mz
y = np.full(len(rp), np.nan); y[zero] = 0; y[one & ~zero] = 1
# 민감도(a): 소멸등록(등록료불납) 사건 기준: E = 등록일 + (소멸등록일 이전 최대 납부연차)년 ∈ (L, END]
lap = rt[(rt.rgstCsName == "소멸등록") & rt.rgstCsReason.isin(["등록료불납", "연차료불납"]) & ~pd.isna(rt.cd)][["pid", "cd"]].reset_index(drop=True)
mm = lap.reset_index().merge(pay[["pid", "pd", "l"]], on="pid")
mm = mm[mm.pd < mm.cd]
lmax = mm.groupby("index").l.max()
lap["lmax"] = lap.index.map(lmax)
lap = lap[lap.lmax.notna()].copy()
lap["E"] = plus_years(rd[lap.pid.values], lap.lmax.values.astype(int))
lapw = lap[(lap.E > L) & (lap.E <= END)]
y_lap = np.isin(rp, lapw.pid.unique()).astype(int)
lap_cd_after = lapw[lapw.cd > DE].pid.nunique()
recov = rt[rt.rgstCsName == "회복등록"]
rec_after = np.isin(rp, recov[recov.cd > L].pid.unique())
out = pd.DataFrame({"pid": rp, "rgst_no": pt.rgst_no.values[rp], "ksic": pt.ksic.values[rp], "y_indep": y, "y_lapse_event": y_lap, "recov_after_L": rec_after})
out.to_parquet(os.path.join(W, "cache", "t4_labels_indep.parquet"), index=False)
# ---- 동결 ev.parquet 대조
ev = pd.read_parquet(os.path.join(FD, "ev.parquet"), columns=["pid", "ksic", "y", "L"])
m = ev.merge(out, on="pid", how="outer", indicator=True)
both = m[m._merge == "both"]
det = both.y_indep.notna()
R = dict(n_patent_table=n_all, n_patent_10_with_rd=int(len(pt)), n_exp_imputed=int(miss_exp.sum()), n_pay_rows=n_pay,
         n_risk_indep=int(len(rp)), n_undetermined_indep=int(np.isnan(y).sum()), pos_indep=int(np.nansum(y)), base_rate_indep=float(np.nanmean(y)),
         n_frozen=int(len(ev)), pos_frozen=int(ev.y.sum()), only_frozen=int((m._merge == "left_only").sum()), only_indep=int((m._merge == "right_only").sum()),
         label_mismatch=int((both.y.values[det.values] != both.y_indep.values[det.values]).sum()),
         ksic_mismatch=int((both.ksic_x.astype(str) != both.ksic_y.astype(str)).sum()),
         L_unique=[str(x) for x in ev.L.unique()],
         sens_lapse_event=dict(pos=int(y_lap.sum()), agree_with_np=float((y_lap[~np.isnan(y)] == y[~np.isnan(y)]).mean()),
                               np1_lap0=int(((y == 1) & (y_lap == 0)).sum()), np0_lap1=int(((y == 0) & (y_lap == 1)).sum()),
                               lapse_regs_after_DATA_END=int(lap_cd_after), note="소멸등록은 실효 뒤 약 321일에 찍혀 2024년 실효분 일부가 DATA_END 뒤라 사건 기준 레이블은 과소"),
         sens_recovery=dict(n_recov_after_L=int(rec_after.sum()), pos_with_recov=int(((y == 1) & rec_after).sum()), neg_with_recov=int(((y == 0) & rec_after).sum())),
         secs=round(time.time() - t0, 1))
json.dump(R, open(os.path.join(W, "cache", "t4_s1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
