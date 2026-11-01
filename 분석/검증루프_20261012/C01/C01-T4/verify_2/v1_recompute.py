# verify_2 v1: s2_lag.json 주장을 원천 sqlite(mode=ro)에서 독립 재계산 (캐시 parquet 미사용)
import sqlite3, os, json, sys
import pandas as pd, numpy as np
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\verify_2")
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
pat = pd.read_sql("select rgst_no, rgstDate, applDate from patent", con)
pay = pd.read_sql("select rgst_no, lastAnnl, statAnnl, payDate from pay", con)
rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName like '%소멸%' or rgstCsName like '%회복%'", con)
con.close()
r = {}
pay["la"] = pd.to_numeric(pay.lastAnnl, errors="coerce")
pay["sa"] = pd.to_numeric(pay.statAnnl, errors="coerce")
r["pay_lastAnnl_na"] = int(pay.la.isna().sum())
mx = pay.groupby("rgst_no").agg(max_annl=("la", "max"), last_pay=("payDate", "max")).reset_index()
df = pat.merge(mx, on="rgst_no", how="left")
df["rg"] = pd.to_datetime(df.rgstDate, format="%Y%m%d", errors="coerce")
# 납부완료일: 등록일 + max_annl 년 (pandas DateOffset로 독립 구현)
ok = df.rg.notna() & df.max_annl.notna()
df.loc[ok, "paid_thru"] = [d + pd.DateOffset(years=int(k)) for d, k in zip(df.rg[ok], df.max_annl[ok])]
df["paid_thru"] = pd.to_datetime(df.paid_thru)
rt["d"] = pd.to_datetime(rt.rgstCsDate, format="%Y%m%d", errors="coerce")
lapev = rt[(rt.rgstCsName == "소멸등록") & rt.rgstCsReason.isin(["등록료불납", "연차료불납"])]
lap_max = lapev.groupby("rgst_no").d.max().rename("lap")
lap_min = lapev.groupby("rgst_no").d.min().rename("lap_min")
rest = rt[rt.rgstCsName.str.startswith("회복")]
rest_max = rest.groupby("rgst_no").d.max().rename("rest")
df = df.merge(lap_max, on="rgst_no", how="left").merge(lap_min, on="rgst_no", how="left").merge(rest_max, on="rgst_no", how="left")
L = df[df.lap.notna()].copy()
L["lag"] = (L.lap - L.paid_thru).dt.days
dev = L[(L.paid_thru >= "2016-01-01") & (L.paid_thru <= "2022-12-31")]
r["dev_n"] = int(len(dev))
qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
r["dev_q"] = {str(k): float(v) for k, v in dev.lag.quantile(qs).items()}
r["dev_q_by_year"] = {int(y): {str(k): float(v) for k, v in g.lag.quantile([0.5, 0.9, 0.95, 0.97, 0.99]).items()} for y, g in dev.groupby(dev.paid_thru.dt.year)}
r["dev_n_by_year"] = {int(y): int(len(g)) for y, g in dev.groupby(dev.paid_thru.dt.year)}
r["dev_share_gt287_by_year"] = {int(y): round(float((g.lag > 287).mean()), 4) for y, g in dev.groupby(dev.paid_thru.dt.year)}
r["dev_share_gt332_by_year"] = {int(y): round(float((g.lag > 332).mean()), 4) for y, g in dev.groupby(dev.paid_thru.dt.year)}
# 이상 시차 분해: 음수, >400
r["dev_neg"] = int((dev.lag < 0).sum()); r["dev_gt400"] = int((dev.lag > 400).sum())
# 분위수 by paid_thru month 2015-07..2025-08 (절단 주의 표시)
L["ptm"] = L.paid_thru.dt.to_period("M")
mm = L[(L.paid_thru >= "2015-01-01") & (L.paid_thru <= "2025-10-31")].groupby("ptm").lag
r["lag_by_ptmonth"] = {str(k): [int(g.size), float(g.quantile(0.5)), float(g.quantile(0.9)), float(g.quantile(0.97)), float(g.max())] for k, g in mm}
# 월별 소멸등록(최신 이벤트 기준과 전체 이벤트 기준 둘 다)
m1 = L[L.lap >= "2023-01-01"].lap.dt.to_period("M").value_counts().sort_index()
m2 = lapev[lapev.d >= "2023-01-01"].d.dt.to_period("M").value_counts().sort_index()
r["monthly_latest"] = {str(k): int(v) for k, v in m1.items()}
r["monthly_allevents"] = {str(k): int(v) for k, v in m2.items()}
a = pd.Series(r["monthly_latest"])
r["avg_2023_24"] = float(a[[k for k in a.index if k[:4] in ("2023", "2024")]].mean())
r["avg_2025_01_08"] = float(a[[k for k in a.index if "2025-01" <= k <= "2025-08"]].mean())
r["avg_2025_09_2026_07"] = float(a[[k for k in a.index if "2025-09" <= k <= "2026-07"]].mean())
# 회복률: 원 주장 정의(납부완료연도별, 회복>=최신소멸) 및 대안 정의
L["restored"] = L.rest.notna() & (L.rest >= L.lap)
r["restore_rate_claimdef"] = {int(y): round(float(g.restored.mean()), 4) for y, g in L[L.paid_thru.dt.year.between(2012, 2024)].groupby(L.paid_thru.dt.year)}
# 대안: 소멸이벤트 연도별(최초 소멸등록 연도), 회복이 하나라도 있는 비율
L["any_rest"] = L.rest.notna()
r["restore_any_by_firstlapyear"] = {int(y): round(float(g.any_rest.mean()), 4) for y, g in L.groupby(L.lap_min.dt.year)}
r["n_patent_with_restore"] = int(df.rest.notna().sum())
r["n_restore_without_lapse"] = int((df.rest.notna() & df.lap.isna()).sum())
r["n_restore_lt_latest_lapse"] = int((df.rest.notna() & df.lap.notna() & (df.rest < df.lap)).sum())
r["n_restore_ge_latest_lapse"] = int((df.rest.notna() & df.lap.notna() & (df.rest >= df.lap)).sum())
# 같은 날 회복/소멸 동시 등록 여부
rr = L[L.restored]
r["restored_same_day_share"] = float((rr.rest == rr.lap).mean()) if len(rr) else None
json.dump(r, open(os.path.join(OUT, "v1_recompute.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
df[["rgst_no", "rg", "max_annl", "paid_thru", "lap", "lap_min", "rest", "last_pay"]].to_parquet(os.path.join(OUT, "v1_base.parquet"), index=False)
print({k: r[k] for k in ["dev_n", "dev_q", "dev_n_by_year", "dev_share_gt287_by_year", "dev_share_gt332_by_year", "avg_2023_24", "avg_2025_01_08", "avg_2025_09_2026_07", "restore_rate_claimdef", "n_patent_with_restore", "n_restore_without_lapse", "n_restore_lt_latest_lapse", "n_restore_ge_latest_lapse", "restored_same_day_share"]})
print(r["dev_q_by_year"])
