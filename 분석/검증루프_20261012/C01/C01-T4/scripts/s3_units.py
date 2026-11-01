# s3: C-A 개발 원점 단위·양성 집계 + 연결 편향 점검. 잠금 원점(2024)은 단위 수만 센다(결과 미생성).
import os, json, sqlite3, time
import pandas as pd, numpy as np
t0clock = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
EXT = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\ext")
lag = json.load(open(os.path.join(OUTD, "s2_lag.json"), encoding="utf-8"))
DELTA = pd.Timedelta(days=lag["Delta_days"])          # 335일 (사전 규칙 결과)
E_MD = lag["E_rule"][5:]                               # '08-31'
df = pd.read_parquet(os.path.join(OUTD, "pat_base.parquet"))
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
pay = pd.read_sql("select rgst_no, cast(lastAnnl as integer) a, payDate from pay", con)
app = pd.read_sql("select rgst_no, applicantCd, applicantAddr from applicant where applicantCd like '1%'", con)
con.close()
pay["pd"] = pd.to_datetime(pay.payDate, format="%Y%m%d", errors="coerce")
app["sido"] = app.applicantAddr.fillna("").str.split().str[0].str[:2]
app = app.drop(columns="applicantAddr")
link = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"), columns=["applicantCd", "nps_hit"]).drop_duplicates("applicantCd")
panel = pd.read_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"), columns=["applicantCd", "snap", "n_emp", "active"])
# 출원인 첫 출원연도(업력 대리) — 제공범위 등록특허 기준
fa = app.merge(df[["rgst_no", "applDate_d"]], on="rgst_no").groupby("applicantCd").applDate_d.min().dt.year.rename("first_appl_y")

def add_years(rg, n):
    y = rg.dt.year + n
    s = pd.to_datetime(dict(year=y, month=rg.dt.month, day=np.where((rg.dt.month == 2) & (rg.dt.day == 29), 28, rg.dt.day)), errors="coerce")
    return s

rows, biasrows, ksic_rows = [], [], []
units_keep = []
for t0 in range(2012, 2025):
    T = pd.Timestamp(f"{t0}-12-31"); W = pd.Timestamp(f"{t0+1}-{E_MD}"); OBS = W + DELTA
    lock = (t0 == 2024)
    # 시점 정합 납부완료일: payDate ≤ T 인 납부만
    pa = pay[pay.pd <= T].groupby("rgst_no").a.max()
    d = df[["rgst_no", "ksic", "rg", "cndrtExptnDate_d", "first_trf_d", "paid_thru", "lapse_reg_d", "restore_reg_d"]].merge(pa.rename("annl_T"), left_on="rgst_no", right_index=True, how="inner")
    d["paid_T"] = add_years(d.rg, d.annl_T)
    live = d[(d.rg <= T) & (d.paid_T > T) & ((d.cndrtExptnDate_d > W) | d.cndrtExptnDate_d.isna()) & ~(d.first_trf_d <= T)].copy()
    n_trf_excl = int(((d.rg <= T) & (d.paid_T > T) & (d.first_trf_d <= T)).sum())
    if not lock:
        restored = d.restore_reg_d.notna() & (d.restore_reg_d >= d.lapse_reg_d) & (d.restore_reg_d <= OBS)
        lap_w = d.lapse_reg_d.notna() & (d.lapse_reg_d <= OBS) & (d.paid_thru > T) & (d.paid_thru <= W) & ~restored
        trf_w = d.first_trf_d.notna() & (d.first_trf_d > T) & (d.first_trf_d <= OBS)
        live["lap"] = lap_w.reindex(live.index).values
        live["trf"] = trf_w.reindex(live.index).values
    else:
        live["lap"] = False; live["trf"] = False
    u = live.merge(app[["rgst_no", "applicantCd"]], on="rgst_no")
    g = u.groupby(["applicantCd", "ksic"]).agg(n_live=("rgst_no", "size"), n_lap=("lap", "sum"), n_gone=("lap", lambda s: 0)).reset_index()
    g["n_gone"] = u.assign(gone=u.lap | u.trf).groupby(["applicantCd", "ksic"]).gone.sum().values
    firm = g.groupby("applicantCd").agg(f_nk=("ksic", "size"), f_live=("n_live", "sum"), f_lap=("n_lap", "sum")).reset_index()
    g = g.merge(firm, on="applicantCd")
    g["y_main"] = (g.n_lap == g.n_live)
    g["y_any"] = (g.n_gone == g.n_live)
    g["wipe"] = (g.f_lap == g.f_live)
    for m in (1, 2, 3):
        for multi in (True, False):
            s = g[(g.n_live >= m) & ((g.f_nk >= 2) if multi else True)]
            rec = dict(t0=t0, m=m, multi_ksic=multi, lock=lock, units=len(s), firms=s.applicantCd.nunique())
            if not lock:
                pos = s[s.y_main]
                rec.update(pos_main=int(s.y_main.sum()), base_rate=round(float(s.y_main.mean()), 5), pos_main_partial=int((s.y_main & ~s.wipe).sum()),
                           pos_main_wipe=int((s.y_main & s.wipe).sum()), pos_any=int(s.y_any.sum()),
                           firms_pos=pos.applicantCd.nunique(), top5_firm_share=round(float(pos.applicantCd.value_counts().head(5).sum() / max(len(pos), 1)), 4))
            rec.update(live_patents=int(len(live)), trf_excluded_patents=n_trf_excl)
            rows.append(rec)
    # 편향 점검: 주 정의(m=2, multi)
    s = g[(g.n_live >= 2) & (g.f_nk >= 2)].merge(link, on="applicantCd", how="left").merge(fa, left_on="applicantCd", right_index=True, how="left")
    s["nps_hit"] = s.nps_hit.fillna(False)
    sido = app.drop_duplicates("applicantCd")[["applicantCd", "sido"]]
    s = s.merge(sido, on="applicantCd", how="left")
    if t0 >= 2015:
        snap = f"{t0}-12"
        pe = panel[panel.snap == snap][["applicantCd", "n_emp", "active"]]
        s = s.merge(pe, on="applicantCd", how="left")
    else:
        s["n_emp"] = np.nan; s["active"] = np.nan
    s["t0"] = t0
    units_keep.append(s[["t0", "applicantCd", "ksic", "n_live", "f_nk", "f_live", "y_main", "y_any", "wipe", "nps_hit", "first_appl_y", "sido", "n_emp", "active"]].assign(y_main=lambda x: x.y_main if not lock else np.nan))
    print(t0, len(s), time.time() - t0clock, flush=True)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUTD, "s3_counts.csv"), index=False, encoding="utf-8-sig")
U = pd.concat(units_keep, ignore_index=True)
U.to_parquet(os.path.join(OUTD, "s3_units_main.parquet"), index=False)
print("done", time.time() - t0clock)
