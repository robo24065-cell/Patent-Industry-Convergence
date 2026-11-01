# -*- coding: utf-8 -*-
"""기업×연도(t0) 특허 피처. t0 시점(cut = t0+1년 1월 1일 미만) 사건만 사용.
원천: registry_full.sqlite(ro) 등록원부 — 출원인코드 기준(현재 권리자 스냅샷 아님), DART 연결 특허만.
출력: cache/patfeat.parquet (corp_code, t0, 피처...), cache/ksic_share.parquet (corp_code, t0, ksic, share)"""
import os, sqlite3
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
T0S = list(range(2012, 2024))

pf = pd.read_parquet(os.path.join(C, "pat_firm.parquet"))
pf = pf[pf.corp_code.notna()].drop_duplicates(["rgst_no", "corp_code"])
rg = set(pf.rgst_no)
con = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
con.execute("create temp table ids(r text primary key)")
con.executemany("insert into ids values(?)", [(r,) for r in rg])
right = pd.read_sql("select rgst_no, rgstCsDate d, rgstCsName n from right where rgst_no in (select r from ids)", con)
own = pd.read_sql("select rgst_no, seq, ownerRgstCsDate d, ownerRgstCsName n from owner where rgst_no in (select r from ids)", con)
pay = pd.read_sql("select rgst_no, payDate d from pay where rgst_no in (select r from ids)", con)
lic = pd.read_sql("select rgst_no, noelsLcnsStartDate d from noelsLcnsList where rgst_no in (select r from ids) union all "
                  "select rgst_no, elsLcnsStartDate d from elsLcnsList where rgst_no in (select r from ids)", con)
cnd = pd.read_sql("select rgst_no, cndrtExptnDate ex, claimCount cc from patent where rgst_no in (select r from ids)", con)
con.close()

def yr(s):
    return pd.to_numeric(s.astype(str).str[:4], errors="coerce")

pf = pf.merge(cnd, on="rgst_no", how="left")
pf["ry"] = yr(pf.rgstDate); pf["ay"] = yr(pf.applDate); pf["exy"] = yr(pf.ex)
lap = right[right.n.fillna("").str.contains("소멸")].copy(); lap["y"] = yr(lap.d)
lap = lap.groupby("rgst_no").y.min()
pf["lapy"] = pf.rgst_no.map(lap)
own = own[own.n.notna() & own.n.str.contains("양도|매각|경매|이전|증여")]
own["y"] = yr(own.d)
tr_first = own.groupby("rgst_no").y.min()           # 최초 양도 연도(출원인 → 타인)
pf["try"] = pf.rgst_no.map(tr_first)
pay["y"] = yr(pay.d); lic["y"] = yr(lic.d)
payc = pay.groupby(["rgst_no", "y"]).size().rename("np").reset_index()
licc = lic.dropna().groupby(["rgst_no", "y"]).size().rename("nl").reset_index()

rows, shares = [], []
for t0 in T0S:
    d = pf[pf.ry <= t0].copy()
    alive = (d.lapy.isna() | (d.lapy > t0)) & (d.exy.isna() | (d.exy > t0))
    d["alive"] = alive
    d["reg3"] = d.ry.between(t0 - 2, t0)
    d["regp3"] = d.ry.between(t0 - 5, t0 - 3)
    d["lap2"] = d.lapy.between(t0 - 1, t0)
    d["tr3"] = d["try"].between(t0 - 2, t0)
    d["alive_m2"] = (d.ry <= t0 - 2) & (d.lapy.isna() | (d.lapy > t0 - 2)) & (d.exy.isna() | (d.exy > t0 - 2))
    d["age"] = np.where(alive, t0 - d.ry, np.nan)
    d["cc"] = pd.to_numeric(d.cc, errors="coerce")
    g = d.groupby("corp_code")
    f = pd.DataFrame({
        "n_all": g.size(), "n_alive": g.alive.sum(), "n_reg3": g.reg3.sum(), "n_regp3": g.regp3.sum(),
        "n_lap2": g.lap2.sum(), "n_tr3": g.tr3.sum(), "n_alive_m2": g.alive_m2.sum(),
        "pat_age": g.age.mean(), "claims": g.cc.mean(), "first_ry": g.ry.min(),
    })
    pm = payc[payc.y.between(t0 - 1, t0)].merge(d[["rgst_no", "corp_code"]], on="rgst_no")
    f["n_pay2"] = pm.groupby("corp_code").np.sum()
    lm = licc[licc.y.between(t0 - 2, t0)].merge(d[["rgst_no", "corp_code"]], on="rgst_no")
    f["n_lic3"] = lm.groupby("corp_code").nl.sum()
    # KSIC 구성 (생존 스톡 기준, 없으면 전체)
    s = d[d.alive].groupby(["corp_code", "ksic"]).size().rename("k").reset_index()
    s["share"] = s.k / s.groupby("corp_code").k.transform("sum")
    f["n_ksic"] = s.groupby("corp_code").size()
    f["hhi"] = s.assign(q=s.share ** 2).groupby("corp_code").q.sum()
    old = set(map(tuple, d[d.ry <= t0 - 3][["corp_code", "ksic"]].drop_duplicates().values))
    new = d[d.reg3][["corp_code", "ksic"]].drop_duplicates()
    new = new[[tuple(x) not in old for x in new.values]]
    f["n_newksic3"] = new.groupby("corp_code").size()
    f = f.fillna({"n_pay2": 0, "n_lic3": 0, "n_ksic": 0, "hhi": 0, "n_newksic3": 0})
    f["t0"] = t0
    rows.append(f.reset_index())
    s["t0"] = t0
    shares.append(s[["corp_code", "t0", "ksic", "share"]])
P = pd.concat(rows, ignore_index=True)
P["reg_mom"] = np.log((P.n_reg3 + 1) / (P.n_regp3 + 1))
P["lap_rate"] = P.n_lap2 / (P.n_alive_m2 + 1)
P["tr_rate"] = P.n_tr3 / (P.n_all + 1)
P["pay_int"] = P.n_pay2 / (P.n_alive + 1)
P["firm_pat_age"] = P.t0 - P.first_ry
P.to_parquet(os.path.join(C, "patfeat.parquet"), index=False)
pd.concat(shares, ignore_index=True).to_parquet(os.path.join(C, "ksic_share.parquet"), index=False)
print(P.groupby("t0").agg(firms=("corp_code", "size"), alive=("n_alive", "sum"), lap=("n_lap2", "sum"), tr=("n_tr3", "sum")))
