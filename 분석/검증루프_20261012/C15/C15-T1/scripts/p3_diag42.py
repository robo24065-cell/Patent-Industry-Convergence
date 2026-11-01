# -*- coding: utf-8 -*-
"""C15-T1 p3: 4.2 포기율 정의 진단 — 첫 불납 소멸의 lmax(마지막 납부 연차)·시차 분포를 정부R&D 기원별로"""
import os, sys, sqlite3, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
con = sqlite3.connect("file:" + os.path.join(BASE, "수집파이프라인", "registry_full.sqlite") + "?mode=ro", uri=True)
NP = pd.read_parquet(os.path.join(W, "cache", "s2_np_events.parquet"))
pt = pd.read_sql("select rgst_no, rgstDate from patent where substr(rgst_no,1,2)='10'", con)
pt["rd"] = pd.to_datetime(pt.rgstDate.str[:8], format="%Y%m%d", errors="coerce"); pt = pt[pt.rd.notna()].reset_index(drop=True)
NP["rgst_no"] = pt.rgst_no.values[NP.pid.values]; NP["rd"] = pt.rd.values[NP.pid.values]
f = NP.sort_values("d").drop_duplicates("pid")
fl = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\ntis_patent_flag.parquet"), columns=["rgst_no", "gov_rnd"]).dropna(subset=["rgst_no"])
fl["rgst_no"] = fl.rgst_no.astype(str); g = fl.groupby("rgst_no").gov_rnd.max()
coh = pt[(pt.rd.dt.year >= 2007) & (pt.rd.dt.year <= 2016)].copy()
coh["gov"] = coh.rgst_no.map(g).fillna(False).astype(bool)
coh = coh.merge(f[["rgst_no", "d", "E", "lmax", "lag", "atype"]], on="rgst_no", how="left")
coh["cd_yrs"] = (coh.d - coh.rd).dt.days / 365.25
coh["E_yrs"] = (coh.E - coh.rd).dt.days / 365.25
for gv, s in coh.groupby("gov"):
    n = len(s)
    print("gov", gv, "n", n)
    print("  lmax dist of first lapse (share of cohort):", (s.lmax.value_counts().sort_index() / n).round(4).head(10).to_dict())
    print("  cd<=5y", round((s.cd_yrs <= 5).mean(), 4), " E<=5y", round((s.E_yrs <= 5).mean(), 4), " E<=4y", round((s.E_yrs <= 4.0001).mean(), 4), " E<5y strict", round((s.E_yrs < 4.99).mean(), 4))
    print("  lag median", s.lag.median(), "cd within 5y but E>5?", int(((s.cd_yrs <= 5) & (s.E_yrs > 5)).sum()))
    print("  cd_yrs at lmax==5 median", s[s.lmax == 5].cd_yrs.median(), " E_yrs lmax==5 median", s[s.lmax == 5].E_yrs.median())
    print("  ever lapse", round(s.d.notna().mean(), 4))
