# -*- coding: utf-8 -*-
"""C15-T1 s4 (사후 탐색, PREREG 변경 기록 1): 4.2 '5년 내 포기 14.2% 대 21.0%'의 창 경계 민감도.
등록 2007–2016 특허('10'), 첫 불납 소멸의 실효일 E = 등록일 + 마지막 납부 연차(년). 지평 h년: E ≤ 등록일+h년(경계 포함)·E < 등록일+h년(경계 제외), 소멸등록일 ≤ 등록일+h년(원 정의).
관측 가능 조건: 등록일+h년 ≤ 2025-10-31(소멸등록 시차 P97 287일, 스냅샷 2026-08-15).
출원인 유형 표준화: 비기원 유형별 비율을 기원 쪽 유형 구성으로 가중(직접 표준화).
입력(읽기만): cache/s2_np_events.parquet(s2), registry_full.sqlite(mode=ro), ntis_patent_flag.parquet. 출력: cache/s4_horizon.csv, cache/s4_out.json"""
import os, sys, json, sqlite3
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
con = sqlite3.connect("file:" + os.path.join(BASE, "수집파이프라인", "registry_full.sqlite") + "?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, rgstDate from patent where substr(rgst_no,1,2)='10'", con)
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant where substr(rgst_no,1,2)='10'", con)
con.close()
pt["rd"] = pd.to_datetime(pt.rgstDate.str[:8], format="%Y%m%d", errors="coerce"); pt = pt[pt.rd.notna()].reset_index(drop=True)
a1 = ap.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no").set_index("rgst_no").applicantCd
pt["atype"] = pt.rgst_no.map(a1).fillna("N").str[0].map({"1": "corp", "2": "univ", "3": "inst", "4": "indiv", "5": "fcorp", "6": "findiv"}).fillna("other")
NP = pd.read_parquet(os.path.join(W, "cache", "s2_np_events.parquet"))
NP["rgst_no"] = pt.rgst_no.values[NP.pid.values]
first = NP.sort_values("d").drop_duplicates("pid").set_index("rgst_no")
fl = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\ntis_patent_flag.parquet"), columns=["rgst_no", "gov_rnd", "has_univ", "has_gri", "has_firm"]).dropna(subset=["rgst_no"])
fl["rgst_no"] = fl.rgst_no.astype(str); fl = fl.groupby("rgst_no")[["gov_rnd", "has_univ", "has_gri", "has_firm"]].max()
c = pt[(pt.rd.dt.year >= 2007) & (pt.rd.dt.year <= 2016)].copy()
for k in ["gov_rnd", "has_univ", "has_gri", "has_firm"]:
    c[k] = c.rgst_no.map(fl[k]).fillna(False).astype(bool)
c["E"] = c.rgst_no.map(first.E); c["d"] = c.rgst_no.map(first.d); c["lmax"] = c.rgst_no.map(first.lmax)
OBS = pd.Timestamp("2025-10-31")
rows = []
grp = {"비기원": ~c.gov_rnd, "정부R&D 기원": c.gov_rnd, "대학 수행": c.has_univ, "출연연 수행": c.has_gri, "기업 수행": c.has_firm}
for h in [3, 4, 5, 6, 7, 8, 10]:
    end = c.rd + pd.DateOffset(years=h)
    ob = end <= OBS
    inc = (c.E.notna() & (c.E <= end)).astype(float); exc = (c.E.notna() & (c.E < end)).astype(float); reg = (c.d.notna() & (c.d <= end)).astype(float)
    for gname, m in grp.items():
        mm = m & ob
        rows.append(dict(h=h, group=gname, n=int(mm.sum()), cohort=f"{int(c[mm].rd.dt.year.min()) if mm.any() else ''}–{int(c[mm].rd.dt.year.max()) if mm.any() else ''}",
                         E_incl=float(inc[mm].mean()), E_excl=float(exc[mm].mean()), regdate=float(reg[mm].mean())))
    # 유형 표준화 격차(비기원을 기원 유형 구성으로)
    g1 = c.gov_rnd & ob; g0 = ~c.gov_rnd & ob
    wt = c[g1].atype.value_counts(normalize=True)
    for col, s in [("E_incl", inc), ("E_excl", exc), ("regdate", reg)]:
        r0 = s[g0].groupby(c[g0].atype).mean()
        std0 = float((r0.reindex(wt.index).fillna(s[g0].mean()) * wt).sum())
        rows.append(dict(h=h, group="비기원(기원 유형 구성으로 표준화)", n=int(g0.sum()), cohort="", **{col: std0}))
H = pd.DataFrame(rows)
H.to_csv(os.path.join(W, "cache", "s4_horizon.csv"), index=False, encoding="utf-8-sig")
lm = c[c.lmax.notna()].groupby("gov_rnd").lmax.value_counts(normalize=False).unstack(0).fillna(0)
lm = lm / c.groupby("gov_rnd").size()
OUT = dict(lmax_share_of_cohort={str(g): {int(k): round(float(v), 4) for k, v in lm[g].items() if k <= 12} for g in lm.columns},
           ever_lapse_by_snapshot={str(g): float(s.d.notna().mean()) for g, s in c.groupby("gov_rnd")},
           lmax5_by_group={gname: float((c[m].lmax == 5).mean()) for gname, m in grp.items()},
           lmax5_by_atype_gov={str(k): float(v) for k, v in c[c.gov_rnd].groupby("atype").lmax.apply(lambda s: (s == 5).mean()).items()},
           lmax5_by_atype_nongov={str(k): float(v) for k, v in c[~c.gov_rnd].groupby("atype").lmax.apply(lambda s: (s == 5).mean()).items()})
json.dump(OUT, open(os.path.join(W, "cache", "s4_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.set_option("display.width", 220)
print(H.pivot_table(index="h", columns="group", values=["regdate", "E_excl", "E_incl"]).round(3).to_string())
print(json.dumps(OUT, ensure_ascii=False, indent=1))
