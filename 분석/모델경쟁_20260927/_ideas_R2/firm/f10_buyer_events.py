# -*- coding: utf-8 -*-
"""아이디어 3(기술 구매기업 예측)용 양수(취득) 사건표. registry owner 이력(ro).
- 이전 사유가 양도류(양도·매각·경매·증여·일부/지분 양도)인 행만. 합병·분할·상속 제외.
- 거래(deal) = (양도인, 양수인, 날짜) 묶음, 거래규모 = 묶음 특허 수.
- 양수인 이름 → DART corp_code: 정규화 이름으로 DART corp_name 및 CORP_APPLICANT 출원인명(→법인번호→DART) 매칭.
🔒 최종시험(원점 2022, 결과 2023–24) 보호: 사건연도 ≤ 2022 만 저장한다.
출력: cache/buy_events.parquet (corp_code, year, deal_size, seller_type, ksic)"""
import os, re, sqlite3
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
C = os.path.join(BASE, r"분석\모델경쟁_20260927\_ideas_R2\firm\cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
YMAX = 2022

con = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
own = pd.read_sql("select o.rgst_no, o.seq, o.ownerName nm, o.ownerRgstCsDate d, o.ownerRgstCsName n, p.ksic "
                  "from owner o join patent p on o.rgst_no=p.rgst_no", con)
con.close()
own["y"] = pd.to_numeric(own.d.astype(str).str[:4], errors="coerce")

def norm(s):
    s = str(s or "")
    s = re.sub(r"주식회사|\(주\)|㈜|유한회사|\(유\)|코리아|\s|\.|,", "", s)
    return s.lower()

own["key"] = own.nm.map(norm)
own = own.sort_values(["rgst_no", "d", "seq"], na_position="first")
# 날짜 묶음별 첫 권리자 = 그 시점 권리자 대표
grp = own.groupby(["rgst_no", "d"], dropna=False).agg(key=("key", "first"), n=("n", "first"), y=("y", "first"),
                                                       ksic=("ksic", "first"), keys=("key", lambda x: list(x))).reset_index()
grp = grp.sort_values(["rgst_no", "d"], na_position="first")
grp["seller"] = grp.groupby("rgst_no").key.shift(1)
nn = grp.n.fillna("")
ev = grp[nn.str.contains("양도|매각|경매|증여") & ~nn.str.contains("합병|분할|상속")].copy()
ev = ev[ev.y <= YMAX]
ev = ev.explode("keys").rename(columns={"keys": "buyer"})
ev = ev[ev.buyer != ev.seller]
ev["deal_size"] = ev.groupby(["seller", "buyer", "d"]).rgst_no.transform("nunique")

dart = pd.read_parquet(os.path.join(C, "dart_firms.parquet"))
m1 = dict(zip(dart.corp_name.map(norm), dart.corp_code))
ca = pd.read_parquet(os.path.join(BASE, r"공모전데이터\외부데이터\DART\corp_applicant.parquet"))
j2c = dict(zip(dart.jurir_no, dart.corp_code))
m2 = {}
for nm, j in zip(ca["출원인명"], ca["jurir"]):
    if j in j2c:
        m2.setdefault(norm(nm), j2c[j])
ev["corp_code"] = ev.buyer.map(m1)
ev.loc[ev.corp_code.isna(), "corp_code"] = ev.loc[ev.corp_code.isna(), "buyer"].map(m2)
print("transfer rows(<=2022)", len(ev), "matched to DART", ev.corp_code.notna().mean().round(3))
e = ev[ev.corp_code.notna()]
e[["corp_code", "rgst_no", "y", "d", "deal_size", "seller", "ksic"]].to_parquet(os.path.join(C, "buy_events.parquet"), index=False)
d = e.drop_duplicates(["corp_code", "seller", "d"])
print(d.groupby("y").agg(deals=("corp_code", "size"), small=("deal_size", lambda s: (s <= 5).sum()), firms=("corp_code", "nunique")).tail(14))
