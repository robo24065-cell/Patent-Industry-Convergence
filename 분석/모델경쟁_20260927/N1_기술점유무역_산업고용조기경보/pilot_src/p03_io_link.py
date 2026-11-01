# -*- coding: utf-8 -*-
"""산업연관표(한국은행 접속표, 2010 경상, 통합소분류 3자리) ↔ KSIC 5자리 연결표와 전후방 가중치.
KSIC5 → IO 기본부문(4자리, 대조표) → 통합소분류(앞 3자리). 개발 원점은 2010표(공표 약 2014 — t0<2014 원점엔 구조 선견 가능성, 보고), 최종시험은 2015표 사용 예정."""
import pandas as pd, numpy as np, os
from ilib import *
IO = os.path.join(BASE, "공모전데이터", "외부데이터", "산업연관표")
k = pd.read_excel(os.path.join(IO, "IO코드와 KSIC 및 HS코드 대조표.xlsx"), sheet_name="IO코드와 KSIC 비교표", header=1, dtype=str)
k.columns = ["io", "nm", "ksic"]; k["io"] = k.io.ffill().str.strip(); k = k.dropna(subset=["ksic"])
k["k5"] = k.ksic.str.strip().str[:5]; k["s3"] = k.io.str[:3]
link = k[["k5", "s3"]].drop_duplicates()
link.to_csv(f"{CACHE}/ksic5_io3_link.csv", index=False)
def ztab(yr):
    d = pd.read_excel(os.path.join(IO, "2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx"), sheet_name=f"{yr}-A표_총거래표(생산자)", header=None)
    cols = d.iloc[4].astype(str).str.strip().tolist(); body = d.iloc[6:].copy(); body.columns = cols
    body = body[body.iloc[:, 0].astype(str).str.fullmatch(r"\d{3}")].copy(); body.index = body.iloc[:, 0].astype(str)
    sec = [c for c in cols if len(c) == 3 and c.isdigit()]
    Z = body[sec].apply(pd.to_numeric, errors="coerce").fillna(0)
    ex = pd.to_numeric(body[[c for c in cols if "수출" in c][0]], errors="coerce") if any("수출" in c for c in cols) else None
    return Z, body, cols
Z, body, cols = ztab(10)
print("부문 수", Z.shape, "50 KSIC 연결", link[link.k5.isin(load_kosis50()["T02"].index)].k5.nunique())
Z.to_parquet(f"{CACHE}/io2010_Z.parquet")
