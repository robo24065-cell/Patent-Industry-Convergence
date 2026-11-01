# -*- coding: utf-8 -*-
"""산업연관표(한국은행 접속표 2010·2015, 생산자가격 통합소분류) → KSIC 5자리 구조 피처 + 연계표.
KSIC5 → IO 기본부문(대조표) → 통합소분류 3자리(파일럿 cache/ksic5_io3_link.csv). 다중 부문은 합산.
피처: io_exr 수출/총산출, io_imr 수입/총공급, io_lab 피용자보수/총산출, io_inv 고정자본형성/총수요, io_upm 중간투입 중 수입품 비중,
      io_dn_cn 하류(판매처) 부문 가중 대중 수입점유(파일럿 KSIC 수준을 부문으로 평균 — 원점별로 계산은 b01에서).
원점 규칙: 2010표(2014 공표)는 개발 원점(t0 ≤ 2018), 2015표(2019 공표)는 최종시험(t0=2021)에 사용. t0<2014 원점의 2010표 사용은 구조 선견 소지 → 한계로 기록."""
import numpy as np, pandas as pd, os
from n1lib import *
IO = os.path.join(BASE, "공모전데이터", "외부데이터", "산업연관표", "2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx")
link = pd.read_csv(f"{CACHE}/ksic5_io3_link.csv", dtype=str)
K50 = load_kosis50(("T02",))["T02"].index
def read(sheet):
    d = pd.read_excel(IO, sheet_name=sheet, header=None)
    cols = d.iloc[4].astype(str).str.strip().tolist(); body = d.iloc[6:].copy(); body.columns = cols
    body = body[body.iloc[:, 0].astype(str).str.strip().str.fullmatch(r"\d{3}|9\d{3}")].copy()
    body.index = body.iloc[:, 0].astype(str).str.strip()
    return body.drop(columns=[cols[0], cols[1]]).apply(pd.to_numeric, errors="coerce").fillna(0)
out = []
for yr in ["10", "15"]:
    T = read(f"{yr}-A표_총거래표(생산자)"); M = read(f"{yr}-A표_수입거래표(생산자)")
    sec = [c for c in T.columns if len(c) == 3 and c.isdigit() and c in T.index]
    Z = T.loc[sec, sec]; Zm = M.loc[sec, sec]
    X = T.loc[sec, "9310"]; EXP = T.loc[sec, "9140"]; IMP = -T.loc[sec, "9321"] if T.loc[sec, "9321"].sum() < 0 else T.loc[sec, "9321"]
    SUP = T.loc[sec, "9390"]; INV = T.loc[sec, "9121"] + T.loc[sec, "9122"]; DEM = T.loc[sec, "9290"]
    COMP = T.loc["9610", sec]; II = T.loc["9590", sec]; IIm = Zm.sum(0)
    for k5 in K50:
        ss = [s for s in link[link.k5 == k5].s3 if s in sec]
        if not ss: continue
        dn = Z.loc[ss].sum(0).drop(ss, errors="ignore")
        out.append({"ksic": k5, "tab": int("20"+yr), "n_sec": len(ss), "sectors": "|".join(ss),
                    "io_exr": EXP[ss].sum()/X[ss].sum(), "io_imr": IMP[ss].sum()/SUP[ss].sum(), "io_lab": COMP[ss].sum()/X[ss].sum(),
                    "io_inv": INV[ss].sum()/DEM[ss].sum(), "io_upm": IIm[ss].sum()/max(II[ss].sum(), 1),
                    "io_dn_self": Z.loc[ss, ss].values.sum()/max(Z.loc[ss].values.sum(), 1)})
    # 하류 가중치(부문→부문) 저장: 하류 대중노출 계산용
    Z.to_parquet(f"{CACHE}/io20{yr}_Z_n1.parquet")
S = pd.DataFrame(out); S.to_parquet(f"{CACHE}/io_struct.parquet")
lk = link[link.k5.isin(K50)].copy(); lk["in_2010"] = lk.s3.isin(S[S.tab == 2010].sectors.str.split("|").explode())
lk.to_csv(f"{HERE}/ext_link_io.csv", index=False, encoding="utf-8-sig")
print("연계율(50개 KSIC 중 IO 부문 연결)", S.groupby("tab").ksic.nunique().to_dict(), "평균 부문수", S.groupby("tab").n_sec.mean().round(2).to_dict())
print(S.groupby("tab")[["io_exr", "io_imr", "io_lab", "io_inv", "io_upm"]].describe().T.round(3).to_string())
print(S[S.tab == 2010].set_index("ksic")[["io_exr", "io_imr", "io_lab"]].corrwith(S[S.tab == 2015].set_index("ksic")[["io_exr", "io_imr", "io_lab"]], method="spearman"))
