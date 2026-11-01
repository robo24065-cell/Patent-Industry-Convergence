# -*- coding: utf-8 -*-
"""I6b: I6 1차 결과(기관 내 AUC 0.53~0.62로 약함)를 보고 사전 선언한 변형 1회.
가설: 공공기술 이전은 '누가 발명했나'(발명자 과거 사업화 이력)가 좌우한다 — 교수 창업·기업 협력 이력.
발명자 코드(등록원부 inventor.inventorCd)로 같은 발명자의 다른 특허(등록 <= t0말)와 그 특허의 사업화 사건(날짜 <= t0말)을 센다.
순서: M0 -> M0+발명자(M1_inventor) -> +NTIS(M2) -> +셀 수요(M3). 채택 규칙 동일. 사건 캐시는 < 2023-01-01."""
import time, numpy as np, pandas as pd
from r4_common import *
T = time.time()
D = pd.read_parquet(f"{CACHE}/I6_panel.parquet")
EV = pd.read_parquet(f"{CACHE}/I6_events_lt2023.parquet"); assert EV.d.max() < 20230101
c = ro()
# 발명자 코드는 대부분 비어 있어(13만/176만 행) 이름 + 최초 공공권리자 기관 어간으로 키를 만든다(같은 기관 동명이인은 한계로 기록)
import re
iv = pd.read_sql("select rgst_no, inventorName from inventor", c).drop_duplicates()
ow = pd.read_parquet(f"{CACHE}/reg_owner_lt2023.parquet"); ow["d"] = pd.to_numeric(ow.d, errors="coerce")
fd = ow.groupby("rgst_no").d.transform("min"); o1 = ow[ow.d == fd].sort_values("seq").drop_duplicates("rgst_no")
stem = o1.ownerName.fillna("").str.replace(r"\s|재단법인|\(재\)|산학협력단|산학협력재단|산학협력|에리카|ERICA|글로컬|GLOCAL|캠퍼스", "", regex=True)
iv = iv.merge(pd.DataFrame({"rgst_no": o1.rgst_no, "stem": stem}), on="rgst_no")
iv["inventorCd"] = iv.inventorName.fillna("").str.replace(" ", "") + "|" + iv.stem
iv = iv[iv.inventorName.fillna("").str.len() >= 2]
pt = pd.read_parquet(f"{CACHE}/reg_patent.parquet", columns=["rgst_no", "rgstDate"]); pt["rg"] = pd.to_numeric(pt.rgstDate, errors="coerce")
iv = iv.merge(pt[["rgst_no", "rg"]], on="rgst_no")
print("inventor rows", len(iv), round(time.time() - T), flush=True)
out = []
for t0, P in D.groupby("t0"):
    end = t0 * 10000 + 1231
    ivt = iv[iv.rg <= end]
    npat = ivt.groupby("inventorCd").rgst_no.nunique()
    ev = EV[EV.d <= end].drop_duplicates("rgst_no")
    comm = ivt[ivt.rgst_no.isin(set(ev.rgst_no))].groupby("inventorCd").rgst_no.nunique()
    ev3 = EV[(EV.d <= end) & (EV.d > end - 30000)].drop_duplicates("rgst_no")
    comm3 = ivt[ivt.rgst_no.isin(set(ev3.rgst_no))].groupby("inventorCd").rgst_no.nunique()
    x = ivt[ivt.rgst_no.isin(set(P.rgst_no))].copy()
    x["np"] = x.inventorCd.map(npat); x["cm"] = x.inventorCd.map(comm).fillna(0); x["cm3"] = x.inventorCd.map(comm3).fillna(0)
    # 자기 특허의 과거 사업화는 prior_comm으로 이미 있으므로 발명자 수치에서 뺀다
    own = x.rgst_no.isin(set(ev.rgst_no)).astype(int); own3 = x.rgst_no.isin(set(ev3.rgst_no)).astype(int)
    x["cm"] = x.cm - own; x["cm3"] = x.cm3 - own3
    gx = x.groupby("rgst_no")
    f = pd.DataFrame({"inv_np_max": np.log1p(gx.np.max()), "inv_cm_max": np.log1p(gx.cm.max()), "inv_cm3_max": np.log1p(gx.cm3.max()),
                      "inv_cm_rate": gx.apply(lambda s: (s.cm.sum() + 0.1) / (s.np.sum() + 5))})
    P = P.merge(f, left_on="rgst_no", right_index=True, how="left")
    P["inv_any_cm"] = (P.inv_cm_max > 0).astype(float)
    out.append(P)
    print(t0, round(P.inv_any_cm.mean(), 3), round(P[P.y == 1].inv_any_cm.mean(), 3), round(P[P.y == 0].inv_any_cm.mean(), 3), round(time.time() - T), flush=True)
D2 = pd.concat(out, ignore_index=True)
D2.to_parquet(f"{CACHE}/I6b_panel.parquet", index=False)
