# -*- coding: utf-8 -*-
"""I6 공공기술 사업화 레이더 — 등록원부 추출(봉인: 사건 날짜 < 2023-01-01만 읽음)."""
import time, pandas as pd, numpy as np
from r4_common import *
t = time.time(); c = ro(); S = SEAL_DATE
pt = pd.read_sql("select rgst_no, ksic, applDate, applNo, claimCount, cndrtExptnDate, ipcCd, rgstDate from patent", c)
pt.to_parquet(f"{CACHE}/reg_patent.parquet", index=False); print("patent", len(pt), round(time.time()-t))
ow = pd.read_sql(f"""select rgst_no, seq, ownerName, ownerNatl, ownerRgstCsDate d, ownerRgstCsName cs from owner
 where ownerRgstCsDate is null or ownerRgstCsDate='' or ownerRgstCsDate < '{S}'""", c)
ow.to_parquet(f"{CACHE}/reg_owner_lt2023.parquet", index=False); print("owner", len(ow), round(time.time()-t))
rt = pd.read_sql(f"select rgst_no, rgstCsDate d, rgstCsName cs, rgstCsReason rs from right where rgstCsDate < '{S}'", c)
rt.to_parquet(f"{CACHE}/reg_right_lt2023.parquet", index=False); print("right", len(rt), round(time.time()-t))
l1 = pd.read_sql(f"select rgst_no, elsLcnsStartDate d, elsLcnsPsnNm nm, 'E' typ from elsLcnsList where elsLcnsStartDate < '{S}'", c)
l2 = pd.read_sql(f"select rgst_no, noelsLcnsStartDate d, noelsLcnsPsnNm nm, 'N' typ from noelsLcnsList where noelsLcnsStartDate < '{S}'", c)
pd.concat([l1, l2]).to_parquet(f"{CACHE}/reg_lic_lt2023.parquet", index=False); print("lic", len(l1), len(l2))
iv = pd.read_sql("select rgst_no, count(*) n_inv from inventor group by rgst_no", c)
iv.to_parquet(f"{CACHE}/reg_ninv.parquet", index=False)
py = pd.read_sql(f"select rgst_no, payDate d, statAnnl, lastAnnl from pay where payDate < '{S}'", c)
py.to_parquet(f"{CACHE}/reg_pay_lt2023.parquet", index=False); print("pay", len(py), round(time.time()-t))
