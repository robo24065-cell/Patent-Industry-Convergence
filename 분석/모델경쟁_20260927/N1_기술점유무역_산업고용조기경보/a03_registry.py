# -*- coding: utf-8 -*-
"""KR 등록원부(수집 registry_full.sqlite, 읽기 전용) → (KSIC, 연도) 권리 동학 집계: 설정등록 수, 소멸등록 수, 권리이전 수, 청구항 합.
사건 일자(등록원부 사건일) 기준이므로 연도 y 의 값은 y 시점에 확정된 정보(스냅샷 아님). 출력 cache/reg_ky.parquet, ext_link_registry.csv."""
import sqlite3, pandas as pd, numpy as np
from n1lib import *
db = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
reg = pd.read_sql("select ksic, substr(rgstDate,1,4) y, count(*) n_reg, sum(cast(claimCount as int)) claims from patent where rgstDate is not null group by 1,2", con)
ext = pd.read_sql("""select p.ksic, substr(r.rgstCsDate,1,4) y, count(*) n_ext, sum(case when r.rgstCsReason like '%불납%' then 1 else 0 end) n_ext_fee
                     from right r join patent p on p.rgst_no=r.rgst_no where r.rgstCsName='소멸등록' group by 1,2""", con)
trf = pd.read_sql("""select p.ksic, substr(o.ownerRgstCsDate,1,4) y, count(distinct o.rgst_no) n_trf from owner o join patent p on p.rgst_no=o.rgst_no
                     where o.ownerRgstCsReason like '%이전%' group by 1,2""", con)
tot = pd.read_sql("select count(*) n, count(distinct ksic) k from patent", con)
R = reg.merge(ext, on=["ksic", "y"], how="outer").merge(trf, on=["ksic", "y"], how="outer").fillna(0)
R = R[R.y.str.fullmatch(r"\d{4}")]; R["y"] = R.y.astype(int); R = R[R.y <= 2021]
R.to_parquet(f"{CACHE}/reg_ky.parquet")
K50 = load_kosis50(("T02",))["T02"].index
lk = pd.DataFrame({"ksic": K50}); lk["in_registry"] = lk.ksic.isin(R.ksic)
lk = lk.merge(R[R.y.between(2005, 2018)].groupby("ksic")[["n_reg", "n_ext", "n_trf"]].sum().reset_index(), on="ksic", how="left")
lk.to_csv(f"{HERE}/ext_link_registry.csv", index=False, encoding="utf-8-sig")
print("등록원부 특허", int(tot.n[0]), "KSIC", int(tot.k[0]), "| 50개 연계", lk.in_registry.mean())
print(R.groupby("y")[["n_reg", "n_ext", "n_trf"]].sum().loc[2005:2021].astype(int).to_string())
