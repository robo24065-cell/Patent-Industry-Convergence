# -*- coding: utf-8 -*-
"""s1: 등록원부(mode=ro) 출원인 주소 → 시도. 출원인코드 × 등록일 × 시도 캐시(cache/reg_addr.parquet)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *
c = ro(REG)
d = pd.read_sql("select a.applicantCd, a.applicantAddr addr, p.rgstDate from applicant a join patent p on a.rgst_no=p.rgst_no", c)
d["rgstDate"] = pd.to_numeric(d.rgstDate.astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
d["sido"] = d.addr.map(sido_of)
d = d[["applicantCd", "rgstDate", "sido"]]
d.to_parquet(os.path.join(CACHE, "reg_addr.parquet"), index=False)
log("rows", len(d), d.sido.value_counts(normalize=True).round(3).head(12).to_dict())
