# -*- coding: utf-8 -*-
"""등록원부 출원인 패널(제공범위 KR 등록특허 656,360): 코드별 첫·마지막 출원연도, 특허 수, 공동출원 비중, 최빈 KSIC, 최빈 이름·주소.
출력 cache/reg_app.parquet (특허×출원인), cache/reg_panel.parquet (코드 단위), cache/s04_out.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
import pandas as pd, numpy as np
r = ro(REG_DB)
A = pd.read_sql('select rgst_no, seq, applicantCd cd, applicantName nm, applicantNatl natl, applicantAddr addr from applicant', r)
P = pd.read_sql('select rgst_no, ksic, applDate, rgstDate from patent', r)
O = {"applicant_rows": len(A), "patents": len(P), "patents_with_applicant": int(A.rgst_no.nunique())}
A["cd"] = A.cd.fillna("").astype(str).str.strip()
O["cd_blank_rows"] = int((A.cd == "").sum())
O["cd_len"] = A.cd.str.len().value_counts().head(5).to_dict()
O["cd_first_digit_rows"] = A.cd.str[:1].value_counts().to_dict()
O["natl_by_first_digit"] = A.groupby(A.cd.str[:1]).natl.agg(lambda s: s.fillna("NA").value_counts().head(4).to_dict()).to_dict()
P["ay"] = pd.to_numeric(P.applDate.astype(str).str[:4], errors="coerce")
A = A.merge(P[["rgst_no", "ksic", "ay"]], on="rgst_no", how="left")
napp = A.groupby("rgst_no").cd.transform("nunique")
A["co"] = (napp >= 2).astype(int)
A.to_parquet(guard_path(os.path.join(CACHE, "reg_app.parquet")))
g = A[A.cd != ""].groupby("cd")
pan = pd.DataFrame({"first_ay": g.ay.min(), "last_ay": g.ay.max(), "n_pat": g.rgst_no.nunique(), "co_share": g.co.mean(),
                    "natl": g.natl.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)})
pan["ksic"] = A[A.cd != ""].groupby(["cd", "ksic"]).size().reset_index().sort_values(0, ascending=False).drop_duplicates("cd").set_index("cd").ksic
pan["reg_name"] = A[A.cd != ""].groupby(["cd", "nm"]).size().reset_index().sort_values(0, ascending=False).drop_duplicates("cd").set_index("cd").nm
pan["n_names"] = g.nm.nunique()
pan["addr"] = A[A.cd != ""].sort_values("ay").groupby("cd").addr.last()
pan.reset_index().to_parquet(guard_path(os.path.join(CACHE, "reg_panel.parquet")))
O["codes"] = len(pan)
O["codes_first_digit"] = pan.index.str[:1].value_counts().to_dict()
json.dump(O, open(guard_path(os.path.join(CACHE, "s04_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
