# -*- coding: utf-8 -*-
import sys, sqlite3, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
c = sqlite3.connect(f"file:{ROOT}/수집파이프라인/business_rights.sqlite?mode=ro", uri=True)
HY = "119980045675"
d = pd.read_sql(f"select rgst_no, rgst_date, appl_no, appl_date from right_list where applicant_cd='{HY}' and substr(rgst_no,1,2)='10' and rgst_date>='2023'", c)
print(pd.crosstab(d.rgst_date.str[:4], d.appl_date.str[:4]).to_string())
print(d.head(10).to_string())
print(d.appl_no.str[:6].value_counts().head(10))
