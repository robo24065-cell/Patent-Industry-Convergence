# -*- coding: utf-8 -*-
# 탐색 probe (판정 미사용): business_rights 현대 포트폴리오의 단독/공동 구성 (rgst_date<=20250822)
import sys, sqlite3, pandas as pd, json
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
BR = ROOT + "/수집파이프라인/business_rights.sqlite"
c = sqlite3.connect(f"file:{BR}?mode=ro", uri=True)
HY = "119980045675"
df = pd.read_sql(f"select rgst_no, rgst_date, appl_no, appl_date, raw from right_list where applicant_cd='{HY}' and substr(rgst_no,1,2)='10' and rgst_date<='20250822'", c)
r = df.raw.map(json.loads)
df["pert"] = r.map(lambda x: x.get("pertDiv")); df["ai"] = r.map(lambda x: x.get("applicantInfo") or ""); df["rh"] = r.map(lambda x: x.get("rightHolderInfo") or "")
df["y"] = df.appl_date.str[:4]
print(df.pert.value_counts())
df["hy_appl"] = df.ai.str.contains("현대자동차"); df["kia_appl"] = df.ai.str.contains("기아")
df["n_appl"] = df.ai.str.count(",") + 1
g = df[df.y.between("2005", "2024")].groupby("y").agg(n=("rgst_no", "size"), hy_appl=("hy_appl", "sum"),
    hy_only=("ai", lambda s: ((s.str.contains("현대자동차")) & (~s.str.contains(","))).sum()),
    hy_kia=("ai", lambda s: ((s.str.contains("현대자동차")) & (s.str.contains("기아"))).sum()))
print(g.to_string())
print(df[df.y == "2019"].ai.value_counts().head(10))
print(df.rgst_no.duplicated().sum())
