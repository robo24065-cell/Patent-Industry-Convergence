# -*- coding: utf-8 -*-
"""C15-T1 p1: 등록원부(mode=ro) 소멸등록 구조 탐색 — 사유·권리유형·연도·중복"""
import sqlite3, json, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
print(pd.read_sql("select name from sqlite_master where type='table'", c).name.tolist())
for t in ["patent", "right", "pay"]:
    print(t, pd.read_sql(f"pragma table_info({t})", c)[["name", "type"]].values.tolist())
q = """select substr(rgst_no,1,2) k, rgstCsName, rgstCsReason, count(*) n, count(distinct rgst_no) np, min(rgstCsDate) dmin, max(rgstCsDate) dmax
from right where rgstCsName like '%소멸%' or rgstCsName like '%포기%' or rgstCsReason like '%포기%' or rgstCsName like '%회복%' group by 1,2,3 order by n desc"""
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(pd.read_sql(q, c).to_string())
print(pd.read_sql("select substr(rgst_no,1,2) k, count(*) from patent group by 1", c))
print(pd.read_sql("select max(rgstCsDate), max(rgstDate) from right join patent using(rgst_no)", c))
print(pd.read_sql("select max(payDate) from pay", c))
q2 = """select substr(rgstCsDate,1,4) y, substr(rgst_no,1,2) k, count(*) n from right where rgstCsName='소멸등록' and rgstCsReason like '%불납%' group by 1,2 order by 1,2"""
print(pd.read_sql(q2, c).pivot(index="y", columns="k", values="n").to_string())
