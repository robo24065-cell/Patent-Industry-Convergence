# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
import pandas as pd
hdr, rows = read_ca()
print("CA header", hdr, "rows", len(rows))
from collections import Counter
print("field counts", Counter(len(r) for r in rows))
for r in rows[:3]:
    print([x[:3] + "*" if i >= 3 else x for i, x in enumerate(r)])
c = ro(DART_DB)
for (n, s) in c.execute("select name, sql from sqlite_master where type='table'"):
    print("DART", n, s[:300])
    print(c.execute(f"select count(*) from [{n}]").fetchone())
d = pd.read_parquet(DART_CA)
print("dart corp_applicant", d.shape, list(d.columns))
print(d.head(2).T)
r = ro(REG_DB)
for n in ["applicant", "owner", "patent"]:
    print(n, r.execute(f"select sql from sqlite_master where name='{n}'").fetchone()[0][:500])
b = ro(BR_DB)
for (n, s) in b.execute("select name, sql from sqlite_master where type in ('table','index')"):
    print("BR", n, (s or "")[:300])
