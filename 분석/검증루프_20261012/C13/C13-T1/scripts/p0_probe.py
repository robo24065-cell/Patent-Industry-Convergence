# -*- coding: utf-8 -*-
# 탐색(PREREG 전): 등록원부 right 표의 심판 예고등록 이름과 연도별 건수만 본다(N8 성적 미계산).
import sys; sys.stdout.reconfigure(encoding="utf-8")
import sqlite3, pandas as pd, sys
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)

rt = pd.read_sql("select rgstCsName, substr(rgstCsDate,1,4) y, count(*) n from right where rgstCsName like '%심판%' or rgstCsName like '%예고%' group by 1,2", c)
pv = rt.pivot_table(index="rgstCsName", columns="y", values="n", aggfunc="sum").fillna(0).astype(int)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40)
print(pv.loc[:, [c for c in pv.columns if c and c >= "2005"]].to_string())
print(pd.read_sql("select substr(rgst_no,1,2) k, count(*) from patent group by 1", c))
