# -*- coding: utf-8 -*-
# 탐색: right 표 사건명 상위 목록, 예고등록 뒤에 오는 사건 종류(말소·확정 등) 분포
import sqlite3, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
v = pd.read_sql("select rgstCsName, count(*) n from right group by 1 order by 2 desc", c)
print(v.head(60).to_string())
print(v[v.rgstCsName.str.contains("말소|확정|취하|예고", na=False)].to_string())
