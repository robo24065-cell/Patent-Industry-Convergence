# -*- coding: utf-8 -*-
# 탐색 probe: business_rights 등록연도 분포, 스냅샷 끝 확인
import sys, sqlite3, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
c = sqlite3.connect(f"file:{ROOT}/수집파이프라인/business_rights.sqlite?mode=ro", uri=True)
HY = "119980045675"
print(pd.read_sql(f"select substr(rgst_date,1,4) ry, count(*) n, max(rgst_date) mx from right_list where applicant_cd='{HY}' and substr(rgst_no,1,2)='10' group by 1 order by 1", c).tail(15).to_string())
print(pd.read_sql(f"select substr(rgst_date,1,4) ry, count(*) n from right_list where substr(rgst_no,1,2)='10' and rgst_date>='2015' group by 1 order by 1", c).to_string())
print(pd.read_sql("select substr(ts,1,7) m, count(*) n, sum(total_count) tc from applicant_query where status='OK' group by 1", c).to_string())
