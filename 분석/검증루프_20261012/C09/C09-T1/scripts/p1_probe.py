# -*- coding: utf-8 -*-
# 탐색 probe (판정 미사용): registry_full 출원인 코드로 본 현대/기아 문서 구성
import sys, duckdb, sqlite3, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
RF = ROOT + "/수집파이프라인/registry_full.sqlite"
c = sqlite3.connect(f"file:{RF}?mode=ro", uri=True)
print(c.execute("select rgst_no, applNo, applDate, ipcCd, ksic, rgstDate from patent limit 2").fetchall())
print(c.execute("select * from applicant limit 3").fetchall())
HY, KIA = "119980045675", "119980003181"
q = f"""
with a as (select rgst_no, max(applicantCd='{HY}') hy, max(applicantCd='{KIA}') kia, count(*) na from applicant group by rgst_no)
select substr(p.applDate,1,4) y, sum(hy and not kia) hy_only, sum(hy and kia) both_, sum(kia and not hy) kia_only, sum(hy and na>1 and not kia) hy_other_joint
from a join patent p using(rgst_no) where hy or kia group by 1 order by 1"""
print(pd.read_sql(q, c).to_string())
