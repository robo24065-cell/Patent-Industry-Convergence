# -*- coding: utf-8 -*-
"""메타 번호 형식 탐색(읽기만)."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd
pd.set_option('display.width', 250)
con = con3()
print(con.execute(f"describe select * from read_parquet('{META}')").df()[['column_name', 'column_type']].to_string())
for cc in ['US', 'JP', 'EP']:
    print(con.execute(f"select cc, appl_no, rgst_no, pub_date, rgst_date, substr(family,1,160) fam from read_parquet('{META}') where cc='{cc}' and family like '%KR%' limit 4").df().to_string())
print(con.execute(f"select cc, count(*) n, count(distinct appl_no) na, sum(family is not null and family<>'')::int nfam from read_parquet('{META}') group by cc order by cc").df().to_string())
