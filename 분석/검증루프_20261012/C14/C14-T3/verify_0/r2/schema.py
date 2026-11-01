import sqlite3, pathlib
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
for f in ['business_rights','registry_full']:
    c = sqlite3.connect((B/'수집파이프라인'/(f+'.sqlite')).as_uri()+'?mode=ro', uri=True)
    for n,s in c.execute("select name,sql from sqlite_master where type in ('table','index')"): print(f, n, s)
c = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri()+'?mode=ro', uri=True)
for r in c.execute("select * from right_list limit 2"): print(r)
for r in c.execute("select * from applicant_query limit 2"): print(r)
