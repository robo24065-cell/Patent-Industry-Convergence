# 탐색: 사건 범주·연도(날짜 < 2023-01-01만. 2023 이후 사건은 조회하지 않음)
import sqlite3, pandas as pd
R=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect(f"file:{R}?mode=ro",uri=True)
CUT='20230101'
q=f"""select rgstCsName, substr(rgstCsDate,1,4) y, count(*) n from right
where rgstCsDate < '{CUT}' and rgstCsDate>='2005' group by 1,2"""
d=pd.read_sql(q,con)
pv=d.pivot_table(index='rgstCsName',columns='y',values='n',aggfunc='sum').fillna(0).astype(int)
pv['tot']=pv.sum(1); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(pv.sort_values('tot',ascending=False).head(40))
print(pd.read_sql("select rgstCsDate, rgstCsName, rgstCsPupos, rgstCsReason from right where rgstCsName like '%질권%' and rgstCsDate<'20230101' limit 8",con))
for t,c in [('elsLcnsList','elsLcnsStartDate'),('noelsLcnsList','noelsLcnsStartDate')]:
    x=pd.read_sql(f"select substr({c},1,4) y, count(*) n, count(distinct rgst_no) np from {t} where {c}<'{CUT}' group by 1",con)
    print(t); print(x.tail(20).to_string())
    print(pd.read_sql(f"select * from {t} where {c}<'{CUT}' limit 3",con).drop(columns=[c for c in [] ]).T)
