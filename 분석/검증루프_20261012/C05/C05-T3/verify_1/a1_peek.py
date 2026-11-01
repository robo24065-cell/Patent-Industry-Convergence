import duckdb,pandas as pd,sys,io,os,sqlite3
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
B=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SH=B+r"\분석\모델경쟁_20260927\_shared"
M=(SH+r"\meta\*.parquet").replace("\\","/")
print(duckdb.sql(f"describe select * from read_parquet('{M}')").df()[['column_name','column_type']].to_string())
print(duckdb.sql(f"select appl_no,rgst_no,appl_date,pub_date,rgst_date from read_parquet('{M}') where cc='KR' limit 5").df())
pr=pd.read_parquet(B+r"\분석\모델경쟁_20260927\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); print(pr.dtypes); print(pr.head()); print(len(pr), pr.y.sum())
c=sqlite3.connect('file:'+B+r"\수집파이프라인\business_rights.sqlite"+'?mode=ro',uri=True)
print(pd.read_sql("select applicant_cd,rgst_no,rgst_date,appl_no,appl_date,business_no from right_list limit 5",c))
print(pd.read_sql("select status,truncated,count(*) n,sum(total_count) tc,sum(fetched) f from applicant_query group by 1,2",c))
print(pd.read_sql("select min(n_in_fusion),max(n_in_fusion),sum(n_in_fusion=0) z from applicant_query",c))
print(pd.read_sql("select substr(rgst_no,1,2) p,count(*) from right_list group by 1",c))
