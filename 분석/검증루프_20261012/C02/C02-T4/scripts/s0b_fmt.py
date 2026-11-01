# s0b: 키 형식 확인(결과 레이블 없음)
import sys, sqlite3, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC=B+r'\분석\모델경쟁_20260927'
c=sqlite3.connect('file:'+B+r'\수집파이프라인\registry_full.sqlite?mode=ro',uri=True)
print(pd.read_sql("select applNo, opnDate, pubDate, rgstDate, ksic from patent limit 3",c))
print(pd.read_sql("select count(*) n, sum(opnDate is null) no_opn, sum(pubDate is null) no_pub from patent",c))
t=pd.read_parquet(MC+r'\T1_산업대기실\TXT_R9\cache\kr_txtfree.parquet'); print(t.head(3), t.k_free.isna().mean(), t.appl_no.str.len().value_counts().head())
r=pq.ParquetFile(MC+r'\T3_기술셀부상예측\ESSENTIAL_R7\rec_conc.parquet'); print(r.schema_arrow.names, r.metadata.num_rows)
ap=pd.read_parquet(MC+r'\T1_산업대기실\cache\app_pub.parquet'); print(ap.dtypes); print(ap.reg.value_counts(), ap.applicant_cd.str[0].value_counts())
lk=pd.read_parquet(MC+r'\_shared\ext\firm_nps_applicant_link.parquet'); print(lk.columns.tolist(), len(lk), lk.nps_hit.mean())
pn=pq.ParquetFile(MC+r'\_shared\ext\firm_nps_firm_panel.parquet'); print(pn.schema_arrow.names, pn.metadata.num_rows)
