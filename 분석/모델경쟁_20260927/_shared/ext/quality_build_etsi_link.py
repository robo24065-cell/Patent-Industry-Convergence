# ETSI SEP 선언 특허 -> 제공 데이터 연결 (공개/등록번호 정규화: 국가코드+숫자, 선행 0 제거)
import duckdb, pandas as pd
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET preserve_insertion_order=false")
con.sql("""create table e as select distinct IPRD_ID, DIPG_ID, patf_id, try_strptime(decl_date,'%Y-%m-%d %H:%M:%S')::date decl_date, g2,g3,g4,g5,
  regexp_extract(trim(t),'^([A-Z]{2})',1) || ltrim(regexp_extract(trim(t),'^[A-Z]{2}([0-9]+)',1),'0') tok
  from (select *, unnest(string_split(publ_no,'|')) t from 'quality_etsi_sep_declarations.parquet' where publ_no is not null and publ_no<>'N/A')
  where regexp_matches(trim(t),'^[A-Z]{2}[0-9]{5,}')""")
con.sql("create table et as select tok, min(decl_date) first_decl, count(distinct IPRD_ID) n_decl, max(g5) g5, max(g4) g4 from e group by 1")
print('ETSI tokens', con.sql("select count(*) from et").fetchall())
con.sql("""create table ptok as select distinct cc, appl_no, appl_year, ksic, main_ipc,
   regexp_extract(t,'^([A-Z]{2})',1) || ltrim(regexp_extract(t,'^[A-Z]{2}([0-9]+)',1),'0') tok
 from (select cc, appl_no, appl_year, ksic, main_ipc, unnest(string_split(coalesce(family,'') || '|' || cc || coalesce(regexp_replace(rgst_no,'[^0-9]','','g'),''),'|')) t
       from read_parquet('../meta/*.parquet')) where regexp_matches(t,'^[A-Z]{2}[0-9]{5,}')""")
con.sql("""create table lk as select p.cc, p.appl_no, p.appl_year, p.ksic, p.main_ipc, min(et.first_decl) first_decl, sum(et.n_decl) n_decl_tok
  from ptok p join et using(tok) group by all""")
con.sql("copy lk to 'quality_etsi_sep_link.parquet' (format parquet)")
r=con.sql("""select ksic, count(distinct cc||appl_no) n_sep, min(first_decl) first_decl from lk group by 1 order by 2 desc""").df()
tot=con.sql("select ksic, count(distinct cc||appl_no) n from read_parquet('../meta/*.parquet') group by 1").df()
r=tot.merge(r,on='ksic',how='left').fillna({'n_sep':0}); r['rate']=r.n_sep/r.n
r.sort_values('n_sep',ascending=False).to_csv('quality_etsi_sep_link_by_ksic.csv',index=False)
print(r.sort_values('n_sep',ascending=False).head(15).to_string()); print('KSIC with >=100 SEP-linked:',(r.n_sep>=100).sum(), ' >=1:',(r.n_sep>=1).sum())
print(con.sql("select cc, count(distinct appl_no) from lk group by 1").fetchall())
