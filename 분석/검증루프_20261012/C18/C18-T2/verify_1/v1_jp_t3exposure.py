# JP 공개일 없는 등록 행(T3 least(pub,rgst)에서 등록일이 공개연도가 되는 행)의 시점별 분포와 T3 JP 카운트 노출
import duckdb,sys,json,os
sys.stdout.reconfigure(encoding='utf-8')
M='C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
c=duckdb.connect(); c.execute("set threads=2; set memory_limit='3GB'")
c.execute(f"""create table j as select appl_no, appl_year, appl_date, pub_date, rgst_date, appl_no like '%U' u,
  (pub_date is null or pub_date='') nopub,
  left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)::int py,
  date_diff('day', strptime(appl_date,'%Y%m%d'), strptime(nullif(rgst_date,''),'%Y%m%d'))/365.25 lag
  from read_parquet('{M}') where cc='JP' and appl_year>=2000""")
R={}
# 특허(비U) 공개일 없음 행: 등록연도별, 출원→등록 2년 이상(조기등록으로 설명 안 되는) 행 수
R['nopub_patent_by_py']=c.sql("""select py, count(*) n_all_jp, sum((nopub and not u)::int) nopub_pat, sum((nopub and not u and lag>=2)::int) nopub_pat_lag2,
   sum((nopub and u)::int) nopub_u from j where py between 2012 and 2025 group by 1 order by 1""").fetchall()
R['pub_month_profile_2019_2023']=c.sql("""select substr(pub_date,1,6) m, count(*) from j where pub_date between '20190101' and '20231231' group by 1 order by 1""").fetchall()
json.dump(R,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'v1_jp_t3exposure.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
for k,v in R.items(): print(k); [print(' ',r) for r in v]
