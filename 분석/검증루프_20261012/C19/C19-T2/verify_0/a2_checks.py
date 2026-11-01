import duckdb, sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
BASE=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
META=(BASE+r'\_shared\meta\*.parquet').replace('\\','/')
V=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C19\C19-T2\verify_0'
tmp=(V+r'\tmp').replace('\\','/')
con=duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set temp_directory='{tmp}'")
R={}
t2=f"'{tmp}/t2.parquet'"
# 1) JP 8자리 비0 시작 토큰의 앞 2자리(연호 연도) 분포, 짝 여부
R['jp8_nz_yy']=con.execute(f"select substr(d,1,2) yy, count(*) from read_parquet({t2}) where c2='JP' and L=8 and substr(d,1,1)<>'0' group by 1 order by 1").fetchall()
R['jp8_0_first3']=con.execute(f"select substr(d,1,2), count(*) from read_parquet({t2}) where c2='JP' and L=8 and substr(d,1,1)='0' group by 1 order by 1").fetchall()
R['jp10_year']=con.execute(f"select substr(d,1,4), count(*) from read_parquet({t2}) where c2='JP' and L=10 group by 1 order by 1").fetchall()
R['us10_year']=con.execute(f"select substr(d,1,4), count(*) from read_parquet({t2}) where c2='US' and L in (10,11) group by 1 order by 1").fetchall()
R['len_by_cc']=con.execute(f"select c2, L, count(*) from read_parquet({t2}) where c2 in ('US','JP','EP','CN','WO','KR','TW','DE','AU') group by 1,2 order by 1,3 desc").fetchall()
# 2) 누락 탐색(넓게): kx 뒤 같은 applNo·cc·yr 안 끝 5자리 같은 쌍 형식
R['resid_last5']=con.execute(f"""with u as (select distinct applNo,c2,yr,L,kx, right(d,5) t5 from read_parquet({t2}) where isreg)
  select c2, string_agg(cast(L as varchar),'/' order by L) f, count(*) from u group by applNo,c2,yr,t5 having count(*)>1""").fetchall() if False else None
R['resid_last5_forms']=con.execute(f"""with u as (select distinct applNo,c2,yr,L,kx, right(d,5) t5 from read_parquet({t2}) where isreg),
  g as (select c2, string_agg(cast(L as varchar),'/' order by L) f from u group by applNo,c2,yr,t5 having count(*)>1)
  select c2,f,count(*) from g group by 1,2 order by 3 desc limit 20""").fetchall()
# 같은 일련번호(끝 6자리)·연도 다른 경우(연도 추정 차)도 확인: applNo·cc·끝6 같고 yr 다른 JP/US 공개
R['resid_last6_anyyr']=con.execute(f"""with u as (select distinct applNo,c2,L,kx, right(d,6) t6 from read_parquet({t2}) where isreg and c2 in ('US','JP','EP') and L>=8),
  g as (select c2, string_agg(cast(L as varchar),'/' order by L) f from u group by applNo,c2,t6 having count(*)>1)
  select c2,f,count(*) from g group by 1,2 order by 3 desc limit 20""").fetchall()
# 3) KR 기록 단위(메타): 독립 구현 — 리스트 대신 unnest 후 그룹
con.execute(f"""create table kr as select appl_no, any_value(family) fam from read_parquet('{META}') where cc='KR' group by appl_no""")
R['kr']=con.execute("select count(*), sum((fam is not null and fam<>'')::int) from kr").fetchone()
con.execute("""create table u as select distinct appl_no, t from (select appl_no, unnest(string_split(fam,'|')) t from kr where fam is not null and fam<>'') where t<>''""")
con.execute(r"""create table u2 as select appl_no, t, substr(t,1,2) c2, regexp_extract(substr(t,3),'^[0-9]*') dg, regexp_full_match(substr(t,3),'[0-9]+') isnum from u""")
con.execute(r"""create table u3 as select *, case when dg='' then t else c2||'#'||cast(cast(dg as hugeint) as varchar) end k1,
  case when dg='' then t
       when c2='US' and length(dg)=11 and substr(dg,5,1)='0' then 'US#P'||substr(dg,1,4)||'#'||cast(cast(substr(dg,6) as bigint) as varchar)
       when c2='US' and length(dg)=10 then 'US#P'||substr(dg,1,4)||'#'||cast(cast(substr(dg,5) as bigint) as varchar)
       when c2='JP' and length(dg)=10 then 'JP#P'||substr(dg,1,4)||'#'||cast(cast(substr(dg,5) as bigint) as varchar)
       when c2='JP' and length(dg)=8 and substr(dg,1,1)<>'0' then 'JP#P'||cast(1988+cast(substr(dg,1,2) as int) as varchar)||'#'||cast(cast(substr(dg,3) as bigint) as varchar)
       else c2||'#'||cast(cast(dg as hugeint) as varchar) end kx from u2""")
# 접미 있는 토큰은 접미 포함 동일성 유지 위해 suffix 붙임
con.execute("create table u4 as select appl_no, t, isnum, case when isnum then k1 else k1||'~'||substr(t,3+length(dg)) end k1, case when isnum then kx else kx||'~'||substr(t,3+length(dg)) end kx from u3")
r=con.execute("""select count(*), avg((n1<n)::int), avg((nx<n)::int), sum(n), sum(n-n1)/sum(n), sum(n-nx)/sum(n), sum(nn) from (select appl_no, count(*) n, count(distinct k1) n1, count(distinct kx) nx, sum((not isnum)::int) nn from u4 group by 1)""").fetchone()
R['kr_records']=r
json.dump(R,open(V+r'\a2_checks.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=str)
print(json.dumps(R,ensure_ascii=False,default=str))
