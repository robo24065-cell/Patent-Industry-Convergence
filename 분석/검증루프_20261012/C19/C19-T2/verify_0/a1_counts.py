# verify_0 a1: 독립 재계산(읽기만). 정규화를 정수 산술로 구현, applNo 단위 그룹.
import duckdb, sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
BASE=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
T4=(BASE+r'\T4_특허포기예측\cache').replace('\\','/')
V=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C19\C19-T2\verify_0'
tmp=(V+r'\tmp').replace('\\','/'); os.makedirs(tmp,exist_ok=True)
con=duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set temp_directory='{tmp}'")
R={}
con.execute(f"""create table tk as select applNo, cc tcc, yr, tok, substr(tok,1,2) c2, substr(tok,3) d from read_parquet('{T4}/s2_fam_tokens.parquet')""")
R['rows']=con.execute("select count(*), count(distinct applNo), sum((not regexp_full_match(d,'[0-9]+'))::int), sum((tcc<>c2)::int) from tk").fetchone()
# registered population (pid): rgstDate non-empty & rgst_no starts '10'
con.execute(f"""create table reg as select distinct applNo from read_parquet('{T4}/patent.parquet') where rgstDate is not null and trim(rgstDate)<>'' and substr(rgst_no,1,2)='10'""")
con.execute("""create table t2 as select tk.*, (r.applNo is not null) isreg, length(d) L,
   c2 || '#' || cast(cast(d as hugeint) as varchar) k1,
   case when c2='US' and length(d)=11 and substr(d,5,1)='0' then 'US#P'||substr(d,1,4)||'#'||cast(cast(substr(d,6) as bigint) as varchar)
        when c2='US' and length(d)=10 then 'US#P'||substr(d,1,4)||'#'||cast(cast(substr(d,5) as bigint) as varchar)
        when c2='JP' and length(d)=10 then 'JP#P'||substr(d,1,4)||'#'||cast(cast(substr(d,5) as bigint) as varchar)
        when c2='JP' and length(d)=8 and substr(d,1,1)<>'0' then 'JP#P'||cast(1988+cast(substr(d,1,2) as int) as varchar)||'#'||cast(cast(substr(d,3) as bigint) as varchar)
        else c2 || '#' || cast(cast(d as hugeint) as varchar) end kx
   from tk left join reg r using(applNo) where regexp_full_match(d,'[0-9]+')""")
R['t2_rows']=con.execute("select count(*), sum(isreg::int) from t2").fetchone()
# exact duplicates within applNo?
R['exact_dup']=con.execute("select count(*)-count(distinct (applNo,tok)) from t2").fetchone()[0]
for sub,cond in [('all','true'),('reg','isreg')]:
    for k in ['k1','kx']:
        q=f"""select count(*) grp, sum(n-1) dup, max(n) mx, sum((ny>1)::int) yrdiff from (select applNo,{k},count(*) n,count(distinct yr) ny from t2 where {cond} group by 1,2) where n>1"""
        R[f'{sub}_{k}']=con.execute(q).fetchone()
    n=con.execute(f"select count(*) from t2 where {cond}").fetchone()[0]
    R[f'{sub}_ntok']=n
    R[f'{sub}_share_k1']=R[f'{sub}_k1'][1]/n; R[f'{sub}_share_kx']=R[f'{sub}_kx'][1]/n
# by cc for reg: F1 pairs and extra pairs (kx dup minus k1 dup)
R['reg_k1_by_cc']=con.execute("select substr(k1,1,2), sum(n-1) from (select applNo,k1,count(*) n from t2 where isreg group by 1,2) where n>1 group by 1 order by 2 desc").fetchall()
R['reg_kx_by_cc']=con.execute("select substr(kx,1,2), sum(n-1) from (select applNo,kx,count(*) n from t2 where isreg group by 1,2) where n>1 group by 1 order by 2 desc").fetchall()
# forms of kx-extra groups: pairs whose k1 differ but kx same
R['reg_extra_forms']=con.execute("""select c2, Ls, count(*) from (select applNo,kx,any_value(c2) c2, string_agg(cast(L as varchar),'/' order by L) Ls, count(distinct k1) nk from t2 where isreg group by 1,2 having count(*)>1) where nk>1 group by 1,2 order by 3 desc""").fetchall()
R['reg_k1_forms']=con.execute("""select c2, Ls, count(*) from (select applNo,k1,any_value(c2) c2, string_agg(cast(L as varchar),'/' order by L) Ls from t2 where isreg group by 1,2 having count(*)>1) group by 1,2 order by 3 desc""").fetchall()
json.dump(R,open(V+r'\a1_counts.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=str)
print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
con.execute(f"copy (select applNo,isreg,yr,c2,d,L,k1,kx from t2) to '{tmp}/t2.parquet' (format parquet)")
