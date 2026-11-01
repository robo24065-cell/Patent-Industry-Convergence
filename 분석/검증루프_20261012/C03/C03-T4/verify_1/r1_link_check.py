# verify_1/r1: C03-A 부산물 반박 검증 — 공개번호 토큰 연결 정확도를 KSIC 외 독립 지표(우선권일·출원인)로 재측정, 무작위·출원번호 해석과 비교
import sys, json, time, duckdb
sys.stdout.reconfigure(encoding='utf-8')
T=time.time()
B='C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
W=B+'/분석/검증루프_20261012/C03/C03-T4'
META=B+'/분석/모델경쟁_20260927/_shared/meta/*.parquet'
KRP=B+'/공모전데이터/작업데이터/kr_patents.parquet'
DOCS=B+'/분석/검증루프_20261012/C02/C02-T4/cache/docs.parquet'
d=duckdb.connect(); d.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
cols=[r[0] for r in d.sql(f"describe select * from read_parquet('{KRP}')").fetchall()]
A,AD,RG,PB,PD,AP,KS=cols[0],cols[1],cols[3],cols[5],cols[6],cols[8],cols[17]
d.execute(f'create table kr as select "{A}" kr_appl, cast("{AD}" as varchar) kr_ad, trim("{PB}") pb, substr(trim("{RG}"),1,9) r9, cast("{KS}" as varchar) kk, "{AP}" kap from read_parquet(\'{KRP}\')')
out={}
out['kr_rows']=d.sql('select count(*), count(distinct kr_appl), count(distinct kr_appl||kk) from kr').fetchone()
out['kr_pb_dup']=d.sql("select count(*) from (select pb from kr where pb<>'' group by pb having count(distinct kr_appl)>1)").fetchone()[0]
out['kr_pb_eq_appl_other']=d.sql("select count(distinct a.pb) from kr a join kr b on a.pb=b.kr_appl where a.pb<>''").fetchone()[0]
d.execute(f"""create table fo as select cc, appl_no, pub_date, prio_cc, prio_date, family, applicant, cast(ksic as varchar) fk from read_parquet('{META}') where cc in ('US','EP') and pub_date<'20250101' and family like '%KR%'""")
d.execute("""create table tk as select distinct cc, appl_no, pub_date, prio_date, applicant, fk, t,
  case when regexp_full_match(t,'KR(10|20)[0-9]{11}') then 'P13' when regexp_full_match(t,'KR(19|20)[0-9]{9}') then 'P11'
       when regexp_full_match(t,'KR(10|20)[0-9]{7}') then 'R9' end typ,
  case when regexp_full_match(t,'KR(10|20)[0-9]{11}') then substr(t,3) when regexp_full_match(t,'KR(19|20)[0-9]{9}') then '10'||substr(t,3) end n13,
  case when regexp_full_match(t,'KR(10|20)[0-9]{7}') then substr(t,3) end r9
  from (select *, unnest(string_split(family,'|')) t from fo) where t like 'KR%'""")
out['tok_by_typ']=d.sql('select typ, count(*) from tk group by 1 order by 1').fetchall()
# 연결 방식별 쌍
d.execute("""create table lk as
 select tk.*, kr.kr_appl, kr.kr_ad, kr.kk, kr.kap, 'PUB' meth from tk join kr on tk.n13=kr.pb where tk.typ in ('P13','P11')
 union all select tk.*, kr.kr_appl, kr.kr_ad, kr.kk, kr.kap, 'APPL' meth from tk join kr on tk.n13=kr.kr_appl where tk.typ in ('P13','P11')
 union all select tk.*, kr.kr_appl, kr.kr_ad, kr.kk, kr.kap, 'REG' meth from tk join kr on tk.r9=kr.r9 where tk.typ='R9' and kr.r9<>''""")
# 무작위 기준: PUB 연결의 KR 쪽을 섞음
d.execute("""create table rnd as select a.*, b.kr_ad kr_ad2, b.kk kk2 from
 (select *, row_number() over (order by hash(cc||appl_no||kr_appl)) rn from lk where meth='PUB') a
 join (select kr_ad, kk, row_number() over (order by hash(kr_appl||'x')) rn from lk where meth='PUB') b using(rn)""")
q="""select {g}, count(*) n, count(distinct cc||appl_no) nf,
 round(avg((fk={kk})::int),4) ksic_agree,
 round(avg(case when prio_date is not null then (list_contains(string_split(prio_date,'|'), {ad}))::int end),4) prio_date_match,
 round(avg(case when prio_date is not null then (abs(datediff('day', strptime({ad},'%Y%m%d'), strptime(list_min(string_split(prio_date,'|')),'%Y%m%d')))<=400)::int end),4) prio_within400d
 from {t} {w} group by 1 order by 1"""
out['by_meth']=d.sql(q.format(g='meth',kk='kk',ad='kr_ad',t='lk',w='')).fetchall()
out['by_meth_typ']=d.sql(q.format(g="meth||'-'||typ",kk='kk',ad='kr_ad',t='lk',w='')).fetchall()
out['random_PUB_shuffled']=d.sql(q.format(g="'RND'",kk='kk2',ad='kr_ad2',t='rnd',w='')).fetchall()
# P13 토큰이 PUB·APPL 둘 다 맞는 경우: 어느 해석이 맞나
d.execute("""create table both_ as select p.cc,p.appl_no,p.t,p.fk,p.prio_date, p.kr_ad ad_pub, p.kk kk_pub, a.kr_ad ad_appl, a.kk kk_appl
 from lk p join lk a on p.cc=a.cc and p.appl_no=a.appl_no and p.t=a.t and p.meth='PUB' and a.meth='APPL'""")
out['both_PUB_and_APPL']=d.sql("""select count(*), round(avg((fk=kk_pub)::int),4), round(avg((fk=kk_appl)::int),4),
 round(avg(case when prio_date is not null then list_contains(string_split(prio_date,'|'),ad_pub)::int end),4),
 round(avg(case when prio_date is not null then list_contains(string_split(prio_date,'|'),ad_appl)::int end),4) from both_""").fetchone()
# APPL만 맞고 PUB는 안 맞는 토큰
out['APPL_only']=d.sql("""select count(*), round(avg((fk=kk)::int),4), round(avg(case when prio_date is not null then list_contains(string_split(prio_date,'|'),kr_ad)::int end),4)
 from lk a where meth='APPL' and not exists (select 1 from lk p where p.meth='PUB' and p.cc=a.cc and p.appl_no=a.appl_no and p.t=a.t)""").fetchone()
# 원 연구의 'KSIC 일치' 재현(법인 문서 한정, docs ksic)
d.execute(f"create table dc as select appl_no, any_value(ksic) kk2 from read_parquet('{DOCS}') where substr(cd,1,1)='1' group by 1")
out['orig_repro_corp']=d.sql("""select meth, count(*), round(avg((fk=kk2)::int),4) from (select distinct cc,appl_no,pub_date,fk,kr_appl,meth from lk) x join dc on x.kr_appl=dc.appl_no where meth in ('PUB','REG') group by 1 order by 1""").fetchall()
# 미등록 KR 문서 도달
out['PUB_kr_unregistered_share']=d.sql("""select round(avg((r9 is null or r9='')::int),4), count(distinct kr_appl) from (select distinct lk.kr_appl, kr.r9 from lk join kr using(kr_appl) where meth='PUB')""").fetchone()
out['elapsed']=round(time.time()-T,1)
json.dump(out, open(W+'/verify_1/r1_out.json','w',encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str, indent=1))
