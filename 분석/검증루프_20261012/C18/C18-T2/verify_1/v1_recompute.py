# C18-T2 verify_1: 타국 같은 유형 결함 주장 독립 재계산 (원자료 읽기만, 쓰기는 verify_1만)
import duckdb, json, sys, os
sys.stdout.reconfigure(encoding='utf-8')
META = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
OUT = os.path.dirname(os.path.abspath(__file__))
con = duckdb.connect()
con.execute("set memory_limit='3GB'; set threads=2")
con.execute(f"""create table m as select cc, appl_no, appl_date, appl_year, rgst_no, rgst_date, pub_date, title, main_ipc, ksic, filename
  from read_parquet('{META}', filename=true) where cc in ('EP','JP')""")
R = {}
R['rows'] = con.sql("select cc, count(*), count(distinct appl_no) from m group by 1 order by 1").fetchall()
# EP 번호 형식
R['ep_len'] = con.sql("select length(appl_no), count(*), min(appl_no), max(appl_no) from m where cc='EP' group by 1 order by 1").fetchall()
# 방법 2: 다른 키 정의 -- 점 앞 본번만 정수로
con.execute(r"""create table ep as select *, try_cast(regexp_extract(appl_no,'^0*([0-9]+)\.',1) as bigint) k, regexp_extract(appl_no,'\.([0-9X])$',1) chk from m where cc='EP'""")
R['ep_k_null'] = con.sql("select count(*) from ep where k is null").fetchone()[0]
con.execute("""create table ep_pr as select a.k, a.appl_no n1, b.appl_no n2, a.chk c1, b.chk c2, a.title t1, b.title t2, a.appl_date d1, b.appl_date d2,
  a.appl_year y1, a.ksic s1, b.ksic s2, a.main_ipc i1, b.main_ipc i2, a.pub_date p1, b.pub_date p2, a.rgst_date r1, b.rgst_date r2
  from ep a join ep b on a.k=b.k and a.appl_no<b.appl_no""")
R['ep_pairs'] = con.sql("""select count(*), count(distinct k), avg((c1=c2)::int) chk_eq,
  avg((lower(trim(t1))=lower(trim(t2)))::int) title_eq, avg((d1=d2)::int) date_eq, avg((s1=s2)::int) ksic_eq, avg((i1=i2)::int) ipc_eq,
  avg((coalesce(p1,'')=coalesce(p2,''))::int) pub_eq, avg((coalesce(r1,'')=coalesce(r2,''))::int) rgst_eq from ep_pr""").fetchall()
R['ep_pair_lens'] = con.sql("select length(n1), length(n2), count(*) from ep_pr group by all order by all").fetchall()
R['ep_pairs_by_year'] = con.sql("""with p as (select y1 y, count(*) n from ep_pr group by 1), t as (select appl_year y, count(*) nrow from ep group by 1)
  select t.y, coalesce(n,0), nrow, round(coalesce(n,0)/nrow*100,2) from t left join p using(y) where t.y between 1998 and 2025 order by 1""").fetchall()
R['ep_rows_per_key_max'] = con.sql("select max(c) from (select k, count(*) c from ep group by 1)").fetchone()[0]
R['ep_examples'] = con.sql("select n1,n2,left(t1,40),left(t2,40),d1,d2,s1,s2 from ep_pr using sample 6 rows (reservoir, 7)").fetchall()
# 같은 파일(KSIC)에 두 형식이 동시에? 짧은 형식 행만 따로 존재하는 경우
R['ep_short_unpaired'] = con.sql("select count(*) from ep a where length(appl_no)<10 and not exists (select 1 from ep b where b.k=a.k and length(b.appl_no)=10)").fetchone()[0]
# JP 충돌
con.execute(r"""create table jp as select *, try_cast(ltrim(regexp_replace(appl_no,'[^0-9]','','g'),'0') as bigint) k, appl_no like '%U' u from m where cc='JP'""")
R['jp_suffix'] = con.sql("select u, count(*) from jp group by 1").fetchall()
R['jp_pairs'] = con.sql("""select a.u ua, b.u ub, count(*), avg((lower(a.title)=lower(b.title))::int), avg((a.appl_date=b.appl_date)::int)
   from jp a join jp b on a.k=b.k and a.appl_no<b.appl_no group by all""").fetchall()
R['jp_same_type_pairs_examples'] = con.sql("""select a.appl_no,b.appl_no,left(a.title,30),left(b.title,30),a.appl_date,b.appl_date from jp a join jp b on a.k=b.k and a.appl_no<b.appl_no and a.u=b.u limit 10""").fetchall()
# 등록 연월 프로필 (JP, EP) + 공개일 동반 여부
con.execute("create table ym as select cc, substr(rgst_date,1,6) ym, count(*) n, sum((pub_date is not null and pub_date<>'')::int) n_pub, sum((rgst_no is not null and rgst_no<>'')::int) n_no from m where rgst_date is not null and rgst_date<>'' group by all")
ym = con.sql("select * from ym where ym between '201801' and '202512' order by cc, ym").fetchall()
with open(os.path.join(OUT, 'v1_rgst_ym.csv'), 'w', encoding='utf-8') as f:
    f.write('cc,ym,n,n_with_pub,n_with_rgstno\n')
    for r in ym: f.write(','.join(map(str, r)) + '\n')
R['jp_both_by_rgst_year'] = con.sql("""select substr(rgst_date,1,4) y, count(*) n, sum((pub_date is not null and pub_date<>'')::int) both_ from m where cc='JP' and rgst_date<>'' and rgst_date is not null
   and substr(rgst_date,1,4) between '2014' and '2025' group by 1 order by 1""").fetchall()
R['jp_nopub_rows'] = con.sql("select count(*), sum((rgst_date is not null and rgst_date<>'')::int) from m where cc='JP' and (pub_date is null or pub_date='')").fetchone()
# rgst_no 있는데 rgst_date 없는 경우(EP 공백이 날짜만 빠진 건지)
R['rgstno_without_date'] = con.sql("select cc, count(*) from m where (rgst_no is not null and rgst_no<>'') and (rgst_date is null or rgst_date='') group by 1").fetchall()
R['rgstdate_without_no'] = con.sql("select cc, count(*) from m where (rgst_no is null or rgst_no='') and (rgst_date is not null and rgst_date<>'') group by 1").fetchall()
R['pub_null'] = con.sql("select cc, sum((pub_date is null or pub_date='')::int), count(*) from m group by 1").fetchall()
# EP 등록 비율 출원연도별 (공백이 출원 코호트에 남기는 흔적)
R['ep_grant_share_by_ay'] = con.sql("select appl_year, count(*), round(avg((rgst_date is not null and rgst_date<>'')::int),3) from m where cc='EP' and appl_year between 2010 and 2022 group by 1 order by 1").fetchall()
json.dump(R, open(os.path.join(OUT, 'v1_recompute.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, default=str))
