# -*- coding: utf-8 -*-
r"""C14-T2 verify_0 v2 (재시도 반박자): links.parquet·alt_assign·s5 산출물을 쓰지 않고
_shared/meta 원본에서 직접 (1) 문서·제공 라벨 (2) KR IPC→KSIC 최빈표 (3) 연결(자체 REG / 자체 R8 결함 재현)을 다시 만들어
진단용 PKR-IPC(발명, 출원 2015-19) US/JP/CN/EP 일치율을 계산한다.
 - 문서 집합: c1_assign.py와 같게 meta ∩ rec_conc(inner).
 - 발명 필터: CN 길이 12/14 & 5번째 '1'/'8', JP 길이 8/17, US·EP 전체.
 - IPC 표: KR(appl_year<=2019) meta.ksic, 키(sg/mg/sc)별 합 >=3, 최빈(동률은 int 코드 작은 것; row_number).
 - SELF_REG: family 토큰 KR(10|20)+7자리 → meta KR rgst_no 앞 9자리 (US/JP/EP). CN은 R8 파일 CN 행(정상 연결).
 - SELF_R8: family 토큰 KR(10|20)+11자리를 KR 출원번호로 읽음(결함 재현) + R8 파일 CN 행.
 - 상속 = 연결된 KR 문서 meta.ksic 최빈(같은 동률 규칙). 없으면 sg→mg→sc 표.
읽기만. 쓰기: verify_0/v2_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb
V = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C14/C14-T2/verify_0"
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
META = MC + "/_shared/meta/*.parquet"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
R8F = MC + "/_ideas_R8/cache/fam_link.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
os.makedirs(V + "/_tmp2", exist_ok=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{V}/_tmp2'")
O = {}
con.execute(f"""create table dd as select m.cc, m.appl_no, m.appl_year, m.rgst_no, m.ksic, m.family, r.sg
  from read_parquet('{META}') m join read_parquet('{RC}') r on r.cc=m.cc and r.appl_no=m.appl_no""")
con.execute("""create table f as select cc, appl_no, ksic, family, sg, split_part(sg,'/',1) mg, left(sg,4) sc from dd
  where cc in ('US','JP','EP','CN') and appl_year between 2015 and 2019 and
  (case when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') when cc='JP' then length(appl_no) in (8,17) else true end)""")
O["n"] = dict(con.sql("select cc, count(*) from f group by 1 order by 1").fetchall()); log(O["n"])
con.execute("create table kr as select appl_no, appl_year, rgst_no, ksic, sg, split_part(sg,'/',1) mg, left(sg,4) sc from dd where cc='KR'")
O["kr_n"] = con.sql("select count(*), count(distinct appl_no), avg((ksic is null)::int) from kr").fetchone()
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tab_{lvl} as with g as (select {lvl} key_, ksic k, count(*) n from kr where appl_year<=2019 and ksic is not null
      and {lvl} is not null and {lvl}<>'' group by 1,2), w as (select key_ from g group by 1 having sum(n)>=3),
      r as (select g.*, row_number() over (partition by key_ order by n desc, cast(k as int) asc) rn from g join w using(key_))
      select key_, k from r where rn=1""")
con.execute("""create table ftab as select f.cc, f.appl_no, f.ksic, coalesce(a.k, b.k, c.k) k from f
  left join tab_sg a on a.key_=f.sg left join tab_mg b on b.key_=f.mg left join tab_sc c on c.key_=f.sc""")
con.execute("create table tok as select distinct cc, appl_no, t from (select cc, appl_no, unnest(string_split(family,'|')) t from f where family like '%KR%') where t like 'KR%'")
con.execute("create table krr as select appl_no kr_appl_no, substr(rgst_no,1,9) r9 from kr where rgst_no is not null and length(rgst_no)>=9")
con.execute(f"create table cnr8 as select distinct cc, appl_no, kr_appl_no from read_parquet('{R8F}') where cc='CN'")
con.execute("""create table l_reg as select distinct t.cc, t.appl_no, k.kr_appl_no from tok t join krr k on k.r9=substr(t.t,3)
   where regexp_full_match(t.t,'KR(10|20)[0-9]{7}') and t.cc in ('US','JP','EP') union select * from cnr8""")
con.execute("""create table l_r8 as select distinct t.cc, t.appl_no, k.appl_no kr_appl_no from tok t join kr k on k.appl_no=substr(t.t,3)
   where regexp_full_match(t.t,'KR(10|20)[0-9]{11}') and t.cc in ('US','JP','EP') union select * from cnr8""")
O["links_in_f"] = {nm: con.sql(f"select cc, count(*), count(distinct appl_no) from {nm} l semi join f using(cc,appl_no) group by 1 order by 1").fetchall() for nm in ("l_reg", "l_r8")}
log(O["links_in_f"])
def run(link):
    if link is None:
        q = "select cc, count(*), avg(coalesce((k=ksic)::int,0)), 0.0, null from ftab group by 1 order by 1"
    else:
        con.execute(f"""create or replace table fam as with g as (select l.cc, l.appl_no, kr.ksic k, count(*) n from {link} l join kr on kr.appl_no=l.kr_appl_no
          where kr.ksic is not null group by 1,2,3), r as (select *, row_number() over (partition by cc, appl_no order by n desc, cast(k as int)) rn from g)
          select cc, appl_no, k from r where rn=1""")
        q = """select t.cc, count(*), avg(coalesce((coalesce(fm.k, t.k)=t.ksic)::int,0)), avg((fm.k is not null)::int),
               avg(case when fm.k is not null then (fm.k=t.ksic)::int end) from ftab t left join fam fm using(cc,appl_no) group by 1 order by 1"""
    return {c: dict(n=n, pkr5=round(a * 100, 3), famshare=round(s * 100, 2), famagree=None if fa is None else round(fa * 100, 1)) for c, n, a, s, fa in con.sql(q).fetchall()}
O["NONE"] = run(None); log("NONE", O["NONE"])
O["SELF_R8"] = run("l_r8"); log("SELF_R8", O["SELF_R8"])
O["SELF_REG"] = run("l_reg"); log("SELF_REG", O["SELF_REG"])
# 민감도: 동률 규칙을 큰 코드 우선으로 바꾼 REG
con.execute("""create or replace table fam as with g as (select l.cc, l.appl_no, kr.ksic k, count(*) n from l_reg l join kr on kr.appl_no=l.kr_appl_no
   where kr.ksic is not null group by 1,2,3), r as (select *, row_number() over (partition by cc, appl_no order by n desc, cast(k as int) desc) rn from g)
   select cc, appl_no, k from r where rn=1""")
O["SELF_REG_tie_desc_famonly"] = con.sql("select t.cc, round(100*avg(coalesce((coalesce(fm.k, t.k)=t.ksic)::int,0)),3) from ftab t left join fam fm using(cc,appl_no) group by 1 order by 1").fetchall()
log("tie desc", O["SELF_REG_tie_desc_famonly"])
json.dump(O, open(V + "/v2_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
