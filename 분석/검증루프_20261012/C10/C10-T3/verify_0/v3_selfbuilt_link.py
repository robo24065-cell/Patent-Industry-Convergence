# -*- coding: utf-8 -*-
r"""C10-T3 verify_0 v3 (2nd verifier): fam_reg.parquet·links.parquet·작업자 산출물을 쓰지 않고
meta family 토큰에서 직접 연결을 다시 만들어 FREE-IPC 해외 재현율(발명, 출원 2015-19)을 재계산.
 - SELF_REG : R9 토큰 KR(10|20)+7자리 -> meta KR rgst_no 앞 9자리
 - SELF_PUB : P13 그대로 / P11 -> '10'+숫자 -> kr_patents 공개번호 (US/EP/JP) ; CN은 R8 CN 행
 - SELF_R8  : P13 토큰을 KR 출원번호로 읽음(결함 재현) ; R8 fam_link 와 행 일치 비교
 - pure     : 상속 없이 IPC 표
읽기만. 쓰기: verify_0/v3_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb
V = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
BASE = V.split("/분석/")[0]
MC = BASE + "/분석/모델경쟁_20260927"
META = MC + "/_shared/meta/*.parquet"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
R8 = MC + "/_ideas_R8/cache/fam_link.parquet"
REGF = MC + "/_shared_report/label_origin_R10/cache/fam_reg.parquet"
KRP = BASE + "/공모전데이터/작업데이터/kr_patents.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
os.makedirs(V + "/_tmp", exist_ok=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{V}/_tmp'")
O = {}
# foreign invention docs 2015-19 with rec_conc
con.execute(f"""create table f as select m.cc::varchar cc, m.appl_no, m.ksic::varchar ksic, m.family, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc
 from read_parquet('{META}') m join read_parquet('{RC}') r on r.cc=m.cc::varchar and r.appl_no=m.appl_no
 where m.cc::varchar in ('US','JP','EP','CN') and m.appl_year between 2015 and 2019
 and (case when m.cc::varchar='CN' then length(m.appl_no) in (12,14) and substr(m.appl_no,5,1) in ('1','8')
           when m.cc::varchar='JP' then length(m.appl_no) in (8,17) else true end)""")
O["n"] = dict(con.sql("select cc, count(*) from f group by 1").fetchall()); log(O["n"])
con.execute(f"create table kr as select appl_no, appl_year, rgst_no from read_parquet('{META}') where cc::varchar='KR'")
con.execute(f"create table kf as select appl_no, k_free from read_parquet('{KF}') where k_free is not null")
con.execute(f"create table krsg as select k.appl_no, k.appl_year, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc from kr k join read_parquet('{RC}') r on r.cc='KR' and r.appl_no=k.appl_no")
# IPC tables (KR appl<=2019, min 3, mode, tie -> smaller code)
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tab_{lvl} as with g as (select s.{lvl} key_, kf.k_free k, count(*) n from krsg s join kf using(appl_no)
      where s.appl_year<=2019 and s.{lvl} is not null and s.{lvl}<>'' group by 1,2),
      w as (select key_ from g group by 1 having sum(n)>=3),
      r as (select g.*, row_number() over (partition by key_ order by n desc, cast(k as int) asc) rn from g join w using(key_))
      select key_, k from r where rn=1""")
con.execute("""create table ftab as select f.cc, f.appl_no, coalesce(a.k, b.k, c.k) k from f
  left join tab_sg a on a.key_=f.sg left join tab_mg b on b.key_=f.mg left join tab_sc c on c.key_=f.sc""")
# tokens
con.execute("""create table tok as select distinct cc, appl_no, t from (select cc, appl_no, unnest(string_split(family,'|')) t from f where family like '%KR%') where t like 'KR%'""")
con.execute(f"create table krpub as select 출원번호 kr_appl_no, trim(공개번호) pb from read_parquet('{KRP}') where 공개번호 is not null and trim(공개번호)<>''")
con.execute("create table krr as select appl_no kr_appl_no, substr(rgst_no,1,9) r9 from kr where rgst_no is not null and length(rgst_no)>=9")
con.execute("""create table l_selfreg as select distinct t.cc, t.appl_no, k.kr_appl_no from tok t join krr k on k.r9=substr(t.t,3)
   where regexp_full_match(t.t,'KR(10|20)[0-9]{7}') and t.cc in ('US','JP','EP')""")
con.execute("""create table l_selfpub as select distinct t.cc, t.appl_no, p.kr_appl_no from tok t join krpub p on p.pb =
   case when regexp_full_match(t.t,'KR(10|20)[0-9]{11}') then substr(t.t,3) when regexp_full_match(t.t,'KR(19|20)[0-9]{9}') then '10'||substr(t.t,3) end
   where t.cc in ('US','JP','EP')""")
con.execute("""create table l_selfr8 as select distinct t.cc, t.appl_no, k.appl_no kr_appl_no from tok t join kr k on k.appl_no=substr(t.t,3)
   where regexp_full_match(t.t,'KR(10|20)[0-9]{11}')""")
con.execute(f"create table l_r8 as select distinct cc, appl_no, kr_appl_no from read_parquet('{R8}')")
con.execute(f"create table l_reg_file as select distinct cc, appl_no, kr_appl_no from read_parquet('{REGF}')")
con.execute("create table cnr8 as select * from l_r8 where cc='CN'")
# compare self-built vs files (restricted to f docs)
def cmp(a, b):
    return con.sql(f"""with A as (select x.* from {a} x semi join f using(cc,appl_no)), B as (select x.* from {b} x semi join f using(cc,appl_no))
      select (select count(*) from A), (select count(*) from B), (select count(*) from (select * from A intersect select * from B))""").fetchone()
O["cmp_selfreg_vs_famreg"] = cmp("l_selfreg", "(select * from l_reg_file where cc<>'CN')")
O["cmp_selfr8_vs_r8"] = cmp("l_selfr8", "l_r8")
log("cmp", O["cmp_selfreg_vs_famreg"], O["cmp_selfr8_vs_r8"])
# link label quality (pair-level provided KSIC agreement) within f
con.execute(f"create table krk as select appl_no kr_appl_no, ksic::varchar kk from read_parquet('{META}') where cc::varchar='KR'")
for nm in ("l_selfreg", "l_selfpub", "l_selfr8"):
    O["pairagree_" + nm] = con.sql(f"select l.cc, count(*), round(avg((f.ksic=k.kk)::int),4) from {nm} l join f using(cc,appl_no) join krk k using(kr_appl_no) group by 1 order by 1").fetchall()
log({k: v for k, v in O.items() if k.startswith("pairagree")})
def recall(link_sql):
    if link_sql is None:
        q = "select f.cc, avg(coalesce((t.k=f.ksic)::int,0)) from f join ftab t using(cc,appl_no) group by 1 order by 1"
    else:
        con.execute(f"""create or replace table fam as with g as (select l.cc, l.appl_no, kf.k_free k, count(*) n from ({link_sql}) l join kf on kf.appl_no=l.kr_appl_no group by 1,2,3),
          r as (select *, row_number() over (partition by cc, appl_no order by n desc, cast(k as int)) rn from g) select cc, appl_no, k from r where rn=1""")
        q = """select f.cc, avg(coalesce((coalesce(fm.k, t.k)=f.ksic)::int,0)) from f join ftab t using(cc,appl_no) left join fam fm using(cc,appl_no) group by 1 order by 1"""
    return {c: round(v * 100, 3) for c, v in con.sql(q).fetchall()}
O["recall"] = {
    "pure": recall(None),
    "r8_file": recall("select * from l_r8"),
    "self_r8": recall("select * from l_selfr8"),
    "self_reg": recall("select * from l_selfreg union select * from cnr8"),
    "file_reg": recall("select * from l_reg_file union select * from cnr8"),
    "self_pub": recall("select * from l_selfpub union select * from cnr8"),
}
for k, v in O["recall"].items():
    log(k, v, "range", min(v.values()), max(v.values()))
json.dump(O, open(V + "/v3_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
