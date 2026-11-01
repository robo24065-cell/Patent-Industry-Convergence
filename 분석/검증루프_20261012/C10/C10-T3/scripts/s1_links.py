# -*- coding: utf-8 -*-
r"""C10-T3 s1: 해외→KR 패밀리 연결 3종(L_R8 결함, L_REG 등록번호, L_PUB 공개번호 토큰) + 품질·선택 편향 점검.
읽기만: _shared/meta, _ideas_R8/cache/fam_link, label_origin_R10/cache/fam_reg, 공모전데이터/작업데이터/kr_patents.parquet
쓰기: C10-T3/cache/links.parquet, res/s1_links.json, res/s1_bias_*.csv"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C10/C10-T3"
META = MC + "/_shared/meta/*.parquet"
R8 = MC + "/_ideas_R8/cache/fam_link.parquet"
REG = MC + "/_shared_report/label_origin_R10/cache/fam_reg.parquet"
KRP = BASE + "/공모전데이터/작업데이터/kr_patents.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
R = {}
con.execute(f"create table krm as select appl_no kr_appl_no, ksic kk from read_parquet('{META}') where cc='KR'")
con.execute(f"create table krpub as select 출원번호 kr_appl_no, trim(공개번호) pb from read_parquet('{KRP}') where trim(공개번호)<>''")
R["kr_pub_unique"] = con.sql("select count(*), count(distinct pb), count(distinct kr_appl_no) from krpub").fetchone()
# 해외 문서 family 토큰
con.execute(f"""create table tok as select distinct cc, appl_no, t from (select cc, appl_no, unnest(string_split(family,'|')) t
               from read_parquet('{META}') where cc<>'KR' and family like '%KR%') where t like 'KR%'""")
con.execute("""create table tk as select cc, appl_no, t,
  case when regexp_full_match(t,'KR(10|20)[0-9]{11}') then 'P13' when regexp_full_match(t,'KR(19|20)[0-9]{9}') then 'P11'
       when regexp_full_match(t,'KR(10|20)[0-9]{7}') then 'R9' else 'other' end typ,
  case when regexp_full_match(t,'KR(10|20)[0-9]{11}') then substr(t,3) when regexp_full_match(t,'KR(19|20)[0-9]{9}') then '10'||substr(t,3) end pub13
  from tok""")
R["token_types"] = con.sql("select cc, typ, count(*) from tk group by 1,2 order by 1,2").fetchall()
# P13 토큰이 공개번호/출원번호 중 어디에 맞나(국가별)
R["p13_match"] = con.sql("""select cc, count(*) n, sum((pub13 in (select pb from krpub))::int) as pub_hit,
    sum((pub13 in (select kr_appl_no from krm))::int) as appl_hit,
    sum(((pub13 in (select pb from krpub)) and (pub13 in (select kr_appl_no from krm)))::int) as both_hit
    from tk where typ='P13' group by 1 order by 1""").fetchall()
log("tokens", R["token_types"], R["p13_match"])
con.execute(f"create table l_r8 as select distinct cc, appl_no, kr_appl_no from read_parquet('{R8}')")
con.execute(f"create table l_reg0 as select distinct cc, appl_no, kr_appl_no from read_parquet('{REG}')")
con.execute("create table l_pub0 as select distinct tk.cc, tk.appl_no, p.kr_appl_no from tk join krpub p on p.pb=tk.pub13 where tk.typ in ('P13','P11')")
con.execute("create table l_reg as select * from l_reg0 union select * from l_r8 where cc='CN'")
# CN family 토큰(P13)은 KR 출원번호(우선권)다. 공개번호로 읽으면 우연 충돌(아래 cn_pub_only_agree)이라 CN은 L_R8 CN 행만 쓴다(PREREG 변경 기록 1).
R["cn_pub_only_agree"] = con.sql("""select count(*), avg((f.ksic=k.kk)::int) from l_pub0 l join read_parquet('""" + META + """') f using(cc, appl_no)
    join krm k using(kr_appl_no) where l.cc='CN' and (l.cc, l.appl_no, l.kr_appl_no) not in (select (cc, appl_no, kr_appl_no) from l_r8)""").fetchone()
con.execute("create table l_pub as select * from l_pub0 where cc<>'CN' union select * from l_r8 where cc='CN'")
con.execute("create table l_uni as select * from l_reg union select * from l_pub")
con.execute(f"create table fo as select cc, appl_no, appl_year, ksic fk, split_part(coalesce(prio_cc,''),'|',1)='KR' kr_origin, family like '%KR%' famkr from read_parquet('{META}') where cc<>'KR'")
q = {}
for nm in ("l_r8", "l_reg", "l_pub", "l_uni"):
    q[nm] = con.sql(f"""select l.cc, count(*) pairs, count(distinct l.appl_no) fdocs, count(distinct l.kr_appl_no) krdocs,
        avg((f.fk=k.kk)::int) agree_pair from {nm} l join fo f using(cc, appl_no) join krm k using(kr_appl_no) group by rollup(l.cc) order by 1""").df()
    q[nm].insert(0, "link", nm)
Q = pd.concat(q.values()); Q.to_csv(W + "/res/s1_link_quality.csv", index=False, encoding="utf-8-sig"); print(Q.round(4).to_string())
# 연결 상호 일치: 해외 문서 기준, REG와 PUB 둘 다 있는 문서에서 KR 문서 집합 교집합 존재 비율
R["reg_pub_overlap"] = con.sql("""with r as (select distinct cc, appl_no from l_reg0), p as (select distinct cc, appl_no from l_pub0),
   b as (select cc, appl_no from r intersect select cc, appl_no from p)
   select r_.cc, (select count(*) from r where r.cc=r_.cc) n_reg, (select count(*) from p where p.cc=r_.cc) n_pub,
     count(*) n_both,
     avg((exists(select 1 from l_reg0 x join l_pub0 y on x.cc=y.cc and x.appl_no=y.appl_no and x.kr_appl_no=y.kr_appl_no
          where x.cc=r_.cc and x.appl_no=r_.appl_no))::int) share_same_kr
   from b r_ group by 1 order by 1""").fetchall()
log("overlap", R["reg_pub_overlap"])
# R8 결함 연결이 L_REG/L_PUB 쌍과 같은 비율(같은 해외 문서→같은 KR 문서)
R["r8_pairs_in_correct"] = con.sql("""select cc, count(*), avg(((cc, appl_no, kr_appl_no) in (select (cc, appl_no, kr_appl_no) from l_uni))::int) from l_r8 group by 1 order by 1""").fetchall()
# 선택 편향: 연결된 해외 문서 vs 안 된 문서 (발명특허 판정 없이 전체, 출원 2015–19)
con.execute("create table lk as select cc, appl_no, max(src) src from (select cc, appl_no, 1 src from l_reg union all select cc, appl_no, 2 src from l_pub) group by 1,2")
con.execute("""create table fb as select f.*, (r.appl_no is not null) linked_reg, (p.appl_no is not null) linked_pub
   from fo f left join (select distinct cc, appl_no from l_reg) r using(cc, appl_no) left join (select distinct cc, appl_no from l_pub) p using(cc, appl_no)""")
b1 = con.sql("""select cc, count(*) n, avg(linked_reg::int) share_reg, avg(linked_pub::int) share_pub, avg(famkr::int) share_famkr,
   avg(case when linked_reg then kr_origin::int end) prioKR_linked_reg, avg(case when not linked_reg then kr_origin::int end) prioKR_unlinked_reg,
   avg(case when linked_pub then kr_origin::int end) prioKR_linked_pub, avg(case when not linked_pub then kr_origin::int end) prioKR_unlinked_pub
   from fb where appl_year between 2015 and 2019 group by 1 order by 1""").df()
b2 = con.sql("""select cc, appl_year, count(*) n, avg(linked_reg::int) share_reg, avg(linked_pub::int) share_pub from fb
   where appl_year between 2005 and 2024 group by 1,2 order by 1,2""").df()
# KSIC 분포 TVD(연결 vs 미연결), 2015–19
def tvd(col):
    d = con.sql(f"""select cc, fk, sum({col}::int) a, sum((not {col})::int) b from fb where appl_year between 2015 and 2019 group by 1,2""").df()
    out = []
    for cc, g in d.groupby("cc"):
        pa = g.a / max(g.a.sum(), 1); pb = g.b / max(g.b.sum(), 1)
        out.append(dict(cc=cc, link=col, tvd=round(float((pa - pb).abs().sum() / 2), 4), n_linked=int(g.a.sum())))
    return out
R["ksic_tvd_linked_vs_unlinked_1519"] = tvd("linked_reg") + tvd("linked_pub")
b1.to_csv(W + "/res/s1_bias_country.csv", index=False, encoding="utf-8-sig"); b2.to_csv(W + "/res/s1_bias_year.csv", index=False, encoding="utf-8-sig")
print(b1.round(4).to_string()); log("tvd", R["ksic_tvd_linked_vs_unlinked_1519"])
con.execute(f"""copy (select 'R8' link, * from l_r8 union all select 'REG', * from l_reg union all select 'PUB', * from l_pub)
   to '{W}/cache/links.parquet' (format parquet)""")
R["link_rows"] = con.sql(f"select link, cc, count(*) from '{W}/cache/links.parquet' group by 1,2 order by 1,2").fetchall()
json.dump(R, open(W + "/res/s1_links.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done", R["link_rows"])
