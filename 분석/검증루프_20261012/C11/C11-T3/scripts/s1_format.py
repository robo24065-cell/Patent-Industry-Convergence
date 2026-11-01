# -*- coding: utf-8 -*-
"""s1 (1) 번호 형식: US 토큰 형식 분포, 코드 키(정규식) 적용, KPSS/RoS/제공 US 출원번호 대조, 오독 집계"""
import json
from common import *
c = con()
TOK = f"read_parquet('{W}/cache/tok/*.parquet')"
c.execute(f"""create table tk as select cc, appl_no, ksic, t,
 case when regexp_full_match(t,'US[0-9]{{7}}') then 'R7'
      when regexp_full_match(t,'US0[0-9]{{7}}') then 'R8z'
      when regexp_full_match(t,'US1[0-2][0-9]{{6}}') then 'R8n'
      when regexp_full_match(t,'US[0-9]{{8}}') then 'R8x'
      when regexp_full_match(t,'US(19|20)[0-9]{{8}}') then 'P10'
      when regexp_full_match(t,'US(19|20)[0-9]{{9}}') then 'P11'
      when regexp_full_match(t,'USD[0-9]+') then 'D'
      when regexp_full_match(t,'USRE[0-9]+') then 'RE'
      when regexp_full_match(t,'USPP[0-9]+') then 'PP'
      when regexp_full_match(t,'USH[0-9]+') then 'H'
      when regexp_full_match(t,'US[0-9]+') then 'Nother'||length(t)-2
      else 'other' end typ,
 regexp_full_match(t,'US[0-9]{{7,8}}') code_key,
 try_cast(substr(t,3) as bigint) num
 from {TOK}""")
R = {}
d = c.sql("select typ, count(*) n_tok, count(distinct cc||appl_no) n_rec, sum(code_key::int) n_codekey, min(num) mn, max(num) mx from tk group by 1 order by 2 desc").df()
print(d.to_string()); R['typ'] = d.to_dict('records')
d2 = c.sql("select cc, typ, count(*) n from tk group by 1,2 order by 1,2").df(); d2.to_csv(W + "/res/s1_tok_by_cc.csv", index=False, encoding='utf-8-sig')
# 기타 샘플
R['other_samples'] = c.sql("select typ, list(t)[1:8] from (select typ, t from tk where typ in ('other','R8x') or typ like 'Nother%' using sample 2000 rows) group by 1").fetchall()
print(R['other_samples'])
# 키 집합
c.execute(f"create table kp as select patent_num::bigint pn, issue_date::date idt, filing_date::date fdt from read_parquet('{EXT}/quality_kpss_2025.parquet') where patent_num is not null")
R['kpss'] = c.sql("select count(*), count(distinct pn), min(pn), max(pn), min(idt), max(idt) from kp").fetchone()
c.execute(f"create table ros as select distinct pn from read_parquet('{EXT}/r3ros_work/cit.parquet') where pn is not null")
R['ros_npat'] = c.sql("select count(*), min(pn), max(pn) from ros").fetchone()
c.execute(f"create table usap as select distinct try_cast(appl_no as bigint) an from read_parquet('{W}/cache/rec.parquet') where cc='US' and regexp_full_match(appl_no,'[0-9]{{8}}')")
c.execute(f"create table usrg as select distinct try_cast(rgst_no as bigint) rn from read_parquet('{W}/cache/rec.parquet') where cc='US' and regexp_full_match(coalesce(rgst_no,''),'[0-9]{{7,8}}')")
R['us_appl_n'] = c.sql("select count(*), min(an), max(an) from usap").fetchone()
m = c.sql("""select typ, count(*) n, avg((num in (select pn from kp))::int) in_kpss, avg((num in (select pn from ros))::int) in_ros,
   avg((num in (select an from usap))::int) eq_us_appl, avg((num in (select rn from usrg))::int) eq_us_rgst,
   avg((num between 3000000 and 13000000)::int) in_rng
   from tk where typ in ('R7','R8z','R8n','R8x','P10','P11') group by 1 order by 1""").df()
print(m.to_string()); R['match'] = m.to_dict('records')
# R8n 세부: 10/11/12 접두별, KPSS 발행연도 분포
R['r8n_prefix'] = c.sql("""select left(substr(t,3),2) pfx, count(*) n, avg((num in (select pn from kp))::int) in_kpss,
   avg((num in (select pn from ros))::int) in_ros, avg((num in (select an from usap))::int) eq_us_appl from tk where typ='R8n' group by 1 order by 1""").fetchall()
print(R['r8n_prefix'])
# 오독 후보 (ii): 정규식 키 중 제공 US 출원번호와 같은 값 & KPSS·RoS 어디에도 없음
R['appl_collide'] = c.sql("""select typ, count(*) n, sum((num in (select an from usap))::int) eq_appl,
   sum((num in (select an from usap) and num not in (select pn from kp) and num not in (select pn from ros))::int) eq_appl_not_in_kpss_ros,
   sum((num not in (select pn from kp) and num not in (select pn from ros))::int) not_in_kpss_ros from tk where code_key group by 1 order by 1""").fetchall()
print(R['appl_collide'])
# (iii) 선행 0: R8z 레코드에 같은 번호 R7 토큰이 같이 있는 비율
R['lead0_pair'] = c.sql("""select count(*), avg((exists(select 1 from tk b where b.cc=a.cc and b.appl_no=a.appl_no and b.typ='R7' and b.num=a.num))::int)
   from (select * from tk where typ='R8z' using sample 200000 rows) a""").fetchone()
# 레코드 단위: 같은 레코드에서 공개번호 토큰(P10/P11)과 등록 토큰 공존
R['rec_mix'] = c.sql("""select cc, count(*) n_rec, avg(has_r::int) has_reg, avg(has_p::int) has_pub, avg((has_p and not has_r)::int) pub_only
   from (select cc, appl_no, max((code_key)::int)=1 has_r, max((typ in ('P10','P11'))::int)=1 has_p from tk group by 1,2) group by 1 order by 1""").fetchall()
print(R['rec_mix'])
# 코드 결과 대조: 공유 fam_us.parquet(family src) 재현
c.execute("create table myfam as select distinct cc, appl_no, ksic, num us_pn from tk where code_key")
R['fam_us_shared'] = c.sql(f"select src, count(*), count(distinct us_pn) from read_parquet('{EXT}/r3ros_work/fam_us.parquet') group by 1").fetchall()
R['fam_us_vs_mine'] = c.sql(f"""select count(*) n_shared_family, sum((m.us_pn is not null)::int) in_mine from read_parquet('{EXT}/r3ros_work/fam_us.parquet') f
   left join myfam m using(cc, appl_no, ksic, us_pn) where f.src='family'""").fetchone()
R['mine_n'] = c.sql("select count(*) from myfam").fetchone()
print(R['fam_us_shared'], R['fam_us_vs_mine'], R['mine_n'])
# 1-4 ASOF 발행일: 범위 밖·미실재 번호
P = f"read_parquet('{EXT}/r3ros_patent_sci.parquet')"
R['patent_sci'] = c.sql(f"""select count(*) n, sum((us_pn > (select max(pn) from kp))::int) above_kpss_max, sum((us_pn in (select pn from kp))::int) in_kpss,
   sum((us_pn in (select pn from ros))::int) in_ros, sum((us_pn not in (select pn from kp) and us_pn not in (select pn from ros))::int) in_neither,
   min(issue_year), max(issue_year) from {P}""").fetchone()
R['patent_sci_neither_by_issueyear'] = c.sql(f"""select issue_year, count(*) n, sum((us_pn not in (select pn from kp) and us_pn not in (select pn from ros))::int) neither
   from {P} group by 1 having issue_year between 1995 and 2025 order by 1""").fetchall()
print(R['patent_sci'])
# own src
R['own'] = c.sql(f"select count(*), sum((us_pn in (select pn from kp))::int), sum((us_pn in (select pn from ros))::int) from read_parquet('{EXT}/r3ros_work/fam_us.parquet') where src='own'").fetchone()
c.execute(f"copy tk to '{W}/cache/tk.parquet' (format parquet)")
json.dump(R, open(W + "/res/s1_format.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
