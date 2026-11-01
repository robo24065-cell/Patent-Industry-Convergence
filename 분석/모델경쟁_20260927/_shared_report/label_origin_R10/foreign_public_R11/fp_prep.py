# -*- coding: utf-8 -*-
r"""R11 foreign_public — 준비(PREREG.txt). 읽기만: c5_R10/alt_assign, T3 rec_conc·txt_assign, label_origin_R10/cache/fam_reg.
쓰기: 이 폴더 cache/, res_prep.json, res_docagree.csv.
- alt_assign의 'family'(R8 결함 연결 상속) 배정을 버리고 c1_assign.py와 같은 규칙의 IPC 표로 다시 배정한다.
- 고친 연결(fam_reg)로 KR 패밀리 문서의 TXT-FREE를 상속하는 FREE-fam 판도 만든다.
출력: cache/fdoc.parquet (패밀리 표본 해외 문서×KR 문서: cc, appl_no, kr_appl_no, prov, code, k_in_code, k_txt, k_free, k_ffam, k_pkr)"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
f = lambda *p: os.path.join(MC, *p).replace("\\", "/")
ALT = f("_shared_report", "fusion_advantage", "c5_R10", "alt_assign", "alt_*.parquet")
RC = f("T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet"); TXT = f("T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet")
FAM = f("_shared_report", "label_origin_R10", "cache", "fam_reg.parquet")
t = time.time(); log = lambda *a: print(*a, f"[{time.time() - t:.0f}s]", flush=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{CACHE.replace(chr(92), '/')}/_tmp'")
MODE = "arg_min(k, -n*100000 + cast(k as int))"  # c1_assign.py와 같은 최빈 규칙(동률은 코드 작은 것)
con.execute(f"""create table d as select a.*, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc
                from read_parquet('{ALT}') a left join read_parquet('{RC}') r using(cc, appl_no)""")
log("문서", con.sql("select cc, count(*), avg((sg is null)::int) from d group by 1 order by 1").fetchall())
R = {}

def tables(name, src, use_code, where):
    key = "code, " if use_code else ""
    con.execute(f"create or replace table krk as select {key} d.sg, d.mg, d.sc, s.k from d join {src} s on d.cc='KR' and d.appl_no=s.appl_no where {where}")
    for lvl in ("sg", "mg", "sc"):
        con.execute(f"""create or replace table {name}_{lvl} as with g as (select {key} {lvl} key_, k, count(*) n from krk
            where {lvl} is not null and {lvl}<>'' group by all), tot as (select {key} key_ from g group by all having sum(n)>=3)
            select {key} key_, {MODE} k from g join tot using({key} key_) group by all""")
    if use_code:
        con.execute(f"create or replace table {name}_code as with g as (select code, k, count(*) n from krk group by all) select code, {MODE} k from g group by 1")

con.execute("create table kfree as select appl_no, k_free k from d where cc='KR'")
con.execute("create table kprov as select appl_no, ksic_prov k from d where cc='KR'")
con.execute(f"create table ktxt as select appl_no, ksic_txt k from read_parquet('{TXT}') where cc='KR' and method='text'")
tables("tf", "kfree", False, "d.appl_year<=2019")
tables("tp", "kprov", False, "d.appl_year<=2019")
tables("tt", "ktxt", True, "d.code is not null")
con.execute("""create table x as select d.cc, d.appl_no, d.appl_year, d.is_inv, d.ksic_prov prov, d.code, d.k_in_code, d.k_cw,
    d.k_txt, d.m_txt, d.k_free, d.m_free, d.k_pkr, d.m_pkr,
    coalesce(f1.k, f2.k, f3.k) r_free, coalesce(p1.k, p2.k, p3.k) r_pkr,
    case when d.code is null then null when d.k_in_code=1 then d.k_cw else coalesce(t1.k, t2.k, t3.k, t4.k) end r_txt
  from d left join tf_sg f1 on f1.key_=d.sg left join tf_mg f2 on f2.key_=d.mg left join tf_sc f3 on f3.key_=d.sc
  left join tp_sg p1 on p1.key_=d.sg left join tp_mg p2 on p2.key_=d.mg left join tp_sc p3 on p3.key_=d.sc
  left join tt_sg t1 on t1.code=d.code and t1.key_=d.sg left join tt_mg t2 on t2.code=d.code and t2.key_=d.mg
  left join tt_sc t3 on t3.code=d.code and t3.key_=d.sc left join tt_code t4 on t4.code=d.code
  where d.cc<>'KR'""")
# 재구축 점검(PREREG: family 아닌 문서에서 재구축 = 저장 값 ≥ 99%)
chk = con.sql("""select cc,
   avg(case when m_free<>'family' and m_free<>'none' then (r_free is not distinct from k_free)::int end) free_rep,
   avg(case when m_pkr<>'family' and m_pkr<>'none' then (r_pkr is not distinct from k_pkr)::int end) pkr_rep,
   avg(case when m_txt like 'extra_%' then (r_txt is not distinct from k_txt)::int end) txt_rep_extra,
   avg(case when m_txt in ('ipc_sg','ipc_mg','ipc_sc','code_mode') then (r_txt is not distinct from k_txt)::int end) txt_rep_t3tab,
   avg(case when m_txt='single' then (r_txt is not distinct from k_txt)::int end) txt_rep_single
   from x group by 1 order by 1""").df()
R["rebuild_check"] = chk.round(5).to_dict("records"); log(chk.to_string())
con.execute("""create table y as select *, case when m_free='family' then r_free else k_free end k_free_p,
   case when m_pkr='family' then r_pkr else k_pkr end k_pkr_p, case when m_txt='family' then r_txt else k_txt end k_txt_p from x""")
# 고친 연결로 KR TXT-FREE 상속(FREE-fam)
con.execute(f"""create table ffam as with g as (select fl.cc, fl.appl_no, s.k, count(*) n from read_parquet('{FAM}') fl
     join kfree s on s.appl_no=fl.kr_appl_no where s.k is not null group by all) select cc, appl_no, {MODE} k from g group by 1,2""")
con.execute("create table z as select y.*, coalesce(ff.k, y.k_free_p) k_ffam, ff.k is not null via_fam from y left join ffam ff using(cc, appl_no)")
# 부산물: 해외 문서 단위 일치(출원 2015–19 발명) — c1_label_agreement의 free5와 같은 정의
da = con.sql("""select cc, count(*) n, avg(coalesce(k_free=prov,false)::int) free5_R10_stored, avg(coalesce(k_free_p=prov,false)::int) free5_pure,
     avg(coalesce(k_ffam=prov,false)::int) free5_famfix, avg(via_fam::int) famfix_share, avg((m_free='family')::int) r8_family_share,
     avg(coalesce(k_txt=prov,false)::int) txt5_R10_stored, avg(coalesce(k_txt_p=prov,false)::int) txt5_pure,
     avg(coalesce(k_pkr_p=prov,false)::int) pkr5_pure, avg(case when via_fam then (k_ffam=prov)::int end) famfix_agree_on_linked
     from z where is_inv and appl_year between 2015 and 2019 group by rollup(cc) order by cc""").df()
da.to_csv(os.path.join(HERE, "res_docagree.csv"), index=False, encoding="utf-8-sig"); log(da.round(4).to_string())
# 패밀리 표본 문서
con.execute(f"""copy (select fl.cc, fl.appl_no, fl.kr_appl_no, z.prov, z.code, z.k_in_code, z.k_txt_p k_txt, z.k_free_p k_free,
     kf.k k_ffam, z.k_pkr_p k_pkr, z.m_free, z.m_txt, fl.fk from read_parquet('{FAM}') fl join z using(cc, appl_no)
     left join kfree kf on kf.appl_no=fl.kr_appl_no) to '{CACHE.replace(chr(92), '/')}/fdoc.parquet' (format parquet)""")
R["fdoc"] = con.sql(f"select cc, count(*), avg((prov=fk)::int), avg((k_free is null)::int), avg((k_txt is null)::int), avg((m_free='family')::int), avg((m_txt='family')::int) from '{CACHE.replace(chr(92), '/')}/fdoc.parquet' group by 1 order by 1").fetchall()
log(R["fdoc"])
json.dump(R, open(os.path.join(HERE, "res_prep.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
