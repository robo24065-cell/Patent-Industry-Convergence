# -*- coding: utf-8 -*-
r"""C15-T3 s1: 모든 해외 문서에 PKR-IPC 배정을 연결별로 다시 만든다(c5_R10 c1_assign.py cascade('pkr', kprov, famsrc=True, use_code=False) 정의).
팔: stored(저장 alt_assign k_pkr) / r8(결함 연결 재구축) / reg(대안 A) / pure(대안 B, 상속 없음).
읽기만: c5_R10/alt_assign, T3 ESSENTIAL_R7/rec_conc.parquet, C10-T3/cache/links.parquet, R3/cache_pub/base_rows.parquet(T3 풀 점검)
쓰기: C15-T3/cache/pkr_alt_<CC>.parquet(제공 라벨 포함 — 작업 폴더 안에만), res/s1_*.json|csv"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C15/C15-T3"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/alt_*.parquet"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
SRC = MC + "/T3_기술셀부상예측/R3/cache_pub/base_rows.parquet"
os.makedirs(W + "/cache/_tmp", exist_ok=True); os.makedirs(W + "/res", exist_ok=True)
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
MODE = "arg_min(k, -n*100000 + cast(k as int))"   # 최빈, 동률은 코드 작은 것 (c1 그대로)
R = {}
con.execute(f"""create table d as select a.cc, a.appl_no, a.appl_year, a.is_inv, a.kr_origin, a.ksic_prov, a.k_pkr, a.m_pkr, r.sg,
   split_part(r.sg,'/',1) mg, left(r.sg,4) sc from read_parquet('{ALT}') a left join read_parquet('{RC}') r using(cc, appl_no)""")
R["rows"] = con.sql("select cc, count(*), count(distinct appl_no), avg((sg is null)::int) from d group by 1 order by 1").fetchall(); log(R["rows"])
con.execute("create table kprov as select appl_no, ksic_prov k from d where cc='KR'")
R["kr_dup_appl"] = con.sql("select count(*) - count(distinct appl_no) from kprov").fetchone()[0]
con.execute("create table krk as select d.sg, d.mg, d.sc, s.k from d join kprov s on d.cc='KR' and d.appl_no=s.appl_no where d.appl_year<=2019")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tp_{lvl} as with g as (select {lvl} key_, k, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by all),
        tot as (select key_ from g group by 1 having sum(n)>=3) select key_, {MODE} k from g join tot using(key_) group by 1""")
con.execute("""create table x as select d.*, coalesce(p1.k, p2.k, p3.k) k_pure,
   case when p1.k is not null then 'ipc_sg' when p2.k is not null then 'ipc_mg' when p3.k is not null then 'ipc_sc' else 'none' end m_pure
   from d left join tp_sg p1 on p1.key_=d.sg left join tp_mg p2 on p2.key_=d.mg left join tp_sc p3 on p3.key_=d.sc where d.cc<>'KR'""")
chk = con.sql("""select cc, count(*) n, avg(case when m_pkr like 'ipc_%' then (k_pure is not distinct from k_pkr)::int end) ipc_rep,
   avg(case when m_pkr like 'ipc_%' then (m_pure = m_pkr)::int end) ipc_method_rep, avg(case when m_pkr='none' then (k_pure is null)::int end) none_rep,
   sum((m_pkr='family')::int) n_family from x group by 1 order by 1""").df()
R["ipc_rebuild"] = chk.round(6).to_dict("records"); log(chk.to_string())
assert (chk.ipc_rep >= 0.99).all(), "IPC 표 재구축 99% 미달 → 중단"
for nm in ("R8", "REG"):
    con.execute(f"""create table f_{nm} as with g as (select l.cc, l.appl_no, s.k, count(*) n from (select distinct cc, appl_no, kr_appl_no from read_parquet('{LINKS}') where link='{nm}') l
        join kprov s on s.appl_no=l.kr_appl_no group by all) select cc, appl_no, {MODE} k from g group by 1,2""")
con.execute("""create table y as select x.cc, x.appl_no, x.appl_year, x.is_inv, x.kr_origin, x.ksic_prov, x.k_pkr k_stored, x.m_pkr, x.k_pure,
   coalesce(f8.k, x.k_pure) k_r8, f8.k is not null fam_r8, coalesce(fr.k, x.k_pure) k_reg, fr.k is not null fam_reg
   from x left join f_R8 f8 using(cc, appl_no) left join f_REG fr using(cc, appl_no)""")
n = con.sql("select count(*), count(distinct cc||'|'||appl_no) from y").fetchone(); assert n[0] == n[1], n
rep = con.sql("""select cc, count(*) n, avg((k_r8 is not distinct from k_stored)::int) r8_eq_stored, sum((k_r8 is distinct from k_stored)::int) n_diff,
   avg(fam_r8::int) fam_r8, avg(fam_reg::int) fam_reg, avg((k_reg is distinct from k_stored)::int) reg_changed, avg((k_pure is distinct from k_stored)::int) pure_changed
   from y group by rollup(cc) order by cc""").df()
R["r8_repro_all_docs"] = rep.round(6).to_dict("records"); log(rep.to_string())
assert (rep.r8_eq_stored >= 0.999).all(), "R8 재구축 99.9% 미달 → 중단"
win = con.sql("""select cc, count(*) n, avg(coalesce(k_stored=ksic_prov,false)::int) a_stored, avg(coalesce(k_r8=ksic_prov,false)::int) r8,
   avg(coalesce(k_reg=ksic_prov,false)::int) reg, avg(coalesce(k_pure=ksic_prov,false)::int) pure, avg(fam_reg::int) famshare_reg
   from y where is_inv and appl_year between 2015 and 2019 group by 1 order by 1""").df()
R["window_agree"] = win.round(6).to_dict("records"); log(win.to_string())
c14 = {"US": (71.90, 70.16), "JP": (73.76, 72.62), "CN": (68.00, 68.00), "EP": (72.82, 71.21)}
R["c14_match"] = {r.cc: dict(reg=round(100 * r.reg, 2), pure=round(100 * r.pure, 2), ok=abs(100 * r.reg - c14[r.cc][0]) <= 0.01 and abs(100 * r.pure - c14[r.cc][1]) <= 0.01)
                  for r in win.itertuples()}
log("C14 대조", R["c14_match"])
# T3 풀(base_rows 해외 레코드) 기준 변화: 레코드 단위
con.execute(f"create table b as select cc, appl_no, ksic prov from read_parquet('{SRC}') where cc<>'KR'")
pool = con.sql("""select b.cc, count(*) n_rec, avg((y.appl_no is null)::int) no_alt_doc,
   avg((y.k_stored is not null)::int) kept_stored, avg((y.k_reg is not null)::int) kept_reg, avg((y.k_pure is not null)::int) kept_pure,
   avg((y.k_r8 is not distinct from y.k_stored)::int) r8_eq_stored,
   avg((y.k_reg is distinct from y.k_stored)::int) changed_reg, avg((y.k_pure is distinct from y.k_stored)::int) changed_pure,
   avg(coalesce(y.k_stored=b.prov,false)::int) agree_stored, avg(coalesce(y.k_reg=b.prov,false)::int) agree_reg, avg(coalesce(y.k_pure=b.prov,false)::int) agree_pure,
   avg(y.fam_r8::int) fam_r8, avg(y.fam_reg::int) fam_reg
   from b left join y using(cc, appl_no) group by rollup(b.cc) order by b.cc""").df()
pool.to_csv(W + "/res/s1_t3pool_change.csv", index=False, encoding="utf-8-sig"); log(pool.round(4).to_string())
R["t3pool"] = pool.round(6).to_dict("records")
# 연결 선택 편향(창, REG 연결 vs 비연결): 국가별 연도·우선권 KR·IPC표 일치
bias = con.sql("""select cc, fam_reg, count(*) n, avg(kr_origin::int) kr_origin, avg(appl_year) mean_year,
   avg(coalesce(k_pure=ksic_prov,false)::int) ipc_agree, avg(coalesce(k_reg=ksic_prov,false)::int) reg_agree
   from y where is_inv and appl_year between 2015 and 2019 and cc<>'CN' group by all order by 1,2""").df()
bias.to_csv(W + "/res/s1_link_bias.csv", index=False, encoding="utf-8-sig"); log(bias.round(4).to_string())
for cc in ("US", "JP", "CN", "EP"):
    con.execute(f"""copy (select cc, appl_no, appl_year, is_inv, kr_origin, ksic_prov, k_stored, k_r8, k_reg, k_pure, fam_r8, fam_reg from y where cc='{cc}')
        to '{W}/cache/pkr_alt_{cc}.parquet' (format parquet, compression zstd)""")
json.dump(R, open(W + "/res/s1_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done")
