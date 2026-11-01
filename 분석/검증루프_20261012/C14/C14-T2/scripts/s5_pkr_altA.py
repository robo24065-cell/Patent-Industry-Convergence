# -*- coding: utf-8 -*-
r"""C14-T2 s5: 부록 C-AGREE 진단용 PKR-IPC를 대안 A(등록번호 연결 상속, C10-T3 L_REG)로 재계산.
정의(c5_R10 c1_assign.py cascade('pkr', 'kprov', famsrc=True, use_code=False)):
  해외 문서 k_pkr = coalesce(패밀리 연결 KR 문서들의 제공 라벨 최빈, IPC 표 sg, mg, sc)
  IPC 표 = KR 문서(appl_year<=2019)의 제공 라벨로 (sg|mg|sc)별 최빈(키 문서 수 >=3, 동률은 코드 작은 것)
  일치 = avg(coalesce(k_pkr = ksic_prov, false)), 창 = 발명(is_inv) 출원 2015–19.
연결만 교체: R8(결함, 원값 재현용) / REG(대안 A) / PUB(보조) / UNI(REG∪PUB) / pure(상속 없음).
읽기만: c5_R10/alt_assign, T3 rec_conc, C10-T3/cache/links.parquet, R11 cache/fdoc.parquet. 쓰기: 이 작업 폴더."""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C14/C14-T2"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/alt_*.parquet"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
FDOC = MC + "/_shared_report/label_origin_R10/foreign_public_R11/cache/fdoc.parquet"
os.makedirs(W + "/cache/_tmp", exist_ok=True)
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
MODE = "arg_min(k, -n*100000 + cast(k as int))"
R = {}
con.execute(f"""create table d as select a.cc, a.appl_no, a.appl_year, a.is_inv, a.ksic_prov, a.k_pkr, a.m_pkr, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc
   from read_parquet('{ALT}') a left join read_parquet('{RC}') r using(cc, appl_no)""")
R["rows"] = con.sql("select cc, count(*), avg((sg is null)::int) from d group by 1 order by 1").fetchall(); log(R["rows"])
con.execute("create table kprov as select appl_no, ksic_prov k from d where cc='KR'")
R["kr_dup_appl"] = con.sql("select count(*) - count(distinct appl_no) from kprov").fetchone()[0]
con.execute("create table krk as select d.sg, d.mg, d.sc, s.k from d join kprov s on d.cc='KR' and d.appl_no=s.appl_no where d.appl_year<=2019")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tp_{lvl} as with g as (select {lvl} key_, k, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by all),
        tot as (select key_ from g group by 1 having sum(n)>=3) select key_, {MODE} k from g join tot using(key_) group by 1""")
con.execute("""create table x as select d.*, coalesce(p1.k, p2.k, p3.k) r_pkr,
   case when p1.k is not null then 'ipc_sg' when p2.k is not null then 'ipc_mg' when p3.k is not null then 'ipc_sc' else 'none' end r_m
   from d left join tp_sg p1 on p1.key_=d.sg left join tp_mg p2 on p2.key_=d.mg left join tp_sc p3 on p3.key_=d.sc where d.cc<>'KR'""")
chk = con.sql("""select cc, avg(case when m_pkr like 'ipc_%' then (r_pkr is not distinct from k_pkr)::int end) ipc_rep,
   avg(case when m_pkr like 'ipc_%' then (r_m = m_pkr)::int end) ipc_method_rep,
   avg(case when m_pkr='none' then (r_pkr is null)::int end) none_rep, sum((m_pkr='family')::int) n_family
   from x group by 1 order by 1""").df()
R["rebuild_check"] = chk.round(6).to_dict("records"); log(chk.to_string())
assert (chk.ipc_rep >= 0.99).all(), "IPC 표 재구축 99% 미달"
con.execute(f"create table lk as select * from read_parquet('{LINKS}')")
con.execute("create table lk_uni as select distinct cc, appl_no, kr_appl_no from lk where link in ('REG','PUB')")
for nm, src in (("r8", "(select cc, appl_no, kr_appl_no from lk where link='R8')"), ("reg", "(select cc, appl_no, kr_appl_no from lk where link='REG')"),
                ("pub", "(select cc, appl_no, kr_appl_no from lk where link='PUB')"), ("uni", "lk_uni")):
    con.execute(f"""create table fp_{nm} as with g as (select l.cc, l.appl_no, s.k, count(*) n from {src} l join kprov s on s.appl_no=l.kr_appl_no
        group by all) select cc, appl_no, {MODE} k from g group by 1,2""")
R["r8_repro"] = con.sql("""select x.cc, avg(case when x.m_pkr='family' then (f.k is not distinct from x.k_pkr)::int end) fam_repro,
   sum((x.m_pkr='family')::int) n_stored_family, sum((f.k is not null)::int) n_r8_inherit
   from x left join fp_r8 f using(cc, appl_no) group by 1 order by 1""").fetchall(); log("r8 repro", R["r8_repro"])
con.execute("""create table y as select x.cc, x.appl_no, x.appl_year, x.is_inv, x.ksic_prov, x.k_pkr k_stored, x.m_pkr, x.r_pkr k_pure,
   coalesce(f8.k, x.r_pkr) k_r8, f8.k is not null fam_r8,
   coalesce(fr.k, x.r_pkr) k_reg, fr.k is not null fam_reg,
   coalesce(fb.k, x.r_pkr) k_pub, fb.k is not null fam_pub,
   coalesce(fu.k, x.r_pkr) k_uni, fu.k is not null fam_uni
   from x left join fp_r8 f8 using(cc, appl_no) left join fp_reg fr using(cc, appl_no) left join fp_pub fb using(cc, appl_no) left join fp_uni fu using(cc, appl_no)""")
n = con.sql("select count(*), count(distinct cc||'|'||appl_no) from y").fetchone(); assert n[0] == n[1], n
arms = ["stored", "r8", "reg", "pub", "uni", "pure"]
def agree(where, tag):
    cols = ", ".join(f"avg(coalesce(k_{a}=ksic_prov,false)::int) pkr5_{a}" for a in arms) + ", " + \
           ", ".join(f"avg(fam_{a}::int) famshare_{a}" for a in ("r8", "reg", "pub", "uni")) + ", " + \
           ", ".join(f"avg(case when fam_{a} then (k_{a}=ksic_prov)::int end) famagree_{a}" for a in ("r8", "reg", "pub", "uni")) + ", " + \
           "avg(case when not fam_reg then (k_pure=ksic_prov)::int end) ipc_agree_nonlinked_reg, avg(case when fam_reg then (k_pure=ksic_prov)::int end) ipc_agree_linked_reg"
    dd = con.sql(f"select cc, count(*) n, {cols} from y where {where} group by rollup(cc) order by cc").df()
    dd.insert(0, "window", tag); return dd
T = pd.concat([agree("is_inv and appl_year between 2015 and 2019", "출원2015-19_발명"), agree("true", "전체")])
T["cc"] = T["cc"].fillna("해외합")
T.to_csv(W + "/cache/s5_pkr_agree_full.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(T[["window", "cc", "n"] + [f"pkr5_{a}" for a in arms] + ["famshare_reg", "famagree_reg"]].round(4).to_string())
# R11: fdoc.k_pkr 가 상속 없는 IPC 표 값(k_pure)과 같은지 -> R11-PKR이 결함 연결에 기대는지
con.execute(f"create table fd as select * from read_parquet('{FDOC}')")
R["r11_fdoc_pkr_vs_pure"] = con.sql("""select fd.cc, count(*) n, avg((fd.k_pkr is not distinct from y.k_pure)::int) same_pure,
   avg((fd.k_pkr is not distinct from y.k_stored)::int) same_stored, avg((y.m_pkr='family')::int) r8_family_share,
   avg((fd.k_pkr is not distinct from y.k_reg)::int) same_reg
   from fd join y using(cc, appl_no) group by 1 order by 1""").fetchall(); log("R11 fdoc", R["r11_fdoc_pkr_vs_pure"])
R["fdoc_rows"] = con.sql("select cc, count(*), count(distinct appl_no) from fd group by 1 order by 1").fetchall()
json.dump(R, open(W + "/cache/s5_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
# 문서 단위 결과 일부 저장(독립 재현 대조용, 제공 라벨 포함 → 작업 폴더 cache 안에만)
con.execute(f"copy (select cc, appl_no, k_reg, k_pure, fam_reg from y where is_inv and appl_year between 2015 and 2019) to '{W}/cache/s5_doc_reg.parquet' (format parquet, compression zstd)")
log("done")
