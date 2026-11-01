# -*- coding: utf-8 -*-
r"""C10-T3 s2: FREE-IPC·TXT 해외 배정을 올바른 연결(L_REG, L_PUB, L_UNION)로 다시 만든다(c5_R10 c1_assign.py 규칙, 연결만 교체).
읽기만: c5_R10/alt_assign, T3 rec_conc·txt_assign, T1 kr_txtfree, C10-T3/cache/links.parquet
쓰기: cache/alt_fix_{CC}.parquet, res/s2_agree.csv, res/s2_stats.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C10/C10-T3"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/alt_*.parquet"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
TXT = MC + "/T3_기술셀부상예측/TXT_R9/txt_assign.parquet"
FREE = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
KM = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/conc_ksic_map.csv"
LINKS = W + "/cache/links.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
MODE = "arg_min(k, -n*100000 + cast(k as int))"
km = pd.read_csv(KM, dtype=str)[["ksic", "code"]]; km["kc"] = km.groupby("code").ksic.transform("size"); con.register("km", km)
R = {}
con.execute(f"""create table d as select a.cc, a.appl_no, a.appl_year, a.is_inv, a.kr_origin, a.ksic_prov, a.code, a.k_in_code,
   a.k_txt, a.m_txt, a.k_free, a.m_free, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc
   from read_parquet('{ALT}') a left join read_parquet('{RC}') r using(cc, appl_no)""")
R["rows"] = con.sql("select cc, count(*), avg((sg is null)::int) from d group by 1 order by 1").fetchall(); log(R["rows"])
con.execute(f"create table kfree as select appl_no, k_free k from read_parquet('{FREE}') where k_free is not null")
R["kr_free_equals_alt"] = con.sql("select avg((d.k_free is not distinct from f.k)::int), count(*) from d left join kfree f using(appl_no) where d.cc='KR'").fetchone()
# ---- FREE 표 (KR 출원 ≤2019, 코드 무관) ----
con.execute("create table krk_f as select d.sg, d.mg, d.sc, f.k from d join kfree f on d.cc='KR' and d.appl_no=f.appl_no where d.appl_year<=2019")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tf_{lvl} as with g as (select {lvl} key_, k, count(*) n from krk_f where {lvl} is not null and {lvl}<>'' group by all),
        tot as (select key_ from g group by 1 having sum(n)>=3) select key_, {MODE} k from g join tot using(key_) group by 1""")
# ---- TXT 표 (T3 풀 KR 'text' 문서, 코드 안) ----
con.execute(f"create table ta as select cc, appl_no, code tcode, ksic_txt, \"method\" tm from read_parquet('{TXT}')")
con.execute("create table krk_t as select r.code, r.sg, r.mg, r.sc, ta.ksic_txt k from ta join d r on r.cc='KR' and r.appl_no=ta.appl_no where ta.cc='KR' and ta.tm='text'")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tt_{lvl} as with g as (select code, {lvl} key_, k, count(*) n from krk_t where {lvl} is not null and {lvl}<>'' group by all),
        tot as (select code, key_ from g group by all having sum(n)>=3) select code, key_, {MODE} k from g join tot using(code, key_) group by all""")
con.execute(f"create table tt_code as with g as (select code, k, count(*) n from krk_t group by all) select code, {MODE} k from g group by 1")
con.execute("""create table x as select d.*, ta.ksic_txt t3k, ta.tm t3m, ta.tcode,
   coalesce(f1.k, f2.k, f3.k) r_free,
   case when d.code is null then null when d.k_in_code=1 then (select min(ksic) from km where km.code=d.code) else coalesce(t1.k, t2.k, t3.k, t4.k) end r_txt
   from d left join ta using(cc, appl_no)
   left join tf_sg f1 on f1.key_=d.sg left join tf_mg f2 on f2.key_=d.mg left join tf_sc f3 on f3.key_=d.sc
   left join tt_sg t1 on t1.code=d.code and t1.key_=d.sg left join tt_mg t2 on t2.code=d.code and t2.key_=d.mg
   left join tt_sc t3 on t3.code=d.code and t3.key_=d.sc left join tt_code t4 on t4.code=d.code
   where d.cc<>'KR'""")
# 재구축 점검(PREREG ≥ 99%)
chk = con.sql("""select cc,
   avg(case when m_free like 'ipc_%' then (r_free is not distinct from k_free)::int end) free_rep,
   avg(case when m_free='none' then (r_free is null)::int end) free_none_rep,
   avg(case when t3m in ('ipc_sg','ipc_mg','ipc_sc','code_mode') then (r_txt is not distinct from k_txt)::int end) txt_rep_t3tab,
   avg(case when m_txt like 'extra_%' then (r_txt is not distinct from k_txt)::int end) txt_rep_extra,
   avg(case when t3m is not null then (t3k is not distinct from k_txt)::int end) t3_equals_stored,
   avg(case when t3m is not null then (tcode is not distinct from code)::int end) t3_code_equals
   from x group by 1 order by 1""").df()
R["rebuild_check"] = chk.round(6).to_dict("records"); log(chk.to_string())
assert (chk.free_rep >= 0.99).all() and (chk.txt_rep_t3tab.fillna(1) >= 0.99).all(), "재구축 99% 미달"
# ---- 연결별 패밀리 상속 ----
links = {"reg": "REG", "pub": "PUB", "r8": "R8"}
con.execute(f"create table lk as select * from read_parquet('{LINKS}')")
con.execute("create table lk_uni as select distinct cc, appl_no, kr_appl_no from lk where link in ('REG','PUB')")
multi = tuple(km.loc[km.kc > 1, "code"].unique())
for nm in ("reg", "pub", "uni", "r8"):
    src = "lk_uni" if nm == "uni" else f"(select cc, appl_no, kr_appl_no from lk where link='{links[nm]}')"
    con.execute(f"""create table ff_{nm} as with g as (select l.cc, l.appl_no, f.k, count(*) n from {src} l join kfree f on f.appl_no=l.kr_appl_no group by all)
        select cc, appl_no, {MODE} k from g group by 1,2""")
    # TXT: 풀 안·복수 코드 해외 문서 ↔ 풀 안 같은 코드 KR 문서 ksic_txt
    con.execute(f"""create table ft_{nm} as with g as (select l.cc, l.appl_no, k.ksic_txt k, count(*) n from {src} l
        join ta fo on fo.cc=l.cc and fo.appl_no=l.appl_no join ta k on k.cc='KR' and k.appl_no=l.kr_appl_no and k.tcode=fo.tcode
        where l.cc<>'KR' and fo.tcode in {multi} group by all) select cc, appl_no, {MODE} k from g group by 1,2""")
# R8 재현: ft_r8 이 txt_assign 'family'와 같은지, ff_r8이 저장 family k_free와 같은지
R["r8_repro"] = con.sql("""select x.cc,
   avg(case when x.m_free='family' then (ff.k is not distinct from x.k_free)::int end) free_family_repro,
   sum((x.m_free='family')::int) n_free_family, sum((ff.k is not null)::int) n_ff_r8,
   avg(case when x.t3m='family' then (ft.k is not distinct from x.t3k)::int end) txt_family_repro,
   sum((x.t3m='family')::int) n_txt_family, sum((ft.k is not null)::int) n_ft_r8
   from x left join ff_r8 ff using(cc, appl_no) left join ft_r8 ft using(cc, appl_no) group by 1 order by 1""").fetchall()
log("r8 repro", R["r8_repro"])
sel = []
for nm in ("reg", "pub", "uni"):
    sel.append(f"coalesce(ff_{nm}.k, x.r_free) k_free_{nm}, (ff_{nm}.k is not null) fam_free_{nm}")
    sel.append(f"""case when x.t3m is null then x.k_txt when x.code in {multi} and ft_{nm}.k is not null then ft_{nm}.k
                   when x.t3m='family' then x.r_txt else x.t3k end k_txt_{nm}, (x.t3m is not null and x.code in {multi} and ft_{nm}.k is not null) fam_txt_{nm}""")
joins = " ".join(f"left join ff_{nm} on ff_{nm}.cc=x.cc and ff_{nm}.appl_no=x.appl_no left join ft_{nm} on ft_{nm}.cc=x.cc and ft_{nm}.appl_no=x.appl_no" for nm in ("reg", "pub", "uni"))
con.execute(f"""create table y as select x.cc, x.appl_no, x.appl_year, x.is_inv, x.kr_origin, x.ksic_prov, x.code, x.k_in_code,
   x.k_free k_free_stored, x.m_free, x.k_txt k_txt_stored, x.m_txt, x.t3m,
   case when x.m_free='family' then x.r_free else x.k_free end k_free_pure,
   case when x.t3m='family' then x.r_txt else x.k_txt end k_txt_pure, {', '.join(sel)}
   from x {joins}""")
n = con.sql("select count(*), count(distinct cc||appl_no) from y").fetchone(); assert n[0] == n[1], n
# KR 행(지표 F 시나리오용): 저장 그대로
con.execute("""create table ykr as select cc, appl_no, appl_year, is_inv, kr_origin, ksic_prov, code, k_in_code, k_free k_free_stored, m_free, k_txt k_txt_stored, m_txt, null t3m,
   k_free k_free_pure, k_txt k_txt_pure, k_free k_free_reg, false fam_free_reg, k_txt k_txt_reg, false fam_txt_reg,
   k_free k_free_pub, false fam_free_pub, k_txt k_txt_pub, false fam_txt_pub, k_free k_free_uni, false fam_free_uni, k_txt k_txt_uni, false fam_txt_uni from d where cc='KR'""")
for CC in ("US", "JP", "CN", "EP", "KR"):
    src = "ykr" if CC == "KR" else f"(select * from y where cc='{CC}')"
    con.execute(f"copy (select * from {src}) to '{W}/cache/alt_fix_{CC}.parquet' (format parquet, compression zstd)")
log("saved")
A = f"read_parquet('{W}/cache/alt_fix_*.parquet')"
arms = ["stored", "pure", "reg", "pub", "uni"]
def agree(where, tag):
    cols = ", ".join(f"avg(coalesce(k_free_{a}=ksic_prov,false)::int) free5_{a}" for a in arms) + ", " + \
           ", ".join(f"avg(coalesce(k_txt_{a}=ksic_prov,false)::int) txt5_{a}" for a in arms) + ", " + \
           ", ".join(f"avg(fam_free_{a}::int) famshare_free_{a}, avg(fam_txt_{a}::int) famshare_txt_{a}" for a in ("reg", "pub", "uni")) + ", " + \
           "avg((m_free='family')::int) famshare_free_r8, avg((t3m='family')::int) famshare_txt_r8, " + \
           ", ".join(f"avg(case when fam_free_{a} then (k_free_{a}=ksic_prov)::int end) famagree_free_{a}" for a in ("reg", "pub", "uni")) + ", " + \
           "avg(case when m_free='family' then (k_free_stored=ksic_prov)::int end) famagree_free_r8"
    dd = con.sql(f"select cc, count(*) n, {cols} from {A} where cc<>'KR' and {where} group by rollup(cc) order by cc").df()
    dd.insert(0, "window", tag); return dd
T = pd.concat([agree("is_inv and appl_year between 2015 and 2019", "출원2015-19_발명"), agree("true", "전체"),
               agree("is_inv and kr_origin and appl_year between 2017 and 2019", "분자_우선권KR_2017-19")])
T.to_csv(W + "/res/s2_agree.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(T[["window", "cc", "n"] + [f"free5_{a}" for a in arms] + [f"txt5_{a}" for a in arms]].round(4).to_string())
# 연결 간 일치: 두 연결 모두 상속받은 문서에서 FREE 라벨이 같은 비율, KSIC 일치
R["reg_vs_pub_label_same"] = con.sql(f"""select cc, sum((fam_free_reg and fam_free_pub)::int) n_both, avg(case when fam_free_reg and fam_free_pub then (k_free_reg=k_free_pub)::int end) free_same,
   avg(case when fam_txt_reg and fam_txt_pub then (k_txt_reg=k_txt_pub)::int end) txt_same,
   avg((k_free_reg is not distinct from k_free_pub)::int) free_same_all, avg((k_txt_reg is not distinct from k_txt_pub)::int) txt_same_all
   from {A} where cc<>'KR' group by rollup(cc) order by 1""").fetchall()
# 저장 대비 라벨이 바뀐 문서 비중
R["changed_vs_stored"] = con.sql(f"""select cc, avg((k_free_reg is distinct from k_free_stored)::int) free_reg, avg((k_free_pub is distinct from k_free_stored)::int) free_pub,
   avg((k_txt_reg is distinct from k_txt_stored)::int) txt_reg, avg((k_txt_pub is distinct from k_txt_stored)::int) txt_pub,
   avg((k_free_pure is distinct from k_free_stored)::int) free_pure, avg((k_txt_pure is distinct from k_txt_stored)::int) txt_pure
   from {A} where cc<>'KR' group by rollup(cc) order by 1""").fetchall()
json.dump(R, open(W + "/res/s2_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("done", R["reg_vs_pub_label_same"], R["changed_vs_stored"])
