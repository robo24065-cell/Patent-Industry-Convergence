# -*- coding: utf-8 -*-
r"""C11-T2 s1: R9 T3 txt_assign의 해외 패밀리 상속을 연결 팔별로 다시 만든다(R9 s01 규칙, 연결만 교체).
 팔: r8(fam_link, 재현) | reg(links REG) | pub(links PUB) | pure(상속 없음)
 KR 행·단일 코드 행은 저장 txt_assign 그대로(링크 무관). TXT-IPC 표는 저장 KR 'text' 배정 + rec_conc sg로 재구축.
읽기만: TXT_R9/txt_assign.parquet, ESSENTIAL_R7/rec_conc.parquet·conc_ksic_map.csv, _ideas_R8/cache/fam_link.parquet, C10-T3/cache/links.parquet·alt_fix_*.parquet
쓰기: C11-T2/cache/txt_assign_{arm}.parquet, res/s1_*.json|csv
사용: python s1_txt_assign_arms.py"""
import os, sys, json, time
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3"
TA = T3 + "/TXT_R9/txt_assign.parquet"; RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; FAM = MC + "/_ideas_R8/cache/fam_link.parquet"
LINKS = C10 + "/cache/links.parquet"; AF = C10 + "/cache/alt_fix_*.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
os.makedirs(W + "/cache/_tmp", exist_ok=True); os.makedirs(W + "/res", exist_ok=True)
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]
multi = sorted(km.groupby("code").ksic.size().loc[lambda s: s > 1].index)
MUL = "(" + ",".join(f"'{c}'" for c in multi) + ")"
MODE = "arg_min(k, -n*100000 + cast(k as int))"   # s01과 같은 최빈·동률 규칙
R = {"n_multi_codes": len(multi)}
con.execute(f"create table ta as select cc, appl_no, code, ksic_txt, \"method\" m, ksic_prov from read_parquet('{TA}')")
R["ta_rows"] = con.sql("select cc, count(*), count(distinct appl_no) from ta group by 1 order by 1").fetchall()
con.execute(f"create table rc as select cc, appl_no, sg from read_parquet('{RC}')")
R["rc_dup"] = con.sql("select count(*) - count(distinct cc||'|'||appl_no) from rc").fetchone()[0]
# KR text 문서 + sg → 표
con.execute("create table krk as select t.code, rc.sg, split_part(rc.sg,'/',1) mg, left(rc.sg,4) sc, t.ksic_txt k from ta t join rc using(cc, appl_no) where t.cc='KR' and t.m='text'")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tab_{lvl} as with g as (select code, {lvl} key_, k, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by all),
        tot as (select code, key_ from g group by all having sum(n)>=3) select code, key_, {MODE} k from g join tot using(code, key_) group by all""")
con.execute(f"create table tab_code as with g as (select code, k, count(*) n from krk group by all) select code, {MODE} k from g group by 1")
# 해외 복수 코드 행
con.execute(f"create table fx as select t.cc, t.appl_no, t.code, t.ksic_txt k_stored, t.m m_stored, rc.sg from ta t left join rc using(cc, appl_no) where t.cc<>'KR' and t.code in {MUL}")
R["fx_rows"] = con.sql("select cc, count(*), sum((sg is null)::int), sum((m_stored='family')::int) from fx group by 1 order by 1").fetchall(); log("fx", R["fx_rows"])
con.execute("""create table fx_tab as select fx.cc, fx.appl_no, coalesce(a.k, b.k, c.k, d.k) k_tab,
   case when a.k is not null then 'ipc_sg' when b.k is not null then 'ipc_mg' when c.k is not null then 'ipc_sc' else 'code_mode' end m_tab
   from fx left join tab_sg a on a.code=fx.code and a.key_=fx.sg left join tab_mg b on b.code=fx.code and b.key_=split_part(fx.sg,'/',1)
   left join tab_sc c on c.code=fx.code and c.key_=left(fx.sg,4) left join tab_code d on d.code=fx.code""")
R["table_rebuild_vs_stored_nonfamily"] = con.sql("""select fx.cc, count(*), avg((t.k_tab=fx.k_stored)::int), avg((t.m_tab=fx.m_stored)::int)
   from fx join fx_tab t using(cc, appl_no) where fx.m_stored<>'family' group by 1 order by 1""").fetchall()
log("표 재구축", R["table_rebuild_vs_stored_nonfamily"])
con.execute("create table krt as select appl_no, code, ksic_txt from ta where cc='KR'")
srcs = {"r8": f"select cc, appl_no, kr_appl_no from read_parquet('{FAM}')",
        "reg": f"select cc, appl_no, kr_appl_no from read_parquet('{LINKS}') where link='REG'",
        "pub": f"select cc, appl_no, kr_appl_no from read_parquet('{LINKS}') where link='PUB'"}
R["link_rows"] = {a: con.sql(f"select cc, count(*), count(distinct cc||'|'||appl_no||'|'||kr_appl_no) from ({q}) group by 1 order by 1").fetchall() for a, q in srcs.items()}
stats = []
for arm in ("r8", "reg", "pub", "pure"):
    if arm == "pure":
        con.execute("create or replace table fam as select null::varchar cc, null::varchar appl_no, null::varchar k where false")
    else:
        con.execute(f"""create or replace table fam as with g as (select l.cc, l.appl_no, k.ksic_txt k, count(*) n from ({srcs[arm]}) l
            join fx on fx.cc=l.cc and fx.appl_no=l.appl_no join krt k on k.appl_no=l.kr_appl_no and k.code=fx.code group by all)
            select cc, appl_no, {MODE} k from g group by 1,2""")
    out = f"{W}/cache/txt_assign_{arm}.parquet"
    con.execute(f"""copy (select t.cc, t.appl_no, t.code,
          case when t.cc<>'KR' and t.code in {MUL} then coalesce(f.k, x.k_tab) else t.ksic_txt end ksic_txt,
          case when t.cc<>'KR' and t.code in {MUL} then (case when f.k is not null then 'family' else x.m_tab end) else t.m end "method", t.ksic_prov
        from ta t left join fam f on f.cc=t.cc and f.appl_no=t.appl_no left join fx_tab x on x.cc=t.cc and x.appl_no=t.appl_no) to '{out}.part' (format parquet, compression zstd)""")
    os.replace(out + ".part", out)
    s = con.sql(f"""select '{arm}' arm, o.cc, count(*) n, sum((o."method"='family')::int) n_family,
         avg((o.ksic_txt is distinct from t.ksic_txt)::int) changed_vs_stored, sum((o.ksic_txt is distinct from t.ksic_txt)::int) n_changed,
         avg((o.ksic_txt=o.ksic_prov)::int) agree5, avg(("method" is distinct from t.m)::int) method_changed
         from read_parquet('{out}') o join ta t using(cc, appl_no) group by 1,2 order by 2""").df()
    s2 = con.sql(f"select count(*), count(distinct cc||'|'||appl_no), sum((ksic_txt is null)::int) from read_parquet('{out}')").fetchone()
    assert s2[0] == s2[1] == sum(r[1] for r in R["ta_rows"]) and s2[2] == 0, s2
    # 코드 정합: 배정 KSIC가 그 문서 코드 안
    bad = con.sql(f"select count(*) from read_parquet('{out}') o join km on km.ksic=o.ksic_txt where km.code<>o.code").fetchone()[0]; assert bad == 0
    stats.append(s); log(arm, s.to_string())
S = pd.concat(stats); S.to_csv(W + "/res/s1_arm_stats.csv", index=False, encoding="utf-8-sig")
# r8 = 저장값 100% (P0)
R["r8_equals_stored"] = con.sql(f"""select avg((o.ksic_txt=t.ksic_txt)::int), avg((o."method"=t.m)::int), count(*) from read_parquet('{W}/cache/txt_assign_r8.parquet') o join ta t using(cc, appl_no)""").fetchone()
log("r8 재현", R["r8_equals_stored"])
# C10 alt_fix k_txt_* 와 대조(풀 안 해외 문서)
R["vs_c10_altfix"] = {}
for arm in ("reg", "pub", "pure"):
    R["vs_c10_altfix"][arm] = con.sql(f"""select o.cc, count(*) n, avg((o.ksic_txt is not distinct from a.k_txt_{arm})::int) same
        from read_parquet('{W}/cache/txt_assign_{arm}.parquet') o join read_parquet('{AF}') a using(cc, appl_no) where o.cc<>'KR' group by 1 order by 1""").fetchall()
R["vs_c10_altfix"]["stored"] = con.sql(f"""select o.cc, count(*) n, avg((o.ksic_txt is not distinct from a.k_txt_stored)::int) same
        from read_parquet('{W}/cache/txt_assign_r8.parquet') o join read_parquet('{AF}') a using(cc, appl_no) where o.cc<>'KR' group by 1 order by 1""").fetchall()
R["altfix_coverage_of_pool_foreign"] = con.sql(f"""select o.cc, count(*), count(a.appl_no) from (select cc, appl_no from ta where cc<>'KR') o
        left join read_parquet('{AF}') a using(cc, appl_no) group by 1 order by 1""").fetchall()
log("C10 대조", R["vs_c10_altfix"])
json.dump(R, open(W + "/res/s1_txt_assign.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("end")
