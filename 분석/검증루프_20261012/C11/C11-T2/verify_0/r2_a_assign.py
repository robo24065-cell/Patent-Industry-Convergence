# -*- coding: utf-8 -*-
# 2차 반박 검증 A: reg txt_assign을 window 함수(row_number)로 독립 재구성 → 작업자 cache/txt_assign_reg.parquet와 문서 단위 대조.
# 또 REG 연결 쌍의 제공 KSIC 일치율(연결 품질)을 R8 결함 연결과 비교. 읽기만, 쓰기는 verify_0만.
import duckdb, json, time, pandas as pd
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
T3 = B + '/분석/모델경쟁_20260927/T3_기술셀부상예측'
W = B + '/분석/검증루프_20261012/C11/C11-T2'; V = W + '/verify_0'
C10 = B + '/분석/검증루프_20261012/C10/C10-T3'
t = time.time()
con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{V}/_r2tmp'")
km = pd.read_csv(T3 + '/ESSENTIAL_R7/conc_ksic_map.csv', dtype=str)[['ksic', 'code']]
con.register('km', km)
con.execute("create table multi as select code from km group by code having count(*)>1")
con.execute(f"create table ta as select cc, appl_no, code, ksic_txt, \"method\" m from read_parquet('{T3}/TXT_R9/txt_assign.parquet')")
con.execute(f"create table rc as select cc, appl_no, sg from read_parquet('{T3}/ESSENTIAL_R7/rec_conc.parquet')")
con.execute("create table fx as select t.cc, t.appl_no, t.code, rc.sg from ta t left join rc using(cc, appl_no) where t.cc<>'KR' and t.code in (select code from multi)")
con.execute("create table kr as select appl_no, code, ksic_txt, m from ta where cc='KR'")
# 패밀리(REG): 같은 코드 KR 문서의 TXT 최빈, 동률은 KSIC 숫자 작은 것
con.execute(f"""create table fam as select cc, appl_no, k from (
  select cc, appl_no, k, row_number() over (partition by cc, appl_no order by n desc, cast(k as int) asc) rn from (
    select fx.cc, fx.appl_no, kr.ksic_txt k, count(*) n
    from (select distinct cc, appl_no, kr_appl_no from read_parquet('{C10}/cache/links.parquet') where link='REG') l
    join fx on fx.cc=l.cc and fx.appl_no=l.appl_no join kr on kr.appl_no=l.kr_appl_no and kr.code=fx.code group by 1,2,3)) where rn=1""")
# TXT-IPC 표: KR text 문서만
con.execute("create table krk as select kr.code, kr.ksic_txt k, rc.sg, split_part(rc.sg,'/',1) mg, left(rc.sg,4) sc from kr join rc on rc.cc='KR' and rc.appl_no=kr.appl_no where kr.m='text'")
for lvl in ('sg', 'mg', 'sc'):
    con.execute(f"""create table t_{lvl} as select code, key_, k from (
      select code, key_, k, sum(n) over (partition by code, key_) tot, row_number() over (partition by code, key_ order by n desc, cast(k as int)) rn from (
        select code, {lvl} key_, k, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by 1,2,3)) where rn=1 and tot>=3""")
con.execute("""create table t_code as select code, k from (select code, k, row_number() over (partition by code order by n desc, cast(k as int)) rn
   from (select code, k, count(*) n from krk group by 1,2)) where rn=1""")
con.execute("""create table mine as select fx.cc, fx.appl_no, coalesce(f.k, a.k, b.k, c.k, d.k) k,
  case when f.k is not null then 'family' when a.k is not null then 'ipc_sg' when b.k is not null then 'ipc_mg' when c.k is not null then 'ipc_sc' else 'code_mode' end m
  from fx left join fam f using(cc, appl_no) left join t_sg a on a.code=fx.code and a.key_=fx.sg
  left join t_mg b on b.code=fx.code and b.key_=split_part(fx.sg,'/',1) left join t_sc c on c.code=fx.code and c.key_=left(fx.sg,4)
  left join t_code d on d.code=fx.code""")
R = {}
R['fx_n'] = con.sql("select count(*) from fx").fetchone()[0]
R['cmp_fx_vs_worker_reg'] = con.sql(f"""select m.cc, count(*), sum((m.k is distinct from w.ksic_txt)::int) diff_k, sum((m.m is distinct from w."method")::int) diff_m,
   sum((m.m='family')::int) n_family from mine m join read_parquet('{W}/cache/txt_assign_reg.parquet') w using(cc, appl_no) group by 1 order by 1""").fetchall()
R['cmp_nonfx_vs_stored'] = con.sql(f"""select count(*), sum((w.ksic_txt is distinct from t.ksic_txt)::int) from read_parquet('{W}/cache/txt_assign_reg.parquet') w join ta t using(cc, appl_no)
   where not (t.cc<>'KR' and t.code in (select code from multi))""").fetchall()
R['worker_rows'] = con.sql(f"select count(*), count(distinct cc||'|'||appl_no) from read_parquet('{W}/cache/txt_assign_reg.parquet')").fetchall()
R['changed_vs_stored_mine'] = con.sql("select m.cc, sum((m.k is distinct from t.ksic_txt)::int) from mine m join ta t using(cc, appl_no) group by 1 order by 1").fetchall()
# 연결 품질: 연결 쌍의 제공 KSIC(base_rows ksic) 일치율, 링크별
con.execute(f"create table br as select cc, appl_no, ksic from read_parquet('{T3}/R3/cache_pub/base_rows.parquet')")
R['link_ksic_agree'] = con.sql(f"""select l.link, l.cc, count(*) n_pairs_in_pool, avg((a.ksic=b.ksic)::int) agree
   from read_parquet('{C10}/cache/links.parquet') l join br a on a.cc=l.cc and a.appl_no=l.appl_no join br b on b.cc='KR' and b.appl_no=l.kr_appl_no
   group by 1,2 order by 1,2""").fetchall()
R['secs'] = round(time.time() - t)
json.dump(R, open(V + '/r2_a.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, default=str))
