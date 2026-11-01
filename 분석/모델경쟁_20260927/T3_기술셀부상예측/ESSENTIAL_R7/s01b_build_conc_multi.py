# -*- coding: utf-8 -*-
"""R7 s01b: 연계표 '관대한' 다중 배정(사전 고정, 2026-10-05) — argmax 단일 배정이 IPC 하나를 산업 하나에만 보내 셀 커버리지가 좁아지는 문제를
덜어 주는 변형. 레코드의 매칭 수준(서브그룹→메인그룹→서브클래스, s01 과 같은 순서)에서 연계표 점유율 p ≥ 0.10 인 50개-틀(27개 코드) 코드
전부에 배정한다(한 레코드가 여러 코드 셀에 들어갈 수 있음 = 제공 데이터의 해외 레코드 다중 파일 등장과 같은 방식).
출력: rec_conc_multi.parquet (cc, appl_no, code, p, lvl)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd, duckdb
import conc_lib as CL
TH = 0.10
t = time.time()
cc, _ = CL.load_concordance()
km = pd.read_csv(os.path.join(HERE, 'conc_ksic_map.csv'), dtype=str); c50 = set(km.code)
tabs = {}
for key in ['sg', 'mg', 'sc']:
    g = cc.groupby([key, 'code']).n.sum().reset_index(); g['p'] = g.n / g.groupby(key).n.transform('sum')
    tabs[key] = g[(g.p >= TH) & g.code.isin(c50)][[key, 'code', 'p']]
keys = {k: set(cc[k]) for k in ['sg', 'mg', 'sc']}
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
for k, v in tabs.items(): con.register(f't_{k}', v)
for k in ['sg', 'mg', 'sc']: con.register(f'k_{k}', pd.DataFrame({k: sorted(keys[k])}))
RC = os.path.join(HERE, 'rec_conc.parquet').replace(os.sep, '/'); OUT = os.path.join(HERE, 'rec_conc_multi.parquet').replace(os.sep, '/')
con.execute(f"""copy (
 with r as (select cc, appl_no, sg, split_part(sg,'/',1) mg, left(sg,4) sc, lvl from read_parquet('{RC}') where lvl > 0)
 select r.cc, r.appl_no, t.code, t.p, r.lvl from r join t_sg t on r.lvl = 1 and r.sg = t.sg
 union all select r.cc, r.appl_no, t.code, t.p, r.lvl from r join t_mg t on r.lvl = 2 and r.mg = t.mg
 union all select r.cc, r.appl_no, t.code, t.p, r.lvl from r join t_sc t on r.lvl = 3 and r.sc = t.sc
) to '{OUT}' (format parquet)""")
print(con.sql(f"select cc, count(*) n_assign, count(distinct cc||appl_no) n_rec, count(*)*1.0/count(distinct cc||appl_no) per_rec from read_parquet('{OUT}') group by rollup(cc)").df(), round(time.time() - t))
