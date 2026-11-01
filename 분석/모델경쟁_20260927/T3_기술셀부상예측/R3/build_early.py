"""공개연도 변형 보조 피처: '조기 공개 신호' — t0년 말까지 공개된 문서를 출원연도로 다시 색인.
 kr_early_n  = KR 출원연도 ≥ t0-1 이면서 공개연도 ≤ t0 인 출원 수(조기공개·조기등록분, log1p)
 kr_early_sh = kr_early_n(원값) / KR 공개연도 t0-2..t0 문서 수
 kr_appc_mom = log((KR 출원 t0-2..t0 & 공개 ≤t0)+1)/((KR 출원 t0-5..t0-3 & 공개 ≤t0)+1)  (출원연도 모멘텀의 실시간판; 절단은 모든 셀 공통)
 fx_appc_mom, fx_early_sh = 해외 4개국 같은 정의
모두 공개연도 ≤ t0 문서만 사용(🔒 캐시 자체가 공개연도 ≤2021).
"""
import os, duckdb, time
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'cache_pub')
T0MAX = int(os.environ.get('T3R3_T0MAX', '2018'))
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
src = open(os.path.join(HERE, '..', 'build_panel.py'), encoding='utf-8').read()
exec(src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')])
bp = os.path.join(CACHE, 'base_rows.parquet').replace(os.sep, '/')
con.execute(f"""create temp table a as select ksic, mgn(ipc_raw) mg, cc, appl_no, min(year) py, min(appl_year) ay
  from read_parquet('{bp}') where mgn(ipc_raw) is not null group by 1,2,3,4""")
out = []
for t0 in range(2005, T0MAX + 1):
    q = f"""select ksic||'|'||mg as cell, {t0} as t0,
      count(*) filter (where cc='KR' and ay>={t0-1} and py<={t0}) kr_e,
      count(*) filter (where cc='KR' and py between {t0-2} and {t0}) kr_w,
      count(*) filter (where cc='KR' and ay between {t0-2} and {t0} and py<={t0}) kr_a0,
      count(*) filter (where cc='KR' and ay between {t0-5} and {t0-3} and py<={t0}) kr_a1,
      count(*) filter (where cc<>'KR' and ay>={t0-1} and py<={t0}) fx_e,
      count(*) filter (where cc<>'KR' and py between {t0-2} and {t0}) fx_w,
      count(*) filter (where cc<>'KR' and ay between {t0-2} and {t0} and py<={t0}) fx_a0,
      count(*) filter (where cc<>'KR' and ay between {t0-5} and {t0-3} and py<={t0}) fx_a1
      from a where py<={t0} and py>={t0-8} group by 1"""
    out.append(con.sql(q).df()); print(t0, round(time.time() - t, 1), flush=True)
import pandas as pd, numpy as np
d = pd.concat(out, ignore_index=True)
d['kr_early_n'] = np.log1p(d.kr_e); d['kr_early_sh'] = d.kr_e / np.maximum(d.kr_w, 1)
d['kr_appc_mom'] = np.log((d.kr_a0 + 1) / (d.kr_a1 + 1)); d['fx_appc_mom'] = np.log((d.fx_a0 + 1) / (d.fx_a1 + 1))
d['fx_early_sh'] = d.fx_e / np.maximum(d.fx_w, 1)
d[['cell', 't0', 'kr_early_n', 'kr_early_sh', 'kr_appc_mom', 'fx_appc_mom', 'fx_early_sh']].to_parquet(os.path.join(CACHE, 'early_t0.parquet'))
print(d.groupby('t0')[['kr_early_sh', 'kr_appc_mom']].mean())
