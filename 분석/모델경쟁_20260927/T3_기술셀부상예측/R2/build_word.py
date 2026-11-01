"""T3 2라운드: 제목 '뜨는 용어' 노출도. 원점 t0 마다 전체 KR(또는 US) 제목 단어의 3년 모멘텀을 계산하고
(t0-2..t0 출원 vs t0-5..t0-3, 창 내 30건 이상 단어만), 각 셀의 최근 3년 출원 제목이 담은 단어 모멘텀의 평균·상위값을 셀 피처로.
모두 출원연도 ≤ t0 자료만 사용. 산출: cache_dev/word_expo.parquet
"""
import os, sys, time, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MAXY = int(os.environ.get('T3R2_MAXYEAR', '2019'))
if MAXY >= 2020 and os.environ.get('T3R2_FINAL_UNLOCK') != '1': raise RuntimeError('🔒')
CACHE = os.path.join(HERE, os.environ.get('T3R2_CACHE', 'cache_dev')).replace('\\', '/')
T0MAX = int(os.environ.get('T3R2_T0MAX', '2016')); T0MIN = int(os.environ.get('T3R2_T0MIN', '2005'))
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
src = open(os.path.join(HERE, '..', 'build_panel.py'), encoding='utf-8').read()
exec(src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')])
con.execute(rf"""create temp table a as select ksic, mgn(ipc_raw) mg, cc, year, appl_no, any_value(title) title
  from read_parquet('{CACHE}/base_rows.parquet') where cc in ('KR','US') and mgn(ipc_raw) is not null and year <= {MAXY} and title is not null group by all""")
# 출원 단위 단어(중복 제거): KR 한글·영문 2자 이상, US 영문 3자 이상(불용어 일부 제외)
STOP = "('the','and','for','with','method','apparatus','device','system','same','thereof','using','having','its','from','manufacturing','use','based')"
con.execute(rf"""create temp table tw as select distinct cc, appl_no, year, w from (
  select cc, appl_no, year, unnest(string_split(regexp_replace(lower(title),'[^0-9a-z가-힣 ]',' ','g'),' ')) w from (select distinct cc, appl_no, year, title from a))
  where (cc='KR' and length(w)>=2) or (cc='US' and length(w)>=3 and w not in {STOP})""")
con.execute("create temp table wy as select cc, w, year, count(*) n from tw group by all")
con.execute("create temp table cellapp as select distinct ksic, mg, cc, appl_no, year from a")
print('tokens', con.sql("select cc, count(*) from tw group by cc").fetchall(), round(time.time() - t, 1), flush=True)
out = []
for t0 in range(T0MIN, T0MAX + 1):
    con.execute(f"""create or replace temp table wm as select cc, w,
        ln((sum(n) filter (where year between {t0-2} and {t0}) + 1)/(coalesce(sum(n) filter (where year between {t0-5} and {t0-3}),0) + 1)) m
      from wy where year between {t0-5} and {t0} group by all having coalesce(sum(n) filter (where year between {t0-2} and {t0}),0) >= 30""")
    q = con.sql(f"""with am as (select t.cc, t.appl_no, avg(m) am, max(m) mx from tw t join wm using(cc, w) where t.year between {t0-2} and {t0} group by all),
        thr as (select cc, quantile_cont(m, 0.9) q from wm group by cc)
      select c.ksic, c.mg, {t0} t0,
        avg(am) filter (where c.cc='KR') kw_expo_mean, avg(mx) filter (where c.cc='KR') kw_expo_max,
        avg(case when mx >= thr.q then 1.0 else 0.0 end) filter (where c.cc='KR') kw_hot_sh,
        avg(am) filter (where c.cc='US') us_kw_expo_mean, avg(case when mx >= thr.q then 1.0 else 0.0 end) filter (where c.cc='US') us_kw_hot_sh
      from cellapp c join am using(cc, appl_no) join thr using(cc) where c.year between {t0-2} and {t0} group by c.ksic, c.mg""").df()
    out.append(q); print(t0, len(q), round(time.time() - t, 1), flush=True)
import pandas as pd
pd.concat(out, ignore_index=True).to_parquet(f'{CACHE}/word_expo.parquet')
print('done', round(time.time() - t, 1))
