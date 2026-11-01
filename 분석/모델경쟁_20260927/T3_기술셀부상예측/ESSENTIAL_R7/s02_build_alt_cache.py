# -*- coding: utf-8 -*-
"""R7 s02: 대체·5개국 제거 시험용 셀 패널 캐시를 만든다(동결 캐시의 base_rows 를 읽기만 하고 변형본을 ESSENTIAL_R7/cache_* 에 쓴다).
  --alt conc : 각 레코드의 KSIC 를 공식 연계표 배정 코드(rec_conc.parquet)로 바꾼다. 50개 KSIC 가 걸치는 27개 코드 밖으로 배정된 레코드는 뺀다.
               같은 출원이 여러 제공 KSIC 파일에 있던 경우 연계표 아래에서는 한 코드로 합쳐진다(build_panel 의 출원 단위 dedupe).
  --alt concm: s01b 다중 배정(점유율 ≥0.10 인 27개 코드 전부)으로 레코드를 복제 배정(rec_conc_multi.parquet).
  --alt kr   : KR 레코드만 남긴다(해외 4개국 레코드 유래 정보 전부 제거). KSIC 는 제공 라벨 그대로.
  --var pub  : 원본 R3/cache_pub/base_rows.parquet (공개연도 ≤2021)  → feat2 t0 ≤2018
  --var app  : 원본 R2/cache_dev/base_rows.parquet (출원연도 ≤2019)  → feat2 t0 ≤2016
이후 1라운드 빌더 build_panel.py · build_feat2.py(수정 없음)를 같은 규칙으로 실행한다(cwd = T3 폴더, T3_CACHE = 절대경로)."""
import os, sys, json, time, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE); sys.path.insert(0, HERE)
import duckdb, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--alt', required=True, choices=['conc', 'concm', 'kr']); ap.add_argument('--one', action='store_true'); ap.add_argument('--var', required=True, choices=['pub', 'app', 'pubF'])
a = ap.parse_args()
SRC = {'pub': os.path.join(T3, 'R3', 'cache_pub', 'base_rows.parquet'), 'app': os.path.join(T3, 'R2', 'cache_dev', 'base_rows.parquet'),
       'pubF': os.path.join(T3, 'R3', 'cache_pub_final', 'base_rows.parquet')}[a.var]   # pubF = 사후 최종 분석용(공개연도 ≤2024)
MAXY, T0MAX = {'pub': (2021, 2018), 'app': (2019, 2016), 'pubF': (2024, 2021)}[a.var]
CACHE = os.path.join(HERE, f'cache_{a.alt}_{a.var}'); os.makedirs(CACHE, exist_ok=True)
t = time.time()
out = os.path.join(CACHE, 'base_rows.parquet').replace(os.sep, '/')
if not os.path.exists(out):
    con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    src = SRC.replace(os.sep, '/')
    cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{src}')").fetchall()]
    if a.alt == 'kr':
        con.execute(f"copy (select * from read_parquet('{src}') where cc='KR') to '{out}' (format parquet, compression zstd)")
    else:
        km = pd.read_csv(os.path.join(HERE, 'conc_ksic_map.csv'), dtype=str); codes50 = sorted(km.code.unique())
        c50 = "(" + ",".join(f"'{c}'" for c in codes50) + ")"
        rc = os.path.join(HERE, 'rec_conc.parquet' if a.alt == 'conc' else 'rec_conc_multi.parquet').replace(os.sep, '/')
        sel = ', '.join('r.code as ksic' if c == 'ksic' else f'b.{c}' for c in cols)
        con.execute(f"""copy (select {sel} from read_parquet('{src}') b join read_parquet('{rc}') r on b.cc = r.cc and b.appl_no = r.appl_no
                     where r.code in {c50}) to '{out}' (format parquet, compression zstd)""")
    st = con.sql(f"""select (select count(*) from read_parquet('{src}')) n_src, count(*) n_out, count(distinct ksic) n_ksic,
                    count(distinct cc||appl_no) n_rec_out, (select count(distinct cc||appl_no) from read_parquet('{src}')) n_rec_src from read_parquet('{out}')""").df()
    print(st.to_string(), flush=True)
    st.to_csv(os.path.join(CACHE, 'base_stats.csv'), index=False)
    print('base', round(time.time() - t), flush=True)
    if a.one: sys.exit(0)
env = {**os.environ, 'T3_CACHE': CACHE, 'T3_MAXYEAR': str(MAXY), 'T3_T0MAX': str(T0MAX), 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
# duckdb 임시 파일(.tmp)이 동결 트랙 폴더에 생기지 않도록 cwd = 이 캐시 폴더. build_feat2.py 가 cwd 의 build_panel.py 를 읽으므로 원본을 그대로 복사(해시 대조)
import shutil, hashlib
shutil.copyfile(os.path.join(T3, 'build_panel.py'), os.path.join(CACHE, 'build_panel.py'))
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert h(os.path.join(T3, 'build_panel.py')) == h(os.path.join(CACHE, 'build_panel.py'))
for step, f in [('build_panel.py', 'panel_ccy.parquet'), ('build_feat2.py', 'feat2.parquet')]:
    if not os.path.exists(os.path.join(CACHE, f)):
        r = subprocess.run([sys.executable, os.path.join(T3, step)], cwd=CACHE, env=env)
        assert r.returncode == 0, step
        print(step, 'done', round(time.time() - t), flush=True)
        if a.one: sys.exit(0)
print('total', round(time.time() - t))
