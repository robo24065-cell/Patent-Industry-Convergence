# -*- coding: utf-8 -*-
"""R8 s01: 순환 없는 검증용 셀 패널 캐시. 동결 base_rows(R3/cache_pub, 공개연도 ≤2021)를 읽기만 하고 변형본을 NONCIRC_R8/cache_* 에 쓴다.
  --alt mg    : KSIC 를 'ALL' 로 지운다 → 셀 = IPC 메인그룹(라벨 무관 단위). 레코드는 출원 단위 1건이므로 이중 집계 없음.
  --alt pcode : 제공 KSIC 를 그 KSIC 의 연계표 코드(R7 conc_ksic_map.csv)로 바꾼다 → 제공 라벨 배정 × 27코드 해상도(P27).
이후 1라운드 빌더 build_panel.py · build_feat2.py(수정 없음)를 같은 규칙으로 실행(cwd = 캐시 폴더, build_panel.py 원본 복사 후 해시 대조)."""
import os, sys, time, argparse, subprocess, shutil, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
import duckdb, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--alt', required=True, choices=['mg', 'pcode']); a = ap.parse_args()
SRC = os.path.join(T3, 'R3', 'cache_pub', 'base_rows.parquet').replace(os.sep, '/'); MAXY, T0MAX = 2021, 2018
CACHE = os.path.join(HERE, f'cache_{a.alt}_pub'); os.makedirs(CACHE, exist_ok=True)
t = time.time()
out = os.path.join(CACHE, 'base_rows.parquet').replace(os.sep, '/')
if not os.path.exists(out):
    con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{SRC}')").fetchall()]
    if a.alt == 'mg':
        sel = ', '.join("'ALL' as ksic" if c == 'ksic' else c for c in cols)
        con.execute(f"copy (select {sel} from read_parquet('{SRC}')) to '{out}' (format parquet, compression zstd)")
    else:
        km = pd.read_csv(os.path.join(T3, 'ESSENTIAL_R7', 'conc_ksic_map.csv'), dtype=str)[['ksic', 'code']]
        con.register('km', km)
        sel = ', '.join('km.code as ksic' if c == 'ksic' else f'b.{c}' for c in cols)
        con.execute(f"copy (select {sel} from read_parquet('{SRC}') b join km on b.ksic = km.ksic) to '{out}' (format parquet, compression zstd)")
    st = con.sql(f"""select (select count(*) from read_parquet('{SRC}')) n_src, count(*) n_out, count(distinct ksic) n_ksic,
                    count(distinct cc||appl_no) n_rec_out from read_parquet('{out}')""").df()
    print(st.to_string(), flush=True); st.to_csv(os.path.join(CACHE, 'base_stats.csv'), index=False)
env = {**os.environ, 'T3_CACHE': CACHE, 'T3_MAXYEAR': str(MAXY), 'T3_T0MAX': str(T0MAX), 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
shutil.copyfile(os.path.join(T3, 'build_panel.py'), os.path.join(CACHE, 'build_panel.py'))
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert h(os.path.join(T3, 'build_panel.py')) == h(os.path.join(CACHE, 'build_panel.py'))
for step, f in [('build_panel.py', 'panel_ccy.parquet'), ('build_feat2.py', 'feat2.parquet')]:
    if not os.path.exists(os.path.join(CACHE, f)):
        r = subprocess.run([sys.executable, os.path.join(T3, step)], cwd=CACHE, env=env)
        assert r.returncode == 0, step
        print(step, 'done', round(time.time() - t), flush=True)
print('total', round(time.time() - t))
