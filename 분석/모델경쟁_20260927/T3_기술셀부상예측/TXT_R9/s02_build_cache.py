# -*- coding: utf-8 -*-
"""R9 s02: TXT 셀 패널 캐시(cache_txt_pub) — 동결 base_rows(R3/cache_pub) 의 ksic 를 TXT 배정(txt_assign.parquet)으로 바꾼 판.
27코드 밖 레코드는 빠진다(C27 팔과 동일). 이후 1라운드 빌더 build_panel.py·build_feat2.py(수정 없음)를 같은 규칙으로 실행(R8 s01 과 같은 방식).
또 라벨 무관 mg 캐시(NONCIRC_R8/cache_mg_pub)의 작은 파일(panel·feat)만 TXT_R9/cache_mg_pub 로 복사(동결 폴더에 쓰기가 생기지 않도록)."""
import os, sys, time, subprocess, shutil, hashlib, glob
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
import duckdb
SRC = os.path.join(T3, 'R3', 'cache_pub', 'base_rows.parquet').replace(os.sep, '/'); MAXY, T0MAX = 2021, 2018
TA = os.path.join(HERE, 'txt_assign.parquet').replace(os.sep, '/')
CACHE = os.path.join(HERE, 'cache_txt_pub'); os.makedirs(CACHE, exist_ok=True)
t = time.time()
out = os.path.join(CACHE, 'base_rows.parquet').replace(os.sep, '/')
if not os.path.exists(out):
    con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    con.execute("set temp_directory='" + os.path.join(HERE, 'duck_tmp').replace(os.sep, '/') + "'")
    cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{SRC}')").fetchall()]
    sel = ', '.join('a.ksic_txt as ksic' if c == 'ksic' else f'b.{c}' for c in cols)
    con.execute(f"copy (select {sel} from read_parquet('{SRC}') b join read_parquet('{TA}') a using(cc, appl_no)) to '{out}' (format parquet, compression zstd)")
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
# mg 캐시 작은 파일 복사(읽기 전용 원본, 해시 대조)
MGS = os.path.join(T3, 'NONCIRC_R8', 'cache_mg_pub'); MGD = os.path.join(HERE, 'cache_mg_pub'); os.makedirs(MGD, exist_ok=True)
for f in ['panel_ccy.parquet', 'feat2.parquet'] + [os.path.basename(x) for x in glob.glob(os.path.join(MGS, 'feat_20*.parquet'))]:
    d = os.path.join(MGD, f)
    if not os.path.exists(d): shutil.copyfile(os.path.join(MGS, f), d)
    assert h(d) == h(os.path.join(MGS, f))
print('mg 캐시 복사', len(os.listdir(MGD)), 'total', round(time.time() - t))
