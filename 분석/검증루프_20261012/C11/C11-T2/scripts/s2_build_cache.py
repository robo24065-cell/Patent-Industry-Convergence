# -*- coding: utf-8 -*-
r"""C11-T2 s2 (R9 s02 사본, 경로만 교체): 팔별 TXT 셀 패널 캐시. 동결 base_rows(R3/cache_pub)의 ksic를 txt_assign_{arm}으로 바꾸고(27코드 밖 제외)
동결 빌더 build_panel.py·build_feat2.py를 수정 없이 실행. mg 캐시(TXT_R9/cache_mg_pub)는 이 폴더로 복사(해시 대조).
사용: python s2_build_cache.py <arm>   (arm = r8 | reg | pub | pure | mgcopy)"""
import os, sys, time, subprocess, shutil, hashlib, glob
sys.dont_write_bytecode = True
import duckdb
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측"; W = BASE + "/분석/검증루프_20261012/C11/C11-T2"
SRC = T3 + "/R3/cache_pub/base_rows.parquet"
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
arm = sys.argv[1]; t = time.time()
if arm == "mgcopy":
    MGS = T3 + "/TXT_R9/cache_mg_pub"; MGD = W + "/cache/mg_pub"; os.makedirs(MGD, exist_ok=True)
    for f in sorted(os.listdir(MGS)):
        d = os.path.join(MGD, f)
        if not os.path.exists(d): shutil.copyfile(os.path.join(MGS, f), d)
        assert h(d) == h(os.path.join(MGS, f))
    print("mg copy", len(os.listdir(MGD))); sys.exit(0)
TA = f"{W}/cache/txt_assign_{arm}.parquet"
CACHE = f"{W}/cache/txt_{arm}"; os.makedirs(CACHE, exist_ok=True)
out = CACHE + "/base_rows.parquet"
if not os.path.exists(out):
    con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
    con.execute(f"set temp_directory='{W}/cache/_tmp'")
    cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{SRC}')").fetchall()]
    sel = ', '.join('a.ksic_txt as ksic' if c == 'ksic' else f'b.{c}' for c in cols)
    con.execute(f"copy (select {sel} from read_parquet('{SRC}') b join read_parquet('{TA}') a using(cc, appl_no)) to '{out}.part' (format parquet, compression zstd)")
    os.replace(out + ".part", out)
    st = con.sql(f"""select (select count(*) from read_parquet('{SRC}')) n_src, count(*) n_out, count(distinct ksic) n_ksic,
                    count(distinct cc||appl_no) n_rec_out from read_parquet('{out}')""").df()
    print(st.to_string(), flush=True); st.to_csv(CACHE + "/base_stats.csv", index=False)
    print("base done", round(time.time() - t)); sys.exit(0)
env = {**os.environ, 'T3_CACHE': CACHE, 'T3_MAXYEAR': '2021', 'T3_T0MAX': '2018', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
shutil.copyfile(T3 + "/build_panel.py", CACHE + "/build_panel.py")
assert h(T3 + "/build_panel.py") == h(CACHE + "/build_panel.py")
for step, f in [('build_panel.py', 'panel_ccy.parquet'), ('build_feat2.py', 'feat2.parquet')]:
    if not os.path.exists(os.path.join(CACHE, f)):
        r = subprocess.run([sys.executable, os.path.join(T3, step)], cwd=CACHE, env=env)
        assert r.returncode == 0, step
        print(step, 'done', round(time.time() - t), flush=True)
print("all built", arm, round(time.time() - t))
