# -*- coding: utf-8 -*-
r"""C15-T3 s2: hyb_pkr 수정판 T3 캐시(c5_R10 c3_build_cache.py 규칙, 라벨 열만 k_pkr_<arm>).
  python s2_build.py <arm>   (arm = reg | pure)  — 단계마다 끊어서 실행(9분 제한): base_rows → build_panel → build_feat2
읽기만: R3/cache_pub/base_rows.parquet, T3 build_panel.py·build_feat2.py(동결), C15-T3/cache/pkr_alt_*.parquet
쓰기: C15-T3/cache/t3_pkr_<arm>/"""
import os, sys, time, subprocess, shutil, hashlib
sys.dont_write_bytecode = True
import duckdb
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C15/C15-T3"
T3 = MC + "/T3_기술셀부상예측"
SRC = T3 + "/R3/cache_pub/base_rows.parquet"
AF = W + "/cache/pkr_alt_*.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
arm = sys.argv[1]; assert arm in ("reg", "pure", "r8")
CACHE = W + f"/cache/t3_pkr_{arm}"; os.makedirs(CACHE + "/_tmp", exist_ok=True)
out = CACHE + "/base_rows.parquet"
if not os.path.exists(out):
    con = duckdb.connect()
    con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{CACHE}/_tmp'")
    cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{SRC}')").fetchall()]
    sel = ", ".join("case when b.cc='KR' then b.ksic else x.k end as ksic" if c == "ksic" else f"b.{c}" for c in cols)
    con.execute(f"""copy (select {sel} from read_parquet('{SRC}') b left join (select cc, appl_no, k_{arm} k from read_parquet('{AF}')) x
                 on x.cc=b.cc and x.appl_no=b.appl_no where b.cc='KR' or x.k is not null) to '{out}.part' (format parquet, compression zstd)""")
    os.replace(out + ".part", out)
    st = con.sql(f"""select s.cc, s.n_src, o.n_out, o.n_out*1.0/s.n_src keep from (select cc, count(*) n_src from read_parquet('{SRC}') group by 1) s
        join (select cc, count(*) n_out from read_parquet('{out}') group by 1) o using(cc) order by 1""").df()
    st.to_csv(CACHE + "/base_stats.csv", index=False); log(st.to_string()); con.close(); sys.exit(0)
env = {**os.environ, "T3_CACHE": CACHE, "T3_MAXYEAR": "2021", "T3_T0MAX": "2018", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
shutil.copyfile(T3 + "/build_panel.py", CACHE + "/build_panel.py")
h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
assert h(T3 + "/build_panel.py") == h(CACHE + "/build_panel.py")
for step, f in [("build_panel.py", "panel_ccy.parquet"), ("build_feat2.py", "feat2.parquet")]:
    if not os.path.exists(os.path.join(CACHE, f)):
        r = subprocess.run([sys.executable, os.path.join(T3, step)], cwd=CACHE, env=env); assert r.returncode == 0, step
        log(step, "done"); sys.exit(0)
log("all built")
