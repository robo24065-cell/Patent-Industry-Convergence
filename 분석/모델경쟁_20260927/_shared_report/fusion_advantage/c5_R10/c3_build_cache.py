# -*- coding: utf-8 -*-
r"""c5_R10 c3: T3 하이브리드 캐시 — 원본 R3/cache_pub/base_rows.parquet(읽기만)에서 해외 레코드의 ksic만 대체 KSIC(c1 alt_assign)로 바꾼다.
KR 레코드는 그대로. 대체 라벨이 없는 해외 레코드는 뺀다. 이후 동결 빌더 build_panel.py·build_feat2.py(수정 없음, 해시 대조 사본)를 같은 규칙으로 실행.
  --alt txt|free|pkr   (txt = R9 T3 TXT 해외 배정, free = FREE-IPC, pkr = 진단)
  --one : 한 단계만 하고 끝냄(9분 제한 대응)"""
import os, sys, time, argparse, subprocess, shutil, hashlib
sys.dont_write_bytecode = True
import duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(HERE, r"..\..\..")); T3 = os.path.join(MC, "T3_기술셀부상예측")
ap = argparse.ArgumentParser(); ap.add_argument("--alt", required=True, choices=["txt", "free", "pkr"]); ap.add_argument("--one", action="store_true"); a = ap.parse_args()
SRC = os.path.join(T3, "R3", "cache_pub", "base_rows.parquet").replace("\\", "/")
ALT = os.path.join(HERE, "alt_assign", "*.parquet").replace("\\", "/")
CACHE = os.path.join(HERE, f"t3cache_hyb_{a.alt}"); os.makedirs(CACHE, exist_ok=True)
out = os.path.join(CACHE, "base_rows.parquet").replace("\\", "/")
t = time.time()
if not os.path.exists(out):
    con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{CACHE.replace(chr(92), '/')}/_tmp'")
    cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{SRC}')").fetchall()]
    sel = ", ".join("case when b.cc='KR' then b.ksic else x.k end as ksic" if c == "ksic" else f"b.{c}" for c in cols)
    con.execute(f"""copy (select {sel} from read_parquet('{SRC}') b left join (select cc, appl_no, k_{a.alt} k from read_parquet('{ALT}') where cc<>'KR') x
                 on x.cc=b.cc and x.appl_no=b.appl_no where b.cc='KR' or x.k is not null) to '{out}.part' (format parquet, compression zstd)""")
    os.replace(out + ".part", out)
    st = con.sql(f"""select s.cc, s.n_src, o.n_out, o.n_out*1.0/s.n_src keep from (select cc, count(*) n_src from read_parquet('{SRC}') group by 1) s
        join (select cc, count(*) n_out from read_parquet('{out}') group by 1) o using(cc) order by 1""").df()
    ag = con.sql(f"""select b.cc, avg((b.ksic=o.ksic)::int) same_label from read_parquet('{SRC}') b join read_parquet('{out}') o using(cc, appl_no) group by 1 order by 1""").df()
    st = st.merge(ag, on="cc"); print(st.to_string(), flush=True); st.to_csv(os.path.join(CACHE, "base_stats.csv"), index=False)
    print("base", round(time.time() - t), flush=True)
    if a.one: sys.exit(0)
env = {**os.environ, "T3_CACHE": CACHE, "T3_MAXYEAR": "2021", "T3_T0MAX": "2018", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
shutil.copyfile(os.path.join(T3, "build_panel.py"), os.path.join(CACHE, "build_panel.py"))
h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
assert h(os.path.join(T3, "build_panel.py")) == h(os.path.join(CACHE, "build_panel.py"))
for step, f in [("build_panel.py", "panel_ccy.parquet"), ("build_feat2.py", "feat2.parquet")]:
    if not os.path.exists(os.path.join(CACHE, f)):
        r = subprocess.run([sys.executable, os.path.join(T3, step)], cwd=CACHE, env=env)
        assert r.returncode == 0, step
        print(step, "done", round(time.time() - t), flush=True)
        if a.one: sys.exit(0)
print("total", round(time.time() - t))
