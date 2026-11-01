# -*- coding: utf-8 -*-
r"""C10-T3 s4: (a) R11 foreign_public FREE 팔이 결함 family 경로를 썼는지 캐시로 확인, (b) T3 풀(R3 base_rows) 해외 레코드 라벨 변화 비중,
(c) 수정판 T3 하이브리드 캐시(base_rows → 동결 build_panel·build_feat2) 생성. c5_R10 c3_build_cache.py와 같은 규칙, 라벨 열만 교체.
  사용: python s4_t3_prep.py check | build <arm> (arm = free_reg | txt_reg | free_pub | txt_pub)  (build는 단계마다 끊어 9분 제한 대응)
읽기만: R3/cache_pub/base_rows, c5_R10/alt_assign, C10-T3/cache/alt_fix_*, R11 cache/fdoc.parquet
쓰기: C10-T3/cache/t3_<arm>/, res/s4_*.json|csv"""
import os, sys, json, time, subprocess, shutil, hashlib
sys.dont_write_bytecode = True
import duckdb, pandas as pd
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
W = BASE + "/분석/검증루프_20261012/C10/C10-T3"
T3 = MC + "/T3_기술셀부상예측"
SRC = T3 + "/R3/cache_pub/base_rows.parquet"
A0 = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"
AF = W + "/cache/alt_fix_*.parquet"
FDOC = MC + "/_shared_report/label_origin_R10/foreign_public_R11/cache/fdoc.parquet"
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
mode = sys.argv[1]
con = duckdb.connect()
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
if mode == "check":
    R = {}
    # (a) R11 fdoc: m_free='family'인 행의 k_free가 결함 상속값(저장 alt k_free)인지, 순수 IPC 재배정값(k_free_pure)인지
    R["r11_fdoc_family_rows"] = con.sql(f"""select f.cc, count(*) n, avg((f.k_free is not distinct from a.k_free_pure)::int) eq_pure,
        avg((f.k_free is not distinct from a.k_free_stored)::int) eq_defect_stored,
        avg((f.k_txt is not distinct from a.k_txt_pure)::int) txt_eq_pure, avg((f.k_txt is not distinct from a.k_txt_stored)::int) txt_eq_stored
        from read_parquet('{FDOC}') f join read_parquet('{AF}') a using(cc, appl_no) where f.m_free='family' group by 1 order by 1""").fetchall()
    R["r11_fdoc_all_rows_eq_pure"] = con.sql(f"""select avg((f.k_free is not distinct from a.k_free_pure)::int), avg((f.k_txt is not distinct from a.k_txt_pure)::int), count(*)
        from read_parquet('{FDOC}') f join read_parquet('{AF}') a using(cc, appl_no)""").fetchone()
    log("R11", R)
    # (b) T3 풀 해외 레코드의 라벨 변화
    con.execute(f"""create table b as select b.cc, b.appl_no, b.ksic prov, a.k_free_stored, a.k_free_reg, a.k_free_pub, a.k_free_pure,
        a.k_txt_stored, a.k_txt_reg, a.k_txt_pub, a.k_txt_pure from read_parquet('{SRC}') b left join read_parquet('{AF}') a using(cc, appl_no) where b.cc<>'KR'""")
    R["pool_rows_dup_check"] = con.sql("select count(*), count(distinct cc||appl_no) from b").fetchone()
    rows = []
    for lab in ("free", "txt"):
        for arm in ("reg", "pub", "pure"):
            d = con.sql(f"""select cc, count(*) n, avg((k_{lab}_stored is not null)::int) kept_stored, avg((k_{lab}_{arm} is not null)::int) kept_arm,
                avg((k_{lab}_{arm} is distinct from k_{lab}_stored)::int) changed,
                avg(coalesce(k_{lab}_stored<>prov, true)::int) hyb_differs_from_prov,
                avg(coalesce(k_{lab}_stored=prov,false)::int) agree_stored, avg(coalesce(k_{lab}_{arm}=prov,false)::int) agree_arm
                from b group by rollup(cc) order by cc""").df()
            d.insert(0, "arm", arm); d.insert(0, "lab", lab); rows.append(d)
    T = pd.concat(rows); T.to_csv(W + "/res/s4_t3pool_change.csv", index=False, encoding="utf-8-sig"); print(T.round(4).to_string())
    json.dump(R, open(W + "/res/s4_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    sys.exit(0)
if mode == "build":
    arm = sys.argv[2]; lab, link = arm.split("_")
    CACHE = W + f"/cache/t3_{arm}"; os.makedirs(CACHE + "/_tmp", exist_ok=True)
    out = CACHE + "/base_rows.parquet"
    if not os.path.exists(out):
        cols = [r[0] for r in con.sql(f"describe select * from read_parquet('{SRC}')").fetchall()]
        sel = ", ".join("case when b.cc='KR' then b.ksic else x.k end as ksic" if c == "ksic" else f"b.{c}" for c in cols)
        con.execute(f"""copy (select {sel} from read_parquet('{SRC}') b left join (select cc, appl_no, k_{lab}_{link} k from read_parquet('{AF}') where cc<>'KR') x
                     on x.cc=b.cc and x.appl_no=b.appl_no where b.cc='KR' or x.k is not null) to '{out}.part' (format parquet, compression zstd)""")
        os.replace(out + ".part", out)
        st = con.sql(f"""select s.cc, s.n_src, o.n_out from (select cc, count(*) n_src from read_parquet('{SRC}') group by 1) s
            join (select cc, count(*) n_out from read_parquet('{out}') group by 1) o using(cc) order by 1""").df()
        st.to_csv(CACHE + "/base_stats.csv", index=False); log(st.to_string()); con.close(); sys.exit(0)
    con.close()
    env = {**os.environ, "T3_CACHE": CACHE, "T3_MAXYEAR": "2021", "T3_T0MAX": "2018", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    shutil.copyfile(T3 + "/build_panel.py", CACHE + "/build_panel.py")
    h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    assert h(T3 + "/build_panel.py") == h(CACHE + "/build_panel.py")
    for step, f in [("build_panel.py", "panel_ccy.parquet"), ("build_feat2.py", "feat2.parquet")]:
        if not os.path.exists(os.path.join(CACHE, f)):
            r = subprocess.run([sys.executable, os.path.join(T3, step)], cwd=CACHE, env=env); assert r.returncode == 0, step
            log(step, "done"); sys.exit(0)
    log("all built")
