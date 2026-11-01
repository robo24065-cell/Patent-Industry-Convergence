# -*- coding: utf-8 -*-
r"""C19-T1 T3 변형 캐시 빌더 (C18-T2 scripts\t3\t3_build.py 사본; 바뀐 것: 쓰기 폴더 = C19-T1, 변형 조건 = JP).
사용: python t3_build.py <VAR> <fin|dev> <base|panel|feat2|p0copy>
 VAR: VJ1 VJ1s VJ3 VRNDJP0..2 VNJP(verify 상한) → cache/lists/<VAR>.parquet 의 JP appl_no 제거 ; VJ2 → JP 공개일 없는 행 py = 출원일+18개월 연도
 base: R3/build_base_pub.py 와 같은 SQL(메타 경로·변형 조건만) → cache/t3_<VAR>_<mode>/base_rows.parquet
 panel/feat2: 공식 T3/build_panel.py, build_feat2.py 를 수정 없이 실행(cwd=변형 캐시)
 p0copy: 공식 cache_pub_final / cache_pub 의 panel_ccy·feat2·feat_* 복사(sha256 대조)."""
import os, sys, time, shutil, hashlib, subprocess, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib19 import *
VAR, MODE, STEP = sys.argv[1], sys.argv[2], sys.argv[3]
MAXY = 2024 if MODE == 'fin' else 2021
CACHE = os.path.join(C, f't3_{VAR}_{MODE}'); os.makedirs(CACHE, exist_ok=True)
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
t = time.time()
if STEP == 'p0copy':
    src = os.path.join(R3, 'cache_pub_final' if MODE == 'fin' else 'cache_pub')
    for f in ['panel_ccy.parquet', 'feat2.parquet'] + [os.path.basename(x) for x in glob.glob(os.path.join(src, 'feat_*.parquet'))]:
        shutil.copyfile(os.path.join(src, f), os.path.join(CACHE, f)); assert h(os.path.join(src, f)) == h(os.path.join(CACHE, f)), f
    out('p0copy done', MODE, len(os.listdir(CACHE))); sys.exit(0)
if STEP == 'base':
    py_orig = "left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)"
    py_expr = py_orig; cond = ''
    if VAR == 'VJ2':
        py_expr = f"""case when cc='JP' and nullif(pub_date,'') is null and nullif(rgst_date,'') is not null and regexp_matches(coalesce(appl_date,''),'^[0-9]{{8}}$')
                           then strftime(try_strptime(appl_date,'%Y%m%d') + interval 18 month, '%Y') else {py_orig} end"""
    elif VAR == 'VNJP':
        cond = "and cc<>'JP'"
    else:
        lst = os.path.join(C, 'lists', f'{VAR}.parquet'); assert os.path.exists(lst), lst
        cond = f"and not (cc='JP' and appl_no in (select appl_no from read_parquet('{ddb(lst)}')))"
    con = con3()
    con.execute(rf"""
copy (
 select ksic, cc, py::int as year, appl_year::int as appl_year, appl_no,
   regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g') as ipc_raw,
   trim(split_part(applicant,'|',1)) as app1,
   case when prio_cc is null then 0
        when regexp_replace(replace(prio_cc,cc,''),'\|','','g')<>'' then 1 else 0 end as prio_for,
   ipc_all, title
 from (select *, {py_expr} as py
       from read_parquet('{META}') where appl_year >= 2000 {cond})
 where py is not null and py::int between 2000 and {MAXY}
) to '{ddb(CACHE)}/base_rows.parquet' (format parquet, compression zstd)""")
    out(VAR, MODE, 'base done', round(time.time() - t, 1))
    out(con.sql(f"select cc, count(*) n, min(year), max(year) from read_parquet('{ddb(CACHE)}/base_rows.parquet') group by cc order by cc").df().to_string())
    out(con.sql(f"select year, count(*) n from read_parquet('{ddb(CACHE)}/base_rows.parquet') where cc='JP' and year between 2017 and {MAXY} group by 1 order by 1").df().T.to_string())
    sys.exit(0)
env = {**os.environ, 'T3_CACHE': CACHE, 'T3_MAXYEAR': str(MAXY), 'T3_T0MAX': '2024' if MODE == 'fin' else '2018', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
if MODE == 'fin': env['T3R3_FINAL_UNLOCK'] = '1'
shutil.copyfile(os.path.join(T3, 'build_panel.py'), os.path.join(CACHE, 'build_panel.py'))
assert h(os.path.join(T3, 'build_panel.py')) == h(os.path.join(CACHE, 'build_panel.py'))
script = {'panel': 'build_panel.py', 'feat2': 'build_feat2.py'}[STEP]
r = subprocess.run([sys.executable, '-B', os.path.join(T3, script)], cwd=CACHE, env=env)
out(STEP, 'rc', r.returncode, round(time.time() - t, 1)); sys.exit(r.returncode)
