# -*- coding: utf-8 -*-
r"""C18-T2 T3 변형 캐시 빌더. 사용: python t3_build.py <VAR> <fin|dev> <base|panel|feat2|p0copy>
 VAR: V1A·VNUS(verify_1 반박용, 아래 cond 참조), V1(US 7자리 쌍 행 제거) V1E(+EP 짧은 형식 행 제거) V2(공개일 없는 US 행 제거 = least(pub,rgst)에서 rgst 제외) V12(V1+V2)
 base: R3/build_base_pub.py 의 SQL 그대로(메타 경로와 변형 조건만 바꿈) → cache/t3_<VAR>_<mode>/base_rows.parquet
 panel/feat2: 공식 T3/build_panel.py, build_feat2.py 를 수정 없이 실행(cwd=변형 캐시, build_panel.py 사본을 cwd에 둠: build_feat2가 매크로를 읽음)
 p0copy: 공식 cache_pub_final / cache_pub 의 panel_ccy·feat2·feat_* 를 cache/t3_P0_<mode>/ 로 복사(sha256 대조) — 공식 폴더에 쓰기 방지
 원자료·공식 폴더는 읽기만."""
import os, sys, time, shutil, hashlib, subprocess, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib18 import *
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
    if VAR in ('V1', 'V12'):
        M = ddb(C) + '/meta_V1.parquet'
    elif VAR == 'V1E':
        M = ddb(C) + '/meta_V1E.parquet'
    else:
        M = META
    cond = "and not (cc='US' and nullif(pub_date,'') is null)" if VAR in ('V2', 'V12') else ''
    # verify_1 반박용: V1A = 쌍에서 8자리 행을 빼고 7자리 행 유지(V1의 반대), VNUS = US 행 전체 제거(상한 시나리오)
    if VAR == 'V1A': cond = f"and not (cc='US' and appl_no in (select a8 from read_parquet('{ddb(C)}/us_dup_pairs.parquet')))"
    if VAR == 'VNUS': cond = "and cc<>'US'"
    # verify_1 위약 대조: VRND = 쌍이 아닌 US 행 중 출원 2000–01년에서 2,099행을 무작위(해시 순, 고정) 제거 — 같은 크기의 무관한 교란
    if VAR == 'VRND': cond = f"""and not (cc='US' and appl_no in (select appl_no from read_parquet('{META}') where cc='US' and appl_year in (2000,2001) and length(appl_no)=8
        and appl_no not in (select a8 from read_parquet('{ddb(C)}/us_dup_pairs.parquet')) order by hash(appl_no || 'c18') limit 2099))"""
    con = con3()
    con.execute(rf"""
copy (
 select ksic, cc, py::int as year, appl_year::int as appl_year, appl_no,
   regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g') as ipc_raw,
   trim(split_part(applicant,'|',1)) as app1,
   case when prio_cc is null then 0
        when regexp_replace(replace(prio_cc,cc,''),'\|','','g')<>'' then 1 else 0 end as prio_for,
   ipc_all, title
 from (select *, left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4) as py
       from read_parquet('{M}') where appl_year >= 2000 {cond})
 where py is not null and py::int between 2000 and {MAXY}
) to '{ddb(CACHE)}/base_rows.parquet' (format parquet, compression zstd)""")
    out(VAR, MODE, 'base done', round(time.time() - t, 1))
    out(con.sql(f"select cc, count(*) n, min(year), max(year) from read_parquet('{ddb(CACHE)}/base_rows.parquet') group by cc order by cc").df().to_string())
    sys.exit(0)
env = {**os.environ, 'T3_CACHE': CACHE, 'T3_MAXYEAR': str(MAXY), 'T3_T0MAX': '2024' if MODE == 'fin' else '2018', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
if MODE == 'fin': env['T3R3_FINAL_UNLOCK'] = '1'
shutil.copyfile(os.path.join(T3, 'build_panel.py'), os.path.join(CACHE, 'build_panel.py'))
assert h(os.path.join(T3, 'build_panel.py')) == h(os.path.join(CACHE, 'build_panel.py'))
script = {'panel': 'build_panel.py', 'feat2': 'build_feat2.py'}[STEP]
r = subprocess.run([sys.executable, '-B', os.path.join(T3, script)], cwd=CACHE, env=env)
out(STEP, 'rc', r.returncode, round(time.time() - t, 1)); sys.exit(r.returncode)
