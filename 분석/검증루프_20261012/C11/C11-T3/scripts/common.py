# -*- coding: utf-8 -*-
import os, sys, time
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import duckdb
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
EXT = MC + "/_shared/ext"
META = MC + "/_shared/meta/*.parquet"
METAD = MC + "/_shared/meta"
KRP = BASE + "/공모전데이터/작업데이터/kr_patents.parquet"
T3R3 = MC + "/T3_기술셀부상예측/R3"
W = BASE + "/분석/검증루프_20261012/C11/C11-T3"
T0 = time.time()
def log(*a):
    print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
def con(mem='3GB'):
    c = duckdb.connect()
    c.execute(f"set memory_limit='{mem}'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{W}/cache/_tmp'")
    return c
