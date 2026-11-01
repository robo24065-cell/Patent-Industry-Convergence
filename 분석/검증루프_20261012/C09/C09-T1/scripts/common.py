# -*- coding: utf-8 -*-
import os
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(ROOT, r"분석\모델경쟁_20260927")
WD = os.path.join(ROOT, r"분석\검증루프_20261012\C09\C09-T1")
CACHE = os.path.join(WD, "cache")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
RF = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/")
DATA_END = "20250822"
HY, KIA = "119980045675", "119980003181"
def ro(path):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
def duck():
    import duckdb
    con = duckdb.connect()
    con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false")
    return con
