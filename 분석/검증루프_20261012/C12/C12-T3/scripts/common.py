# -*- coding: utf-8 -*-
"""C12-T3 공용 경로. 입력은 모두 읽기 전용, 쓰기는 C12\\C12-T3 안만."""
import os, sys, re, html
sys.dont_write_bytecode = True
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = os.path.join(ROOT, "분석")
WD = os.path.join(AN, r"검증루프_20261012\C12\C12-T3")
CACHE = os.path.join(WD, "cache"); LOGS = os.path.join(WD, "logs"); DL = os.path.join(WD, "download")
T3 = os.path.join(AN, r"모델경쟁_20260927\T3_기술셀부상예측")
R3 = os.path.join(T3, "R3")
BASE_ROWS = os.path.join(R3, r"cache_pub_final\base_rows.parquet").replace("\\", "/")
META = os.path.join(AN, r"모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
RF = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite")
PI_CSV = os.path.join(ROOT, r"공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv")
C09V0 = os.path.join(AN, r"검증루프_20261012\C09\C09-T3\verify_0")
C07DL = os.path.join(AN, r"검증루프_20261012\C07\C07-T1\download")
MANU = os.path.join(AN, r"보고서_20261012\manuscript_v3.md")


def ro(path):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def duck():
    import duckdb
    con = duckdb.connect()
    con.execute("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false")
    return con


def n1(s):
    """C04-T3 N1 정규화 그대로."""
    s = html.unescape(s or '').lower()
    for w in ['주식회사', '(주)', '㈜', '(株)', '（주）']:
        s = s.replace(w, '')
    return re.sub(r"[\s\.,·\-\(\)\[\]'\"“”‘’&/]+", '', s)
