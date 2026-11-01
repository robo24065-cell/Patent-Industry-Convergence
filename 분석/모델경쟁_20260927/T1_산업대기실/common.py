# T1 산업 대기실 공용 경로/유틸 (읽기전용 접속만)
import sqlite3, pathlib, sys
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B / "분석/모델경쟁_20260927/T1_산업대기실"
CACHE = T1 / "cache"
CACHE.mkdir(exist_ok=True)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def ro(rel):
    return sqlite3.connect((B / rel).as_uri() + "?mode=ro", uri=True)

BR = lambda: ro("수집파이프라인/business_rights.sqlite")
RG = lambda: ro("수집파이프라인/registry_full.sqlite")
DT = lambda: ro("공모전데이터/외부데이터/DART/dart_company.sqlite")
KR_REG = B / "공모전데이터/작업데이터/kr_registered.parquet"
SNAP_CUT = '20250822'   # 제공 KR 등록 최종일
