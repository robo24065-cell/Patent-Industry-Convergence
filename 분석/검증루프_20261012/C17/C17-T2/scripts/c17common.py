# -*- coding: utf-8 -*-
"""C17-T2 공용: 경로·정규화·읽기 전용 연결. 트랙·원천 파일은 읽기만 한다."""
import sys, pathlib, json, re, sqlite3
import numpy as np, pandas as pd
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
HERE = pathlib.Path(__file__).resolve().parent.parent; CC = HERE/'cache'
C16 = B/'분석/검증루프_20261012/C16/C16-T1'; C03 = B/'분석/검증루프_20261012/C03/C03-T1'
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; TC = T1/'cache'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'
REG = B/'수집파이프라인/registry_full.sqlite'; BIZ = B/'수집파이프라인/business_rights.sqlite'
KP = B/'공모전데이터/작업데이터/kr_patents.parquet'; CA = B/'공모전데이터/외부데이터/DART/corp_applicant.parquet'
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()          # T1 prep_app.py·C16과 같은 정규화
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']


def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)


def digits(s):
    s = re.sub(r'\D', '', str(s or ''))
    return s if len(s) == 13 else ''


def jdump(o, p): json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def wilson(x, n, z=1.959963984540054):
    if n == 0: return [None, None, None]
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


def size_bin(nd):
    nd = np.asarray(nd); return np.where(nd <= 5, 'S', np.where(nd <= 30, 'M', 'L'))


def is_foreign(code): return pd.Series(code).astype(str).str[0].isin(['5', '6', '7']).values
