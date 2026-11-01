# -*- coding: utf-8 -*-
"""C15-T2 T1 공용: 경로·시드 묶음 정의(PREREG §1). guard15 를 먼저 설치한 뒤 트랙 R3 모듈을 읽기 전용 import 한다."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard15 as G
HERE = G.HERE
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'
sys.path.insert(0, str(R3)); os.chdir(str(T1))
PTR = 'prep_tr.parquet'
CT1 = HERE/'cache'/'t1'


def fcp(c):
    return R3/'cache_final'/'o2019' if c == 2019 else R3/'cache'/f'o{c}'


def seeds(b):
    """PREREG §1: b=0 공식, b=1..9 σ=100+b"""
    if b == 0:
        return dict(xgb=[0, 1], neg_base=1234, firm=0, base_xgb=0, sbb_neg=2019, sbb_xgb=0)
    s = 100 + b
    return dict(xgb=[2 * s, 2 * s + 1], neg_base=1234 + 10000 * b, firm=s, base_xgb=s, sbb_neg=2019 + 10000 * b, sbb_xgb=s)


def rd(b, c):
    p = CT1/f'b{b}'/('final' if c == 2019 else 'dev'); p.mkdir(parents=True, exist_ok=True); return p


def adir(c):
    p = CT1/'A'/f'o{c}'; p.mkdir(parents=True, exist_ok=True); return p
