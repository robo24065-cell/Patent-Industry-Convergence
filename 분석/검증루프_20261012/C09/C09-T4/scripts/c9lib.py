# C09-T4 공용
import os, re, json, sqlite3
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B + r'\수집파이프라인\registry_full.sqlite'
MC = B + r'\분석\모델경쟁_20260927'
T4D = MC + r'\T4_특허포기예측'
N4D = MC + r'\N4_특허행태_기업폐업조기경보'
W = B + r'\분석\검증루프_20261012\C09\C09-T4'
C = W + r'\cache'


def ro():
    return sqlite3.connect(f'file:{DB}?mode=ro', uri=True)


def load_hol():
    """h1_holidays.py 산출: 공휴일 집합(R2용), datetime64[D] 배열"""
    h = pd.read_csv(os.path.join(C, 'holidays_kr_2000_2026.csv'))
    return np.array(sorted(set(pd.to_datetime(h.date).values.astype('datetime64[D]'))))


def dow(d):
    """월=0 ... 일=6 (1970-01-01 목요일)"""
    return (np.asarray(d, dtype='datetime64[D]').view('int64') + 3) % 7


def adjust_due(due, rule, hol=None):
    """due: datetime64[D] 배열. R1: 토·일 → 다음 월요일. R2: 토·일·공휴일 → 다음 영업일(연쇄). 그 외 규칙은 그대로."""
    due = np.asarray(due, dtype='datetime64[D]')
    out = due.copy()
    ok = ~np.isnat(due)
    if rule == 'R1':
        w = dow(due)
        add = np.where(w == 5, 2, np.where(w == 6, 1, 0))
        out[ok] = due[ok] + add[ok].astype('timedelta64[D]')
    elif rule == 'R2':
        out[ok] = np.busday_offset(due[ok], 0, roll='forward', weekmask='1111100', holidays=hol)
    return out
