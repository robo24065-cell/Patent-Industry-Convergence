# -*- coding: utf-8 -*-
"""C17-T1 공용: 경로·읽기 전용 원천·C16-T1 캐시(읽기 전용) 경로. 쓰기는 C17-T1 안에서만(guard17)."""
import sys, pathlib, json, os, sqlite3
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
MC = B/'분석/모델경쟁_20260927'; T1 = MC/'T1_산업대기실'; R3 = T1/'R3'; FC = R3/'cache_final'/'o2019'; RD = R3/'runs'/'FROZEN_final'
CACHE = T1/'cache'; KR = B/'공모전데이터/작업데이터/kr_registered.parquet'; KP = B/'공모전데이터/작업데이터/kr_patents.parquet'
PIPE = B/'수집파이프라인'; RGF = PIPE/'registry_full.sqlite'; BRS = PIPE/'business_rights.sqlite'
EXTD = MC/'_shared'/'ext'; CA = B/'공모전데이터/외부데이터/DART/corp_applicant.parquet'
C16 = B/'분석/검증루프_20261012/C16/C16-T1'; C16C = C16/'cache'; C15 = B/'분석/검증루프_20261012/C15/C15-T2'
HERE = pathlib.Path(__file__).resolve().parent.parent; CC = HERE/'cache'
CUT = 2019; CAPV = 20191231; END = 20250822
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
OFFICIAL = dict(skill=0.47267044268107444, auc=0.9002187222427036, top1000=313)


def ro(p):
    return sqlite3.connect(f"file:{pathlib.Path(p).as_posix()}?mode=ro", uri=True)


def s8(x):
    s = pd.Series(x).fillna('').astype(str).str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values


def jdump(o, p): json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)


def kidx():
    K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); return K, {k: i for i, k in enumerate(K)}


def official_frame():
    """공식 평가 패널 전체 행(prep_te 순서): firm,k,newc,prior_any,px,pl,sbb,base6,y_off(공식 y_u, C16 labels = C15 final_labels 대조 완료)."""
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'newc', 'prior_any'])
    fz = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'px', 'pl']); assert (fz.firm.values == te.firm.values).all() and (fz.k.values == te.k.values).all()
    bo = pd.read_parquet(FC/'baseline_scores.parquet'); sb = np.load(C16C/'p0'/'sbb_all.npy')
    L = pd.read_parquet(C16C/'labels.parquet'); Lt = L[L.src == 'te'].reset_index(drop=True); assert (Lt.firm.values == te.firm.values).all() and (Lt.k.values == te.k.values).all()
    O = te.copy(); O['px'] = fz.px.values; O['pl'] = fz.pl.values; O['sbb'] = sb; O['y_off'] = Lt.y_u.values.astype(np.int8)
    O['y_reg'] = Lt.y_reg.values.astype(np.int8); O['y_pub'] = Lt.y_pub.values.astype(np.int8)
    for c in BK: O[c] = bo[c].values
    return O
