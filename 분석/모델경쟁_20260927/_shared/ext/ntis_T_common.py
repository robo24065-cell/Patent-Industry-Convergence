# -*- coding: utf-8 -*-
"""NTIS 트랙별 의미 검정 공용 함수 (쓰기는 _shared/ext/ntis_T_work/ 안의 ntis_ 파일만).
- 트랙 폴더·원본 DB는 읽기만 한다.
- 🔒 최종시험 구간 결과변수는 계산하지 않는다(각 스크립트에서 원점 상한 assert).
"""
import os, json
import numpy as np, pandas as pd

EXT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(EXT, 'ntis_T_work')
ROOT = os.path.dirname(os.path.dirname(EXT))          # 모델경쟁_20260927
T1 = os.path.join(ROOT, 'T1_산업대기실')
T3 = os.path.join(ROOT, 'T3_기술셀부상예측')
T4 = os.path.join(ROOT, 'T4_특허포기예측')
os.makedirs(WORK, exist_ok=True)

GRP_MAP = {'대학': 'univ', '출연연구소': 'gri', '국공립연구소': 'gri', '대기업': 'big', '중견기업': 'big',
           '중소기업': 'sme', '기타': 'oth', '정부부처': 'oth'}
GRPS = ['univ', 'gri', 'big', 'sme', 'oth']


def out(name):
    assert name.startswith('ntis_')
    return os.path.join(WORK, name)


def peak_mem_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return -1


def fe_ols(df, y, xs, fe, cluster):
    """일원 고정효과(fe 열 하나, 결합키) 제거 후 OLS, 군집 강건 표준오차.
    반환: DataFrame(coef, se, t, n, n_clu)"""
    d = df[[y] + xs + [fe, cluster]].dropna()
    g = d.groupby(fe, sort=False)
    Y = (d[y] - g[y].transform('mean')).values.astype(float)
    X = np.column_stack([(d[c] - g[c].transform('mean')).values.astype(float) for c in xs])
    keep = np.abs(X).sum(1) >= 0
    XtX = X.T @ X
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtX)
    b = inv @ (X.T @ Y)
    u = Y - X @ b
    cl = pd.factorize(d[cluster].values)[0]
    S = np.zeros((cl.max() + 1, len(xs)))
    np.add.at(S, cl, X * u[:, None])
    G = cl.max() + 1
    meat = S.T @ S
    adj = G / max(G - 1, 1)
    V = adj * inv @ meat @ inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return pd.DataFrame({'coef': b, 'se': se, 't': b / np.where(se > 0, se, np.nan), 'n': len(d), 'n_clu': G}, index=xs)


def zscore(s):
    s = pd.Series(s, dtype=float)
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def rank_norm(s):
    s = pd.Series(s, dtype=float)
    return (s.rank(pct=True) - 0.5).fillna(0.0)


def save_json(obj, name):
    with open(out(name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
