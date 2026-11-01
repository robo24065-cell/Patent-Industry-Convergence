# C14-T4 공용. 쓰기는 C14\C14-T4 안에서만. 날짜 가드: 공개일 < 20250101.
import sys, os, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
LOOP = B + r'\분석\검증루프_20261012'
W = LOOP + r'\C14\C14-T4'
C = W + r'\cache'
C12 = LOOP + r'\C12\C12-T4'
C12C = C12 + r'\cache'
C13 = LOOP + r'\C13\C13-T4'
C13C = C13 + r'\cache'
CUT = 20250101
SEED = 20261017
EVAL_Y = [2020, 2021, 2022, 2023, 2024]
CCS = ['CN', 'EP']
E_LOCK_A = 15961.345174791417   # C12 gates.json P0(a) exp_pos
TARGET = 0.002


def out(*a):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(*a, flush=True)


def guard_parquet_pub(path, col='pub'):
    """날짜 가드: 표의 최대 공개일이 2025-01-01 미만인지 확인."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=[col])
    mx = int(pd.Series(t.column(col).to_numpy()).max())
    assert mx < CUT, (path, mx)
    return mx


class AUCPrep:
    """점수별 동률 그룹을 미리 만들고 가중치만 바꿔 가중 AUC 계산(동률 0.5)."""
    def __init__(self, score, y):
        u, g = np.unique(score, return_inverse=True)
        self.g = g.astype(np.int64); self.ng = len(u); self.y = np.asarray(y).astype(bool)
        self.gp = self.g[self.y]; self.gn = self.g[~self.y]

    def auc(self, w):
        wp = np.bincount(self.gp, weights=w[self.y], minlength=self.ng)
        wn = np.bincount(self.gn, weights=w[~self.y], minlength=self.ng)
        cn = np.cumsum(wn) - wn
        P = wp.sum(); N = wn.sum()
        return float((wp * (cn + 0.5 * wn)).sum() / (P * N))


def dskill(a, b):
    return (a - b) / (1 - b)


def load_c12_frames(models=('SB', 'PI', 'pub', 'prov', 'free', 'hsc', 'all', 'pubctx', 'scagg')):
    fr = []
    for v in EVAL_Y:
        P = None
        for g in ['sb', 'base', 'pubarms', 'ref']:
            q = pd.read_parquet(C12C + rf'\pred_{v}_{g}.parquet')
            if P is None:
                P = q[['cc', 'aid', 's', 'y', 'w']].copy()
            else:
                assert (P.aid.values == q.aid.values).all() and (P.s.values == q.s.values).all() and (P.cc.values == q.cc.values).all()
            for m in q.columns:
                if m in models:
                    P[m] = q[m].values
        P['v'] = v
        fr.append(P)
    return fr
