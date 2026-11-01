# C18-T4 공용. 쓰기는 C18\C18-T4 안에서만. 다른 바퀴 캐시는 읽기만.
import sys, os, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
LOOP = B + r'\분석\검증루프_20261012'
W = LOOP + r'\C18\C18-T4'
C = W + r'\cache'
C12C = LOOP + r'\C12\C12-T4\cache'
C13C = LOOP + r'\C13\C13-T4\cache'
C14C = LOOP + r'\C14\C14-T4\cache'
C17C = LOOP + r'\C17\C17-T4\cache'
SEED = 20261017
Z80 = 1.959964 + 0.841621
FRACS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]

# 잠금 예상 양성(이전 바퀴 게이트 값, 읽기 전용 원천에서 재확인)
E_LOCK = {'K1': 15961.345174791417, 'K2': 1981.51, 'K3': 3408.0, 'K4': 1960.4}
MAIN_V = {'K1': 2024, 'K2': 2023, 'K3': 2023, 'K4': 2023}
PUB_ARMS = {'K1': ['free', 'hsc', 'scagg'], 'K2': ['free', 'hsc'], 'K3': ['free', 'hsc'], 'K4': ['free', 'hsc', 'off']}


def out(*a):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(*a, flush=True)


class AUCPrep:
    """점수별 동률 그룹을 미리 만들고 가중치만 바꿔 가중 AUC(동률 0.5). C14 lib14와 같은 식."""
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


class StratAUC:
    """층(k) 안 쌍만 쓰는 층화 AUC = Σ_k P_k N_k AUC_k / Σ_k P_k N_k. 점수 동률 0.5."""
    def __init__(self, score, y, k):
        k = np.asarray(k); y = np.asarray(y).astype(bool)
        o = np.lexsort((score, k))
        ks = k[o]; sc = np.asarray(score)[o]
        # (k, 점수) 그룹
        key = np.empty(len(o), np.int64)
        newg = np.ones(len(o), bool); newg[1:] = (ks[1:] != ks[:-1]) | (sc[1:] != sc[:-1])
        g = np.cumsum(newg) - 1
        key[o] = g
        self.g = key; self.ng = int(g.max()) + 1
        gk = np.empty(self.ng, np.int64); gk[g] = ks
        self.gk = gk
        self.kfirst = np.r_[True, gk[1:] != gk[:-1]]
        self.kid = np.cumsum(self.kfirst) - 1
        self.nk = int(self.kid.max()) + 1
        self.y = y; self.gp = self.g[y]; self.gn = self.g[~y]

    def auc(self, w):
        wp = np.bincount(self.gp, weights=w[self.y], minlength=self.ng)
        wn = np.bincount(self.gn, weights=w[~self.y], minlength=self.ng)
        cs = np.cumsum(wn)
        # 층별 누적 음성: 층 시작 직전 누적을 빼기
        start = np.where(self.kfirst)[0]
        base = np.r_[0.0, cs][start]           # 층 시작 전 누적
        cn = cs - wn - base[self.kid]
        num = (wp * (cn + 0.5 * wn)).sum()
        Pk = np.bincount(self.kid, weights=wp, minlength=self.nk)
        Nk = np.bincount(self.kid, weights=wn, minlength=self.nk)
        den = (Pk * Nk).sum()
        return float(num / den)


def load_frame(cand, v):
    return pd.read_parquet(C + rf'\fr_{cand}_{v}.parquet')


def codes(series):
    return pd.factorize(series, sort=True)[0].astype(np.int64)
