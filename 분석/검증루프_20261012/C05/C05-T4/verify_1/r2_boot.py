# verify_1 r2: M1 vs M0 Δskill CI의 시드 민감도(다른 시드 3개, 독립 구현) + DeLong(원점별·잠금) + 원점 고정효과 메타 합성.
# 사용법: python r2_boot.py dev <seed> <nb> | lock <seed> <nb> | delong
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); C = os.path.join(W, 'cache'); O = os.path.join(W, 'verify_1')
EVAL = ['201812','201912','202012','202112','202212','202312']
T0 = time.time()

class WA:  # 가중 AUC(동점 0.5) — libB 비참조 독립 구현
    def __init__(self, y, s):
        r = rankdata(s, method='dense').astype(np.int64) - 1
        self.r = r; self.k = r.max() + 1; self.y = y.astype(bool)
    def auc(self, w):
        g1 = np.bincount(self.r[self.y], weights=w[self.y], minlength=self.k)
        g0 = np.bincount(self.r[~self.y], weights=w[~self.y], minlength=self.k)
        c0 = np.cumsum(g0) - g0
        return float((g1 * (c0 + 0.5 * g0)).sum() / (g1.sum() * g0.sum()))

def delong(y, s1, s0):
    y = y.astype(bool); out = []
    m = y.sum(); n = (~y).sum()
    V10 = []; V01 = []; A = []
    for s in (s1, s0):
        pos = s[y]; neg = s[~y]
        r_all = rankdata(np.concatenate([pos, neg])); r_pos = rankdata(pos); r_neg = rankdata(neg)
        a = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (r_all[:m] - r_pos) / n; v01 = 1 - (r_all[m:] - r_neg) / m
        A.append(a); V10.append(v10); V01.append(v01)
    S10 = np.cov(np.vstack(V10)); S01 = np.cov(np.vstack(V01))
    S = S10 / m + S01 / n
    d = A[0] - A[1]; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    return dict(auc1=A[0], auc0=A[1], dauc=d, se=float(np.sqrt(var)), z=float(d / np.sqrt(var)), p=float(2 * norm.sf(abs(d / np.sqrt(var)))),
                skill=(A[0]-A[1])/(1-A[1]))

mode = sys.argv[1]
if mode == 'delong':
    res = {}; ws = []; ds = []
    for v in EVAL:
        d0 = pd.read_parquet(os.path.join(C, f'pred_{v}_M0.parquet')); d1 = pd.read_parquet(os.path.join(C, f'pred_{v}_M1.parquet'))
        r = delong(d0.y.values, d1.p.values, d0.p.values); res[v] = r
        ws.append(1 / r['se'] ** 2); ds.append(r['dauc'])
    ws = np.array(ws); ds = np.array(ds)
    fe = float((ws * ds).sum() / ws.sum()); fese = float(np.sqrt(1 / ws.sum()))
    res['fixed_effect_dauc'] = dict(est=fe, se=fese, ci=[fe - 1.96 * fese, fe + 1.96 * fese], p=float(2 * norm.sf(abs(fe / fese))),
                                    note='원점 간 같은 기업 상관 무시(독립 가정) — 근사')
    f = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet')); l = pd.read_parquet(os.path.join(C, 'lock_labels.parquet'))
    res['lock'] = delong(l.y.values, f.score_M1.values, f.score_M0.values)
    json.dump(res, open(os.path.join(O, 'r2_delong.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float)); sys.exit(0)

seed = int(sys.argv[2]); nb = int(sys.argv[3]); rng = np.random.default_rng(seed)
if mode == 'dev':
    P = {}
    for v in EVAL:
        d0 = pd.read_parquet(os.path.join(C, f'pred_{v}_M0.parquet')); d1 = pd.read_parquet(os.path.join(C, f'pred_{v}_M1.parquet'))
        P[v] = (d0.key.values, WA(d0.y.values, d0.p.values), WA(d0.y.values, d1.p.values))
    keys = pd.Index(sorted(set(np.concatenate([P[v][0] for v in EVAL]))))
    codes = {v: keys.get_indexer(P[v][0]) for v in EVAL}
    out = []
    for b in range(nb):
        cnt = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
        sks = []
        for v in EVAL:
            w = cnt[codes[v]]; a0 = P[v][1].auc(w); a1 = P[v][2].auc(w); sks.append((a1 - a0) / (1 - a0))
        out.append(np.mean(sks))
    out = np.array(out)
else:
    f = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet')); l = pd.read_parquet(os.path.join(C, 'lock_labels.parquet'))
    y = l.y.values; A0 = WA(y, f.score_M0.values); A1 = WA(y, f.score_M1.values); n = len(y)
    out = []
    for b in range(nb):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        a0 = A0.auc(w); a1 = A1.auc(w); out.append((a1 - a0) / (1 - a0))
    out = np.array(out)
r = dict(mode=mode, seed=seed, nb=nb, mean=float(out.mean()), ci=[float(np.quantile(out, .025)), float(np.quantile(out, .975))],
         frac_le0=float((out <= 0).mean()), elapsed=round(time.time() - T0))
json.dump(r, open(os.path.join(O, f'r2_{mode}_{seed}.json'), 'w'), indent=1)
print(r)
