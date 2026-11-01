# verify_0 V2: P0(b) 검정력 독립 재계산(evalkit12·s3 미사용; sklearn roc_auc_score, 다항 군집 재표집, 다른 난수 구현)
# + 진단(판정 밖): 합성 설계에서 군집 효과(u_a, v_s)를 뺀 경우의 SE.
import sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
sys.stdout.reconfigure(encoding='utf-8')
W = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C12/C12-T4'
t = time.time()
fr = [pd.read_parquet(f'{W}/cache/pred_{v}_sb.parquet') for v in range(2020, 2025)]
allf = pd.concat(fr)
firms = pd.Index((allf.cc + '_' + allf.aid.astype(str)).unique()); scs = pd.Index(np.unique(allf.s))
FI = [firms.get_indexer(f.cc + '_' + f.aid.astype(str)) for f in fr]; SI = [scs.get_indexer(f.s) for f in fr]
rng = np.random.default_rng(99)


def make_z(with_cluster):
    u = rng.normal(size=len(firms)); vv = rng.normal(size=len(scs))
    Z = []
    for k, f in enumerate(fr):
        e = rng.normal(size=len(f))
        Z.append(((u[FI[k]] + vv[SI[k]] + e) / np.sqrt(3)) if with_cluster else e)
    return Z


def ds(s1, s0, y, w):
    a1 = roc_auc_score(y, s1, sample_weight=w); a0 = roc_auc_score(y, s0, sample_weight=w)
    return (a1 - a0) / (1 - a0)


def calib(Z):
    lo, hi = 0.0, 4.0
    for _ in range(30):
        mid = (lo + hi) / 2
        m = np.mean([ds(f.SB.values + mid * f.y.values + Z[k], f.SB.values, f.y.values, f.w.values) for k, f in enumerate(fr)])
        if abs(m - 0.005) < 0.0002:
            break
        lo, hi = (mid, hi) if m < 0.005 else (lo, mid)
    return mid, m


def boot(Z, mu, B, kinds):
    S1 = [f.SB.values + mu * f.y.values + Z[k] for k, f in enumerate(fr)]
    out = {k: [] for k in kinds}
    for b in range(B):
        cf = rng.multinomial(len(firms), np.full(len(firms), 1 / len(firms))).astype(float)
        cs = rng.multinomial(len(scs), np.full(len(scs), 1 / len(scs))).astype(float)
        for kd in kinds:
            vals = []
            for k, f in enumerate(fr):
                bw = cf[FI[k]] if kd == 'firm' else cf[FI[k]] * cs[SI[k]]
                w = f.w.values * bw
                vals.append(ds(S1[k], f.SB.values, f.y.values, w))
            out[kd].append(np.mean(vals))
    return {k: float(np.std(v, ddof=1)) for k, v in out.items()}


res = {}
SEL = sys.argv[1]
rng = np.random.default_rng(99 if SEL == 'prereg_design' else 199)
for nm, wc, B in [x for x in [('prereg_design', True, 60), ('no_cluster_effects', False, 40)] if x[0] == SEL]:
    Z = make_z(wc)
    mu, m = calib(Z)
    se = boot(Z, mu, B, ['firm', 'two'])
    res[nm] = {'mu': mu, 'dskill': m, 'B': B, 'SE': se, 'power': {k: float(norm.cdf(0.005 / v - 1.96)) for k, v in se.items()}}
    print(nm, res[nm], round(time.time() - t, 1), flush=True)
json.dump(res, open(f'{W}/verify_0/v2_out_{SEL}.json', 'w'), indent=1)
