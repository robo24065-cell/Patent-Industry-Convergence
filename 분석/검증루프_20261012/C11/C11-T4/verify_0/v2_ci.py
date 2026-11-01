# verify_0 V2 집계: AUC 차의 기업 군집 부트스트랩(200회, 시드 99) — 평가 원점 2022–24 평균
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C11/C11-T4/verify_0'
VS = [2022, 2023, 2024]


def prep(s):
    return np.unique(s, return_inverse=True)[1]


def wauc(g, y, w):
    wp = np.bincount(g, w * y); wn = np.bincount(g, w * (1 - y)); cn = np.cumsum(wn) - wn
    return (wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


comp = {'a_p-fi': ('a_p', 'a_fi'), 'a_p-hs': ('a_p', 'a_hs'), 'b_p-fi': ('b_p', 'b_fi'), 'b_p-hs': ('b_p', 'b_hs'),
        'c_fiOutcome_fi-p': ('c_fi_fi', 'c_fi_p'), 'c_hsOutcome_hs-p': ('c_hs_hs', 'c_hs_p')}
res = {}
for name, (n1, n0) in comp.items():
    D = []
    for v in VS:
        A = pd.read_parquet(f'{OUT}/v2_pred_{n1}_{v}.parquet'); Bp = pd.read_parquet(f'{OUT}/v2_pred_{n0}_{v}.parquet')
        assert (A.fid.values == Bp.fid.values).all() and (A.y.values == Bp.y.values).all()
        D.append((prep(A.s.values), prep(Bp.s.values), A.y.values.astype(float), A.fid.values))
    allf = pd.Index(np.unique(np.concatenate([d[3] for d in D])))
    obs = np.mean([wauc(a, y, np.ones(len(y))) - wauc(b, y, np.ones(len(y))) for a, b, y, f in D])
    W = np.random.default_rng(99).poisson(1.0, size=(200, len(allf))).astype(float)
    bs = []
    for i in range(200):
        bs.append(np.mean([wauc(a, y, W[i, allf.get_indexer(f)]) - wauc(b, y, W[i, allf.get_indexer(f)]) for a, b, y, f in D]))
    res[name] = dict(obs=float(obs), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    print(name, res[name], flush=True)
json.dump(res, open(f'{OUT}/v2_ci.json', 'w'), indent=1)
