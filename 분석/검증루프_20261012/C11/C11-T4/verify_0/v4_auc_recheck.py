# verify_0 V4 (재개 바퀴): pred 파일에서 AUC·Δskill을 sklearn으로 독립 재계산 + u 파일 레이블 대조
# + 국가 통합 기업 군집(같은 키 문자열이면 CN·EP 같은 군집) 부트스트랩(시드 777, 400회)
import sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C11\C11-T4'
C = W + r'\cache'
t = time.time()
# aid -> key (s1_build.load와 같은 방식으로 재구성)
keymap = {}
for cc in ['CN', 'EP']:
    d = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did'])
    a = pd.read_parquet(C + rf'\app_{cc}.parquet', columns=['did', 'key'])
    m = a.merge(d, on='did', how='inner')
    keymap[cc] = pd.Index(m.key.unique())
    del d, a, m
print('keys', {k: len(v) for k, v in keymap.items()}, round(time.time() - t, 1), flush=True)
res = {'auc': {}, 'label_match': {}}
P = {}
for v in [2020, 2021, 2022, 2023, 2024]:
    p = pd.read_parquet(C + rf'\pred_A_p_{v}.parquet')
    u = pd.read_parquet(C + rf'\u_A_{v}.parquet', columns=['cc', 'aid', 'k', 'risk_p', 'y_p'])
    u = u[u.risk_p]
    fid_u = u.aid.values + np.where(u.cc.values == 'EP', 10_000_000, 0)
    ok = (len(u) == len(p)) and (fid_u == p.fid.values).all() and (u.k.values == p.k.values).all() and (u.y_p.values.astype(int) == p.y.values).all()
    res['label_match'][v] = bool(ok)
    res['auc'][v] = {c: float(roc_auc_score(p.y, p[c])) for c in ['m', 'pi', 'sb']}
    aid = p.fid.values % 10_000_000
    p['key'] = np.where(p.cc.values == 'CN', keymap['CN'].values[np.minimum(aid, len(keymap['CN']) - 1)], keymap['EP'].values[np.minimum(aid, len(keymap['EP']) - 1)])
    P[v] = p
    print(v, res['auc'][v], ok, round(time.time() - t, 1), flush=True)
sk = lambda a, b: (a - b) / (1 - b)
for base in ['pi', 'sb']:
    res[f'dskill_{base}'] = float(np.mean([sk(res['auc'][v]['m'], res['auc'][v][base]) for v in P]))
# 국가 통합 군집 수
allk = pd.Index(np.unique(np.concatenate([p.key.values for p in P.values()])))
allf = pd.Index(np.unique(np.concatenate([p.fid.values for p in P.values()])))
res['n_clusters_fid'] = int(len(allf)); res['n_clusters_key'] = int(len(allk))
# 가중 AUC (정렬 1회)
class Prep:
    def __init__(s, sc, y):
        u, inv = np.unique(np.asarray(sc, float), return_inverse=True); s.g = inv; s.G = len(u); s.y = np.asarray(y).astype(bool)
    def auc(s, w):
        wp = np.bincount(s.g, w * s.y, s.G); wn = np.bincount(s.g, w * ~s.y, s.G)
        cn = np.cumsum(wn) - wn
        return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
B = 400
for clus in ['fid', 'key']:
    idx = allf if clus == 'fid' else allk
    rng = np.random.default_rng(777)
    Wt = rng.poisson(1.0, size=(B, len(idx))).astype(float)
    preps = {v: (Prep(p.m, p.y), Prep(p.pi, p.y), Prep(p.sb, p.y), idx.get_indexer(p[clus].values)) for v, p in P.items()}
    bs = {'pi': [], 'sb': []}
    for b in range(B):
        for base, j in [('pi', 1), ('sb', 2)]:
            vals = []
            for v, pr in preps.items():
                w = Wt[b, pr[3]]
                vals.append(sk(pr[0].auc(w), pr[j].auc(w)))
            bs[base].append(np.mean(vals))
    for base in ['pi', 'sb']:
        a = np.array(bs[base])
        res[f'boot_{clus}_{base}'] = dict(lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)), se=float(a.std(ddof=1)))
    print(clus, res[f'boot_{clus}_pi'], res[f'boot_{clus}_sb'], round(time.time() - t, 1), flush=True)
json.dump(res, open(W + r'\verify_0\v4_out.json', 'w'), indent=1)
print('done', round(time.time() - t, 1))
