# v4: 저장 예측으로 AUC·D·대비 재계산(자체 순위 기반 가중 AUC), 기업 군집 다항 부트스트랩, 귀무 파일 중복 제거 요약
import sys, json, glob, numpy as np, pandas as pd
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T4'
C = W + r'\cache'
MOD = ['pub', 'prov', 'free', 'hsc', 'all', 'pubctx', 'scagg', 'PI']

def wauc(y, s, w):
    o = np.argsort(s, kind='mergesort'); s = s[o]; y = y[o]; w = w[o]
    u, start = np.unique(s, return_index=True)
    g = np.searchsorted(u, s)
    wp = np.bincount(g, weights=w * y, minlength=len(u)); wn = np.bincount(g, weights=w * (1 - y), minlength=len(u))
    cn = np.cumsum(wn) - wn
    return (wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())

ds = lambda a, b: (a - b) / (1 - b)
fr = []
for v in range(2020, 2025):
    P = None
    for g in ['sb', 'base', 'pubarms', 'ref']:
        q = pd.read_parquet(C + rf'\pred_{v}_{g}.parquet')
        if P is None: P = q
        else:
            assert (P[['cc', 'aid', 's', 'y']].values == q[['cc', 'aid', 's', 'y']].values).all()
            assert np.allclose(P.w.values, q.w.values)
            for m in q.columns:
                if m in MOD: P[m] = q[m].values
    u = pd.concat([pd.read_parquet(C + rf'\u_{cc}_{v}.parquet', columns=['aid', 's', 'y', 'w']) for cc in ['CN', 'EP']], ignore_index=True)
    assert (u.aid.values == P.aid.values).all() and (u.y.values == P.y.values).all() and np.allclose(u.w.values, P.w.values)
    fr.append(P)
res = {}
def contr(fs, wts=None):
    A = []
    for k, f in enumerate(fs):
        w = f.w.values.astype(float) * (1 if wts is None else wts[k])
        y = f.y.values.astype(float)
        A.append({m: wauc(y, f[m].values.astype(float), w) for m in MOD})
    c = {}
    c['d_v'] = [min(ds(a['prov'], a['free']), ds(a['prov'], a['hsc'])) for a in A]
    c['E_prov_to_pub'] = [ds(a['all'], a['pubctx']) for a in A]
    c['E_pub_to_prov'] = [ds(a['all'], a['prov']) for a in A]
    c['prov_vs_scagg'] = [ds(a['prov'], a['scagg']) for a in A]
    c['prov_vs_pub'] = [ds(a['prov'], a['pub']) for a in A]
    c['auc_gap_prov_bestpub'] = [a['prov'] - max(a['free'], a['hsc']) for a in A]
    return A, c
A, c = contr(fr)
res['auc'] = A; res['point'] = {k: (v, float(np.mean(v))) for k, v in c.items()}
print(json.dumps(res['point'], indent=0))
# 기업 군집 다항 재표집(원점 공유), B=100
B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
keys = pd.Index(pd.concat([f.cc + ':' + f.aid.astype(str) for f in fr]).unique())
idx = [keys.get_indexer(f.cc + ':' + f.aid.astype(str)) for f in fr]
rng = np.random.default_rng(777)
bs = {k: [] for k in c}
for b in range(B):
    cnt = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
    _, cb = contr(fr, [cnt[i] for i in idx])
    for k in c: bs[k].append(np.mean(cb[k]))
res['ci_firm_multinomial'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
print(json.dumps(res['ci_firm_multinomial'], indent=0))
# 귀무 요약(b 중복 제거)
nul = {}
for fn in glob.glob(C + r'\p5null_*.json'):
    j = json.load(open(fn))
    for n in j.get('nulls', []):
        if n['b'] in nul: assert abs(nul[n['b']] - n['G']) < 1e-9, (n['b'], fn)
        nul[n['b']] = n['G']
G = np.array([nul[b] for b in sorted(nul)])
res['null'] = dict(n=len(G), bmin=min(nul), bmax=max(nul), mean=float(G.mean()), sd=float(G.std(ddof=1)), q95=float(np.percentile(G, 95)), max=float(G.max()),
                   share_reproduced=float(G.mean() / 0.12600339714959405), resid=0.12600339714959405 - float(G.mean()))
print(res['null'])
json.dump(res, open(W + r'\verify_0\v4_out.json', 'w'), indent=1)
