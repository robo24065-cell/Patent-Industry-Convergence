# verify_0 V3: P4 D·d_v와 반박 (i)(ii) 수치를 저장 예측으로 독립 재계산(evalkit12·s4 미사용).
# sklearn roc_auc_score, 다항 군집 재표집(기업 (cc,aid), 서브클래스 s), B=100.
import sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
W = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C12/C12-T4'
t = time.time()
M = ['prov', 'free', 'hsc', 'all', 'pubctx', 'scagg', 'pub']
fr = []
for v in range(2020, 2025):
    a = pd.read_parquet(f'{W}/cache/pred_{v}_base.parquet')
    for g in ['pubarms', 'ref']:
        q = pd.read_parquet(f'{W}/cache/pred_{v}_{g}.parquet')
        assert (q.aid.values == a.aid.values).all() and (q.s.values == a.s.values).all()
        for m in q.columns:
            if m in M:
                a[m] = q[m].values
    fr.append(a)
allf = pd.concat(fr)
firms = pd.Index((allf.cc + '_' + allf.aid.astype(str)).unique()); scs = pd.Index(np.unique(allf.s))
FI = [firms.get_indexer(f.cc + '_' + f.aid.astype(str)) for f in fr]; SI = [scs.get_indexer(f.s) for f in fr]


def ds(a, b):
    return (a - b) / (1 - b)


def contr(aucs):
    d = [min(ds(x['prov'], x['free']), ds(x['prov'], x['hsc'])) for x in aucs]
    return {'D': float(np.mean(d)), 'd_v': d,
            'E_prov_to_pub': float(np.mean([ds(x['all'], x['pubctx']) for x in aucs])),
            'E_pub_to_prov': float(np.mean([ds(x['all'], x['prov']) for x in aucs])),
            'prov_vs_scagg': float(np.mean([ds(x['prov'], x['scagg']) for x in aucs])),
            'prov_vs_pub': float(np.mean([ds(x['prov'], x['pub']) for x in aucs]))}


def aucs_w(wfun):
    out = []
    for k, f in enumerate(fr):
        w = f.w.values * wfun(k)
        out.append({m: roc_auc_score(f.y.values, f[m].values, sample_weight=w) for m in M})
    return out


pt = contr(aucs_w(lambda k: 1.0))
print('point', pt, round(time.time() - t, 1), flush=True)
rng = np.random.default_rng(4242)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 40
bs = {'firm': [], 'two': []}
for b in range(B):
    cf = rng.multinomial(len(firms), np.full(len(firms), 1 / len(firms))).astype(float)
    cs = rng.multinomial(len(scs), np.full(len(scs), 1 / len(scs))).astype(float)
    bs['firm'].append(contr(aucs_w(lambda k: cf[FI[k]])))
    bs['two'].append(contr(aucs_w(lambda k: cf[FI[k]] * cs[SI[k]])))
    if time.time() - t > 500:
        break
res = {'point': pt, 'B_done': len(bs['firm'])}
for kd in bs:
    res['ci_' + kd] = {k: [float(np.percentile([x[k] for x in bs[kd]], 2.5)), float(np.percentile([x[k] for x in bs[kd]], 97.5))]
                       for k in ['D', 'E_prov_to_pub', 'E_pub_to_prov', 'prov_vs_scagg']}
    res['se_' + kd] = {k: float(np.std([x[k] for x in bs[kd]], ddof=1)) for k in ['D', 'prov_vs_scagg']}
print(json.dumps(res), round(time.time() - t, 1), flush=True)
json.dump(res, open(f'{W}/verify_0/v3_out.json', 'w'), indent=1)
