# verify_1: fr_K1_{v}(_lr).parquet에서 sklearn 가중 AUC로 제공-SCAGG/FREE/HSC 재계산(읽기만)
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
C = sys.argv[1]; OUT = sys.argv[2]
R = {}
for L in ['', '_lr']:
    for v in [2020, 2021, 2022, 2023, 2024]:
        F = pd.read_parquet(C + rf'\fr_K1_{v}{L}.parquet')
        w = F.w.values.astype(float)
        a = {m: roc_auc_score(F.y.values, F[m].values, sample_weight=w) for m in ['base', 'prov', 'free', 'hsc', 'scagg']}
        # 층화
        st = {}
        for m in a:
            num = den = 0.0
            for kk, g in F.groupby('k'):
                if g.y.nunique() < 2: continue
                wp = g.w[g.y == 1].sum(); wn = g.w[g.y == 0].sum()
                num += roc_auc_score(g.y.values, g[m].values, sample_weight=g.w.values) * wp * wn; den += wp * wn
            st[m] = num / den
        corr = float(np.corrcoef(F.prov.values, F.scagg.values)[0, 1])
        R[f'{v}{L}'] = {'auc': a, 'strat': st, 'corr_prov_scagg_score': corr}
        print(v, L, {m: round(a['prov'] - a[m], 5) for m in ['free', 'hsc', 'scagg']}, 'strat', round(st['prov'] - st['scagg'], 5), 'corr', round(corr, 4), flush=True)
for L in ['', '_lr']:
    ks = [k for k in R if (k.endswith('_lr') if L else not k.endswith('_lr'))]
    for m in ['free', 'hsc', 'scagg', 'base']:
        d = [R[k]['auc']['prov'] - R[k]['auc'][m] for k in ks]
        ds = [R[k]['strat']['prov'] - R[k]['strat'][m] for k in ks]
        print('MEAN', L or 'xgb', m, round(np.mean(d), 6), 'pos', sum(x > 0 for x in d), 'strat', round(np.mean(ds), 6))
    gb = np.mean([R[k]['auc']['prov'] - R[k]['auc']['base'] for k in ks]); gs = np.mean([R[k]['auc']['scagg'] - R[k]['auc']['base'] for k in ks])
    print('scagg share of prov gain over base', L or 'xgb', round(gs / gb, 3))
json.dump(R, open(OUT, 'w'), indent=1)
