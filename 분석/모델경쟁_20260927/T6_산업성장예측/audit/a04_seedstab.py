# 감사 4: v38 vs v41 vs v45 시드 집합 안정성 + 쌍대 부트스트랩
import sys, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H); os.chdir(H)
import numpy as np, pandas as pd, importlib
import final_run as fr
R = importlib.import_module('70_run2')
F = fr.build_panel(2019, 2019)
b = pd.read_parquet('preds/baseline.parquet'); bm = np.abs(b.y).mean()
res = {}
for name in ['v38', 'v41', 'v45']:
    sp = R.V[name]['spec']
    sks = []
    for k in range(4):
        s2 = dict(sp); s2['seeds'] = list(range(5 * k, 5 * k + 5))
        p = fr.rolling(F, fr.VAL_YEARS, s2)
        sks.append(1 - (p.y - p.pred).abs().mean() / bm)
    res[name] = sks
    print(name, np.round(sks, 4), 'mean', round(np.mean(sks), 4))
a = pd.read_parquet('preds/v41.parquet'); c = pd.read_parquet('preds/v38.parquet')
m = a.merge(c, on=['ksic', 'year', 'y'], suffixes=('41', '38'))
m['d'] = (m.y - m.pred41).abs() - (m.y - m.pred38).abs()
g = m.groupby('ksic').d.agg(['sum', 'count']); rng = np.random.default_rng(0)
bs = [(lambda i: g['sum'].values[i].sum() / g['count'].values[i].sum())(rng.integers(0, len(g), len(g))) for _ in range(5000)]
print('v41-v38 MAE diff', m.d.mean().round(5), 'CI', np.percentile(bs, [2.5, 97.5]).round(5))
