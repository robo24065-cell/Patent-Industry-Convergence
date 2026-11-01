# x2: 양성 분포 TVD 경고의 표본잡음 기준선 — 개발 각 원점 양성 vs 나머지 개발 원점 양성(원점 밖) TVD, 그리고 n=255 무작위 재표본 TVD.
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import C, DEV, LOCK, W
u = pd.read_parquet(os.path.join(C, 'units.parquet'), columns=['key', 's0', 'y', 'ind2', 'sido'])
D = u[u.s0.isin(DEV) & (u.y == 1)]
tvd = lambda a, b: 0.5 * (a.reindex(a.index.union(b.index), fill_value=0) - b.reindex(a.index.union(b.index), fill_value=0)).abs().sum()
res = {}
rng = np.random.default_rng(0)
for v in ['ind2', 'sido']:
    loo = [round(float(tvd(D[D.s0 == s][v].value_counts(normalize=True), D[D.s0 != s][v].value_counts(normalize=True))), 4) for s in DEV]
    pool = D[v].value_counts(normalize=True)
    sim = [tvd(D[v].sample(255, replace=True, random_state=int(rng.integers(1e9))).value_counts(normalize=True), pool) for _ in range(500)]
    res[v] = dict(dev_origin_vs_rest=loo, dev_origin_range=[min(loo), max(loo)], sim255_mean=round(float(np.mean(sim)), 4), sim255_p95=round(float(np.quantile(sim, 0.95)), 4))
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(os.path.join(C, 'x2_tvd_ref.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
