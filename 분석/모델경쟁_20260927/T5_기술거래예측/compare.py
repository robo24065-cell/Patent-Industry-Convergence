# 버전 비교: 양도인 군집 쌍대 부트스트랩 ΔAUC, 번들 크기별 AUC, 상위 10 번들 양도인 제외 AUC
# python compare.py v12 v13 [v5 ...]
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import roc_auc_score
from common import *
vs = sys.argv[1:]
va = load_panel(VAL_L, ['rgst_no', 'L', 'cur', 'y_MAIN', 'y_date'])
for v in vs:
    va = va.merge(pd.read_parquet(f'cache/pred_{v}.parquet').rename(columns={'p': v}), on=['rgst_no', 'L'])
pos = va.y_MAIN == 1
va['bundle'] = 0; va.loc[pos, 'bundle'] = va[pos].groupby(['L', 'cur', 'y_date']).rgst_no.transform('size').values
top = va[pos].cur.value_counts().head(10).index
res = []
for v in vs:
    r = dict(v=v, auc=roc_auc_score(va.y_MAIN, va[v]))
    for lo, hi in [(1, 1), (2, 5), (6, 50), (51, 10 ** 6)]:
        k = (~pos) | ((va.bundle >= lo) & (va.bundle <= hi)); r[f'b{lo}-{hi}'] = roc_auc_score(va.y_MAIN[k], va[v][k])
    k = ~va.cur.isin(top); r['ex_top10'] = roc_auc_score(va.y_MAIN[k], va[v][k])
    res.append(r)
print(pd.DataFrame(res).round(4).to_string())
# 쌍대 부트스트랩 (첫 버전 대비)
rng = np.random.default_rng(1)
codes, uniq = pd.factorize(va.cur.fillna('NA')); order = np.argsort(codes, kind='stable'); cs = codes[order]
st = np.searchsorted(cs, np.arange(len(uniq))); en = np.searchsorted(cs, np.arange(len(uniq)), side='right')
y = va.y_MAIN.values; P = {v: va[v].values for v in vs}
D = {v: [] for v in vs[1:]}
for b in range(200):
    idx = np.concatenate([order[st[c]:en[c]] for c in rng.integers(0, len(uniq), len(uniq))])
    a0 = roc_auc_score(y[idx], P[vs[0]][idx])
    for v in vs[1:]: D[v].append(roc_auc_score(y[idx], P[v][idx]) - a0)
for v in vs[1:]:
    d = np.array(D[v]); print(f'{v} - {vs[0]}: dAUC mean {d.mean():.4f} 95%CI [{np.percentile(d, 2.5):.4f}, {np.percentile(d, 97.5):.4f}] P(d>0)={np.mean(d > 0):.2f}')
