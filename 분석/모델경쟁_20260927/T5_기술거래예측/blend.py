# 예측 결합: python blend.py <out_ver> <mode> <v_a> <v_b> [w]
# mode geo: exp(w*log pa + (1-w)*log pb) / rank: w*rank(a)+(1-w)*rank(b)
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from common import *
from evalkit import *
out, mode, a, b = sys.argv[1:5]; w = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
va = load_panel(VAL_L, ['rgst_no', 'L', 'cur', 'y_MAIN'])
d = va.merge(pd.read_parquet(f'cache/pred_{a}.parquet').rename(columns={'p': 'a'}), on=['rgst_no', 'L']) \
      .merge(pd.read_parquet(f'cache/pred_{b}.parquet').rename(columns={'p': 'b'}), on=['rgst_no', 'L'])
if mode == 'geo':
    d['p'] = np.exp(w * np.log(d.a.clip(1e-6)) + (1 - w) * np.log(d.b.clip(1e-6)))
else:
    d['p'] = w * d.a.rank(pct=True) + (1 - w) * d.b.rank(pct=True)
mt = metrics(d.y_MAIN, d.p); ci = cluster_boot(d.y_MAIN.values, d.p.values, d.cur.values)
print(out, {k: round(x, 4) for k, x in mt.items()}, ci[0].round(4), ci[1].round(4))
for L in VAL_L:
    k = (d.L == L).values; print('  L', L, {kk: round(x, 4) for kk, x in metrics(d.y_MAIN[k], d.p[k]).items()})
d[['rgst_no', 'L', 'p']].to_parquet(f'cache/pred_{out}.parquet', index=False)
json.dump(dict(auc=mt['auc'], prauc=mt['prauc'], p1=mt['p1'], ci_auc=list(ci[0]), ci_pr=list(ci[1])), open(f'cache/metrics_{out}.json', 'w'))
