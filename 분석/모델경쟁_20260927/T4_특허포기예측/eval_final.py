# 최종 후보(v37) 검증 지표: 출원인 군집 부트스트랩 CI, 기준선 대비 차이, 세그먼트별 AUC
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score, average_precision_score
from run import *
ev = get_frame(EVAL_L, None)
p37 = pd.read_parquet('preds/pred_v37.parquet'); p0 = pd.read_parquet('preds/pred_v0.parquet')
assert (p37.pid.values == ev.pid.values).all()
ev['p'] = p37.p.values; ev['p0'] = p0.p.values
rng = np.random.default_rng(0)
g = ev.groupby('acd').indices; keys = list(g.keys()); idxs = [g[k] for k in keys]
res = []
for b in range(200):
    s = rng.integers(0, len(keys), len(keys)); ii = np.concatenate([idxs[i] for i in s])
    y = ev.y.values[ii]
    a1 = roc_auc_score(y, ev.p.values[ii]); a0 = roc_auc_score(y, ev.p0.values[ii])
    pr1 = average_precision_score(y, ev.p.values[ii]); pr0 = average_precision_score(y, ev.p0.values[ii])
    res.append((a1, a0, a1 - a0, (a1 - a0) / (1 - a0), pr1, pr0))
r = np.array(res)
out = {'n_rows': int(len(ev)), 'n_patents': int(ev.pid.nunique()), 'n_applicant_clusters': int(len(keys)), 'events': int(ev.y.sum()),
       'auc_v37': roc_auc_score(ev.y, ev.p), 'auc_v0': roc_auc_score(ev.y, ev.p0),
       'prauc_v37': average_precision_score(ev.y, ev.p), 'prauc_v0': average_precision_score(ev.y, ev.p0)}
for i, k in enumerate(['auc_v37', 'auc_v0', 'auc_diff', 'skill', 'prauc_v37', 'prauc_v0']):
    out[k + '_ci95'] = list(np.percentile(r[:, i], [2.5, 97.5]).round(4))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open('final_models/bootstrap_v37.json', 'w'), indent=1, ensure_ascii=False)
# 세그먼트
pt = T.load()['pt']
ev['ageb'] = pd.cut(ev.age, [0, 1, 2, 3, 4, 6, 8, 10, 20])
ev['size'] = pd.cut(ev.port_n, [0, 1, 10, 100, 1000, 1e9], labels=['1', '2-10', '11-100', '101-1000', '1000+'])
ev['rgy'] = ev.rgst_year
def seg(col):
    return ev.groupby(col, observed=True).apply(lambda d: pd.Series({'n': len(d), 'events': int(d.y.sum()), 'rate': d.y.mean(),
            'auc_v0': roc_auc_score(d.y, d.p0) if d.y.nunique() > 1 else np.nan, 'auc_v37': roc_auc_score(d.y, d.p) if d.y.nunique() > 1 else np.nan,
            'mean_p': d.p.mean()})).round(3)
with open('final_models/segments_v37.txt', 'w', encoding='utf-8') as fo:
    for col in ['atype', 'ageb', 'size', 'rgy', 'ksic', 'L']:
        s = seg(col); fo.write(f'## {col}\n{s.to_string()}\n\n'); print(col); print(s.to_string())
