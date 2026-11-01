# 사후 진단 2: 대형 출원인 2위의 KSIC 분포, 상위 10 출원인 제외 시 산업 순위상관 (모형 수정 없음)
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OD = os.path.join(R2, 'runs_final', 'F22')
ev = pd.read_parquet(os.path.join(OD, 'ev.parquet'), columns=['pid', 'ksic', 'acd', 'atype', 'y'])
spec = json.load(open(os.path.join(R2, 'final_spec_r2.json'), encoding='utf-8'))['spec']; W = spec['ensemble']['weights']
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
pf = {c: np.load([f for f in glob.glob(os.path.join(OD, 'pred', f'{c}_*.npy')) if '_b' not in os.path.basename(f)[len(c) + 1:]][0]) for c in list(W) + ['base']}
ev['p'] = 1 / (1 + np.exp(-sum(W[c] * lg(pf[c]) for c in W) / sum(W.values()))); ev['pb'] = pf['base']
ev['cl'] = np.where(ev.acd == 'NA', 'P' + ev.pid.astype(str), ev.acd)
cs = ev.cl.value_counts()
out = {}
a2 = ev[ev.cl == cs.index[1]]
out['applicant2'] = dict(atype=a2.atype.iloc[0], n=len(a2), top_ksic=a2.ksic.value_counts().head(5).to_dict(), realized=float(a2.y.mean()), pred=float(a2.p.mean()), base=float(a2.pb.mean()))
for k in (1, 2, 10):
    m = ~ev.cl.isin(cs.index[:k])
    g = ev[m].groupby('ksic').agg(y=('y', 'mean'), p=('p', 'mean'), pb=('pb', 'mean'))
    out[f'industry_spearman_excl_top{k}'] = dict(model=float(spearmanr(g.p, g.y)[0]), baseline=float(spearmanr(g.pb, g.y)[0]))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(R2, 'audit', 'diag_final2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
