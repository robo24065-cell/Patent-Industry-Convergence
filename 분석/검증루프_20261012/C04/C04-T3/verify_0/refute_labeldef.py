import sys, hashlib, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_parquet(sys.argv[1])
d['kr_fut'] = np.rint((d.kr_W + 1) * np.exp(d.y) - 1)
err = np.abs((d.kr_W + 1) * np.exp(d.y) - 1 - d.kr_fut).max()
d['inc'] = d.kr_fut - d.kr_W
d['h'] = [int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in d.cell]
k = int(round(0.1 * len(d)))
# 공식 재현: y 내림 -> inc 내림 -> md5
o = d.sort_values(['y', 'inc', 'h'], ascending=[False, False, True]).index[:k]
off = d.index.isin(o).astype(int)
res = dict(n=len(d), k=k, rint_err=float(err), official_reproduced_mismatch=int((off != d.yb).sum()))
# 글자 그대로: 절대 증가 상위 10% (동률 -> md5, 대안 동률 y)
for nm, keys, asc in [('abs_tie_md5', ['inc', 'h'], [False, True]), ('abs_tie_y', ['inc', 'y', 'h'], [False, False, True])]:
    oo = d.sort_values(keys, ascending=asc).index[:k]; lab = d.index.isin(oo).astype(int)
    a = roc_auc_score(lab, d.p); b = roc_auc_score(lab, d.p_base)
    res[nm] = dict(overlap=int((lab & d.yb).sum()), auc=a, auc_base=b, skill=(a - b) / (1 - b),
                   krW_le1=int(((lab == 1) & (d.kr_W <= 1)).sum()), min_inc=float(d.inc[lab == 1].min()),
                   boundary_inc=float(d.inc[lab == 1].min()), n_at_boundary=int((d.inc == d.inc[lab == 1].min()).sum()))
res['official'] = dict(auc=roc_auc_score(d.yb, d.p), auc_base=roc_auc_score(d.yb, d.p_base),
                       krW_le1=int(((d.yb == 1) & (d.kr_W <= 1)).sum()), min_inc=float(d.inc[d.yb == 1].min()),
                       krW0_min_inc=float(d.inc[(d.yb == 1) & (d.kr_W == 0)].min()),
                       krW0_inc2_n=int(((d.yb == 1) & (d.kr_W == 0) & (d.inc == 2)).sum()),
                       y_threshold=float(d.y[d.yb == 1].min()))
# 상대 증가(비율) 대안: kr_fut/kr_W (kr_W=0 제외 불가) -> 생략. 절대 증가로 스코어 직접 비교: 모형 p vs inc 상관
res['spearman_p_inc'] = float(pd.Series(d.p).corr(d.inc, method='spearman'))
res['spearman_p_y'] = float(pd.Series(d.p).corr(d.y, method='spearman'))
json.dump(res, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
