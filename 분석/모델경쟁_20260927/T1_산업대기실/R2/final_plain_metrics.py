# 최종시험 결과의 비전문가용 지표(상위 K 적중률·포착률) — 최종 채점 후 산출(모형·판정 변경 없음)
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding='utf-8')
R2 = pathlib.Path(__file__).resolve().parent; T1 = R2.parent; CACHE = T1/'cache'; C, END = 2017, '20250822'
o = pd.read_parquet(R2/'runs/FROZEN_final/o2017.parquet', columns=['firm', 'k', 'newc', 'px', 'pl'])
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}
a = pd.read_parquet(CACHE/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic']); a = a[a.ay >= 2000].copy()
a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
p = a.pub_date.fillna('').astype(str); r = a.rgst_date.fillna('').astype(str); vis = ((p.str.len() == 8) & (p <= END)) | ((r.str.len() == 8) & (r <= END))
ok = pd.MultiIndex.from_arrays([o.firm.values, o.k.values])
prior = ok.isin(pd.MultiIndex.from_frame(a[a.ay <= C][['applicant_cd', 'k']].drop_duplicates()))
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay']); ins = ins[(ins.ay > C) & (ins.ay <= C + 5)].copy(); ins['k'] = ins.ksic.map(kidx)
y = np.maximum(ok.isin(pd.MultiIndex.from_frame(ins[['applicant_cd', 'k']].dropna().astype({'k': int}).drop_duplicates())),
               ok.isin(pd.MultiIndex.from_frame(a[vis & (a.ay > C) & (a.ay <= C + 5)][['applicant_cd', 'k']].drop_duplicates()))).astype(int)
keep = ~prior; y = y[keep]; n = keep.sum(); s = 0.9 * rankdata(o.px.values[keep]) / n + 0.1 * rankdata(o.pl.values[keep]) / n
bs = pd.read_parquet(R2/'cache_final/o2017/baseline_scores.parquet').loc[keep].reset_index(drop=True)['xgb_y_u'].values
def topk(score, yy, frac=None, k=None):
    k = k or int(round(len(yy) * frac)); idx = np.argsort(-score, kind='stable')[:k]; return dict(k=int(k), hits=int(yy[idx].sum()), precision=float(yy[idx].mean()), recall=float(yy[idx].sum() / yy.sum()))
out = dict(base_rate=float(y.mean()), n=int(n), pos=int(y.sum()))
for nm, sc in (('model', s), ('baseline_xgb_y_u', bs)):
    out[nm] = {'top1000': topk(sc, y, k=1000), 'top1pct': topk(sc, y, 0.01), 'top5pct': topk(sc, y, 0.05), 'top10pct': topk(sc, y, 0.10)}
# 산업별 상위 20 법인 적중(보고서의 '산업별 대기실 명단' 사용 방식)
d = pd.DataFrame({'k': o.k.values[keep], 's': s, 'b': bs, 'y': y})
out['per_ksic_top20_precision_model'] = float(d.sort_values('s', ascending=False).groupby('k').head(20).y.mean())
out['per_ksic_top20_precision_baseline'] = float(d.sort_values('b', ascending=False).groupby('k').head(20).y.mean())
json.dump(out, open(R2/'final_plain_metrics.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1, ensure_ascii=False))
