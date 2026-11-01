# C19-T1 verify_0: duckdb 없이 pyarrow+pandas로 원 메타에서 급증 행 수·VJ1 제외 수·이중 수록 수·포착률 재계산, 저장 예측에서 Δskill 재계산
import sys, os, glob, json; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import W, C, MC, out
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
files = sorted(glob.glob(os.path.join(MC, '_shared', 'meta', '*.parquet')))
cols = ['appl_no', 'appl_date', 'appl_year', 'rgst_no', 'rgst_date', 'pub_date', 'title', 'cc']
parts = []
for f in files:
    t = pq.read_table(f, columns=cols)
    t = t.filter(pc.equal(pc.cast(t['cc'], pa.string()), 'JP'))
    parts.append(t.drop(['cc']).to_pandas())
d = pd.concat(parts, ignore_index=True)
for c in ('rgst_no', 'rgst_date', 'pub_date'):
    d[c] = d[c].replace('', np.nan)
R = {'n_jp_rows': int(len(d)), 'n_jp_unique_appl': int(d.appl_no.nunique())}
d['nopub'] = d.pub_date.isna(); d['hasrg'] = d.rgst_date.notna()
# T3 공개연도: least(pub, rgst) 문자열 비교(YYYYMMDD)
pp_, rr_ = d.pub_date.fillna('99999999'), d.rgst_date.fillna('99999999')
lp = np.where(pp_ <= rr_, pp_, rr_); lp = np.where(lp == '99999999', None, lp)
d['py'] = pd.Series(lp).str[:4]
d['u'] = d.appl_no.str.endswith('U')
sp = d[(d.appl_year >= 2000) & d.nopub & d.hasrg]
R['surge_rows_patent_py'] = {y: int(((sp.py == y) & ~sp.u).sum()) for y in ('2020', '2021', '2022', '2023')}
R['surge_rows_all_py'] = {y: int((sp.py == y).sum()) for y in ('2021', '2022')}
R['vj1_rows_py2000_2024'] = int(sp.py.astype(float).between(2000, 2024).sum())
R['pubno10_rows'] = int((d.nopub & d.rgst_no.fillna('').str.fullmatch(r'\d{10}')).sum())
# 독립 정규화 키(파이썬): 8자리 → 1988+앞2 & 일련6, 10자리 → 그대로
def key(a):
    u = a.endswith('U'); s = a[:-1] if u else a
    if s.isdigit() and len(s) == 8: return f'{1988 + int(s[:2])}{s[2:]}' + ('U' if u else '')
    if s.isdigit() and len(s) == 10: return s + ('U' if u else '')
    return a
d['k'] = d.appl_no.map(key)
d['tn'] = d.title.fillna('').str.replace(r'\s+', '', regex=True).str.lower()
pubd = d[~d.nopub]
kset = pubd.groupby('k').appl_no.apply(set).to_dict()
tset = pubd[pubd.tn != ''].groupby(['tn', 'appl_date']).appl_no.apply(set).to_dict()
npx = d[d.nopub & d.hasrg].copy()
npx['k1'] = [bool(kset.get(k, set()) - {a}) for k, a in zip(npx.k, npx.appl_no)]
npx['k2'] = [bool(tset.get((t, ad), set()) - {a}) if t != '' else False for t, ad, a in zip(npx.tn, npx.appl_date, npx.appl_no)]
npx['dup'] = npx.k1 | npx.k2
R['dup_nopub_all'] = int(npx.dup.sum())
R['dup_nopub_a2000'] = int(npx[npx.appl_year >= 2000].dup.sum())
R['dup_nopub_a2000_py2000_2024'] = int(npx[(npx.appl_year >= 2000) & npx.py.astype(float).between(2000, 2024)].dup.sum())
R['dup_by_year_patent'] = {y: int(npx[(npx.rgst_date.str[:4] == y) & ~npx.u].dup.sum()) for y in ('2020', '2021', '2022')}
# 포착률(특허, 공개일 없음, 등록연도) / 特許公報
koho = {2016: 217778, 2017: 199592, 2018: 191073, 2019: 180399, 2020: 179017, 2021: 181368, 2022: 210998}
cnt = npx[~npx.u].rgst_date.str[:4].value_counts()
cap = {y: cnt.get(str(y), 0) / koho[y] for y in koho}
rb = np.mean([cap[y] for y in range(2016, 2021)])
R['rel_nopub_2021_2022'] = [cap[2021] / rb, cap[2022] / rb]
exc = sum(cnt.get(str(y), 0) - rb * koho[y] for y in (2021, 2022)); dupc = sum(R['dup_by_year_patent'][str(y)] for y in (2021, 2022))
R['dup_share_of_excess'] = dupc / exc
k1c = {y: int(npx[(npx.rgst_date.str[:4] == y) & ~npx.u].k1.sum()) for y in ('2016', '2020', '2021', '2022')}
R['dup_k1only_by_year_patent'] = k1c
R['dup_k1only_share_of_excess'] = (k1c['2021'] + k1c['2022']) / exc
R['excess_2y'] = exc
# Δskill 재계산(저장 예측, sklearn)
T = os.path.join(C, 't3res'); P = pd.read_parquet(os.path.join(T, 'preds_P0_final.parquet'))
sk = lambda a, b: (a - b) / (1 - b)
s0 = sk(roc_auc_score(P.yb, P.p), roc_auc_score(P.yb, P.p_base))
R['skill_P0'] = s0
pj = {}
for f in sorted(glob.glob(os.path.join(T, 'preds_*_final.parquet'))):
    v = os.path.basename(f)[6:-14]; V = pd.read_parquet(f)
    pj[v] = sk(roc_auc_score(V.yb, V.p), roc_auc_score(V.yb, V.p_base)) - s0
R['delta_final_sklearn'] = pj
json.dump(R, open(os.path.join(W, 'verify_0', 'verify_0_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
out(json.dumps(R, ensure_ascii=False, indent=1, default=str))
