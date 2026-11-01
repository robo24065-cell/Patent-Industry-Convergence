# 트렌드 산출물 초안(검증 구간 기준). 모든 사건은 < 2019-01-01 (홀드아웃 결과창 미접촉).
#  trend_ksic_year.csv : KSIC×L 관측 3년 양도율 + 예측평균(표본구분: in_sample L<=2011 / label_overlap 2012-13 / out_of_sample 2014-15)
#  ksic_rank_val.csv   : L2014·2015 KSIC별 예측 vs 관측 양도율 + 스피어만
#  flow_type_year.csv  : 양도 사건 연도(2003-2018)별 양도인유형→양수인유형 건수 (MAIN), 합병·분할(CORP)·상속(INH) 연도별 건수
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr
from common import *
TN = {1: '국내법인', 2: '대학', 3: '연구기관', 4: '국내개인', 5: '외국', 7: '국가·공공', 0: '미상'}
KN = pd.read_csv(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv",
                 encoding='utf-8-sig', usecols=['세세분류코드', '산업명']).drop_duplicates('세세분류코드').set_index('세세분류코드').산업명
a = pd.read_parquet('cache/final_2006_2011_2006-2007-2008-2009-2010-2011.parquet')
b = pd.read_parquet('cache/final_2006_2011_2012-2013-2014-2015.parquet')
d = pd.concat([a, b]); d['ksic'] = d.ksic.astype(int)
d['sample'] = np.select([d.L <= 2011, d.L <= 2013], ['in_sample', 'label_overlap'], 'out_of_sample')
t = d.groupby(['ksic', 'L', 'sample']).agg(n=('p', 'size'), obs_rate=('y_MAIN', 'mean'), pred_mean=('p', 'mean'), pos=('y_MAIN', 'sum')).reset_index()
t['산업명'] = t.ksic.map(KN)
t.round(5).to_csv('trend_ksic_year.csv', index=False, encoding='utf-8-sig')
tot = d.groupby(['L', 'sample']).agg(n=('p', 'size'), obs_rate=('y_MAIN', 'mean'), pred_mean=('p', 'mean')).reset_index()
print(tot.round(4).to_string())
# 검증 KSIC 순위
out = []
for L in VAL_L:
    k = t[t.L == L]
    rho = spearmanr(k.pred_mean, k.obs_rate)
    print('L', L, 'KSIC 예측평균 vs 관측률 spearman', round(rho.correlation, 3), 'p', round(rho.pvalue, 4))
    out.append(k.assign(rank_pred=k.pred_mean.rank(ascending=False), rank_obs=k.obs_rate.rank(ascending=False), spearman=rho.correlation))
r = pd.concat(out); r.round(5).to_csv('ksic_rank_val.csv', index=False, encoding='utf-8-sig')
top = r[r.L == 2015].sort_values('rank_pred').head(10)[['ksic', '산업명', 'pred_mean', 'obs_rate', 'rank_obs']]
print('L2015 예측 상위 10 산업 (2015-2017 거래 활성 예상)\n', top.round(4).to_string())
# 흐름 유형 추이
own = pd.read_parquet('cache/own.parquet'); own['d'] = pd.to_datetime(own.d, format='%Y%m%d', errors='coerce')
assert own.d.max() < HOLD_CUT
own['cls'] = own.rs.map(classify); own['nm'] = own.ownerName.map(norm)
own = own.sort_values(['rgst_no', 'd', 'seq'])
g = own.groupby(['rgst_no', 'd'], sort=False).agg(prim=('nm', 'first'), raw=('ownerName', 'first'), natl=('ownerNatl', 'first'),
                                                   cls=('cls', lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index()
g = g.sort_values(['rgst_no', 'd']); g['k'] = g.groupby('rgst_no').cumcount(); g.loc[g.k == 0, 'cls'] = None
app = pd.read_parquet('cache/app.parquet'); app['nm'] = app.applicantName.map(norm); app['pref'] = app.applicantCd.str[:1]
n2p = app.dropna(subset=['pref']).groupby('nm').pref.agg(lambda s: s.value_counts().index[0]).to_dict()
def ot(nm, raw, natl):
    p = n2p.get(nm)
    if p is not None and p.isdigit():
        p = int(p)
        if p == 2: return 2 if ('대학' in str(raw) or '산학' in str(raw)) else 7
        return 5 if p == 6 else p
    return name_type(raw, natl)
g['otype'] = [ot(x, y, z) for x, y, z in zip(g.prim, g.raw, g.natl)]
g['from_type'] = g.groupby('rgst_no').otype.shift(1)
e = g[g.cls.notna()].copy(); e['year'] = e.d.dt.year
e = e[e.year >= 2003]
m = e[e.cls == 'MAIN']
f = m.groupby(['year', 'from_type', 'otype']).size().reset_index(name='n')
f['flow'] = f.from_type.map(TN) + '→' + f.otype.map(TN)
f['deals'] = m.groupby(['year', 'from_type', 'otype']).apply(lambda x: x.groupby(['d', 'prim']).ngroups).values
f.to_csv('flow_type_year.csv', index=False, encoding='utf-8-sig')
piv = f.pivot_table(index='flow', columns='year', values='n', aggfunc='sum').fillna(0)
piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index].head(10)
print(piv.astype(int).to_string())
c = e.groupby(['year', 'cls']).rgst_no.nunique().unstack().fillna(0).astype(int); print(c.to_string())
c.to_csv('transfer_class_year.csv', encoding='utf-8-sig')
