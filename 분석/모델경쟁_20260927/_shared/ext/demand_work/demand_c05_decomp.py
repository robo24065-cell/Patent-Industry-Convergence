# 가격-물량 분해: KOSIS 5자리 출하액(명목) = ECOS 가격(PPI 품목/기본분류) x 실질물량
# 수요형(가격↑·물량↑) / 공급·기술형(가격↓·물량↑) / 위축형 구분 -> 이후 3년 특허 성장 비교 (결과 ≤2016)
import pandas as pd, numpy as np, pathlib, statsmodels.formula.api as smf
OUT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext")
W = OUT / 'demand_work'
ks = pd.read_csv(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv", encoding='utf-8-sig')
ks = ks[(ks['기준'] == '10차') & (ks['항목코드'] == 'T04')][['세세분류코드', '연도', '값']]
ks.columns = ['ksic', 'year', 'ship']; ks['ksic'] = ks.ksic.astype(str)
g = pd.read_parquet(OUT / 'demand_ksic_price_panel_1996_2021.parquet')
# 품목 우선, 없으면 기본분류
pi = g[g.series == 'ppi_item'][['ksic', 'year', 'lidx']].rename(columns={'lidx': 'lp_item'})
pb = g[g.series == 'ppi_basic'][['ksic', 'year', 'lidx']].rename(columns={'lidx': 'lp_basic'})
x = ks.merge(pi, on=['ksic', 'year'], how='left').merge(pb, on=['ksic', 'year'], how='left')
x['lp'] = x.lp_item.fillna(x.lp_basic)
x = x.dropna(subset=['lp', 'ship']).sort_values(['ksic', 'year'])
x['lq'] = np.log(x.ship.clip(lower=1)) - x.lp
x.to_parquet(OUT / 'demand_ksic_price_volume_2008_2020.parquet', index=False)
print('분해 가능 KSIC:', x.ksic.nunique())
p = pd.read_parquet(W / 'demand_pat_ksic_cc_year_le2016.parquet'); p['ksic'] = p.ksic.astype(str)
out = []
for cc_set, lab in [(['KR'], 'KR'), (['KR', 'US', 'JP', 'CN', 'EP'], '5국')]:
    pc = p[p.cc.isin(cc_set)].groupby(['ksic', 'year']).n.sum().unstack(fill_value=0).reindex(columns=range(1995, 2017), fill_value=0)
    lp = x.pivot(index='ksic', columns='year', values='lp'); lq = x.pivot(index='ksic', columns='year', values='lq')
    rr = []
    for t in range(2011, 2014):   # 2010 경제총조사 결측 -> t-2 에 2010 이 걸리지 않게 2011~(2009->2011 은 2010 미사용)
        for k in lp.index:
            if k not in pc.index or pd.isna(lp.loc[k].get(t)) or pd.isna(lp.loc[k].get(t - 2)): continue
            past = pc.loc[k, [t - 2, t - 1, t]].sum(); fut = pc.loc[k, [t + 1, t + 2, t + 3]].sum()
            if past < 10: continue
            rr.append(dict(ksic=k, t=t, dp=lp.loc[k, t] - lp.loc[k, t - 2], dq=lq.loc[k, t] - lq.loc[k, t - 2],
                           gpat=np.log(fut + 1) - np.log(past + 1)))
    r = pd.DataFrame(rr)
    for c in ['dp', 'dq', 'gpat']:
        r[c + '_w'] = r[c] - r.groupby('t')[c].transform('mean')
    r['type'] = np.select([(r.dp_w > 0) & (r.dq_w > 0), (r.dp_w <= 0) & (r.dq_w > 0), (r.dp_w > 0) & (r.dq_w <= 0)], ['수요형', '기술·공급형', '비용형'], '위축형')
    cl = {'groups': r.ksic.astype('category').cat.codes}
    f = smf.ols('gpat_w ~ dp_w + dq_w', r).fit(cov_type='cluster', cov_kwds=cl)
    print(lab, 'n', len(r), 'β_price', round(f.params['dp_w'], 3), 'p', round(f.pvalues['dp_w'], 4), 'β_qty', round(f.params['dq_w'], 3), 'p', round(f.pvalues['dq_w'], 4))
    print(r.groupby('type').gpat_w.agg(['count', 'mean']).round(3))
    out.append(dict(pat=lab, n=len(r), b_price=f.params['dp_w'], p_price=f.pvalues['dp_w'], b_qty=f.params['dq_w'], p_qty=f.pvalues['dq_w'],
                    **{f'mean_{a}': b for a, b in r.groupby('type').gpat_w.mean().items()}))
pd.DataFrame(out).to_csv(W / 'demand_test_decomp.csv', index=False, encoding='utf-8-sig')
