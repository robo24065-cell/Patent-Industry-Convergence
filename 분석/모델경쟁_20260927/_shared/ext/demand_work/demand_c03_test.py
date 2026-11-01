# 의미 검정: KSIC×연도 물가 변화 -> 이후 특허 출원 성장 (개발 구간, 결과 ≤2016)
import pandas as pd, numpy as np, pathlib, statsmodels.formula.api as smf
from scipy.stats import spearmanr
OUT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext")
W = OUT / 'demand_work'
d = pd.read_parquet(OUT / 'demand_ecos_price_item_annual_1995_2021.parquet')
d = d[(d.cur_code.isna()) | (d.cur_code == 'W')]          # 수입·수출은 원화기준
m = pd.read_csv(OUT / 'demand_ksic_ecos_price_map.csv', dtype=str)
x = m.merge(d, on=['series', 'item_code']).dropna(subset=['value'])
x = x.sort_values(['ksic', 'series', 'item_code', 'year'])
x['dl'] = x.groupby(['ksic', 'series', 'item_code']).value.transform(lambda s: np.log(s).diff())
# 품목 등가 평균 로그변화 -> KSIC 연쇄지수
g = x.dropna(subset=['dl']).groupby(['ksic', 'series', 'year']).agg(dl=('dl', 'mean'), n_items=('item_code', 'nunique')).reset_index()
g['lidx'] = g.groupby(['ksic', 'series']).dl.cumsum()
g.to_parquet(OUT / 'demand_ksic_price_panel_1996_2021.parquet', index=False)
cov = g[(g.year >= 2005) & (g.year <= 2016)].groupby(['series', 'ksic']).year.nunique()
print('개발구간 2005-2016 12년 모두 관측된 KSIC 수:', (cov == 12).groupby('series').sum().to_dict())
print('2005-2016에 1년이라도 관측된 KSIC 수:', cov.groupby('series').size().to_dict())
anyk = g[(g.year >= 2005) & (g.year <= 2016)].ksic.nunique(); print('어느 계열이든 연결된 KSIC:', anyk, '/50')

p = pd.read_parquet(W / 'demand_pat_ksic_cc_year_le2016.parquet')
p['ksic'] = p.ksic.astype(str)
res = []
for cc_set, lab in [(['KR'], 'KR'), (['KR', 'US', 'JP', 'CN', 'EP'], '5국')]:
    pc = p[p.cc.isin(cc_set)].groupby(['ksic', 'year']).n.sum().unstack(fill_value=0).reindex(columns=range(1995, 2017), fill_value=0)
    for ser in ['ppi_basic', 'ppi_item', 'imp_item', 'exp_item']:
        li = g[g.series == ser].pivot(index='ksic', columns='year', values='lidx')
        rows = []
        for t in range(2001, 2014):   # 결과 t+1..t+3 <= 2016
            for k in li.index:
                if k not in pc.index: continue
                try:
                    dp = li.loc[k, t] - li.loc[k, t - 3]
                    dpf = li.loc[k, t + 3] - li.loc[k, t]
                except KeyError:
                    continue
                past = pc.loc[k, [t - 2, t - 1, t]].sum(); fut = pc.loc[k, [t + 1, t + 2, t + 3]].sum()
                if np.isnan(dp) or past < 10: continue
                rows.append(dict(ksic=k, t=t, dp=dp, dpf=dpf, gpat=np.log(fut + 1) - np.log(past + 1),
                                 gpat_past=np.log(past + 1) - np.log(pc.loc[k, [t - 5, t - 4, t - 3]].sum() + 1) if t - 5 >= 1995 else np.nan))
        r = pd.DataFrame(rows)
        if len(r) < 30: continue
        for c in ['dp', 'dpf', 'gpat', 'gpat_past']:
            r[c + '_w'] = r[c] - r.groupby('t')[c].transform('mean')   # 연도 고정효과 제거
        rho, pv = spearmanr(r.dp_w, r.gpat_w)
        f = smf.ols('gpat_w ~ dp_w', r).fit(cov_type='cluster', cov_kwds={'groups': r.ksic.astype('category').cat.codes})
        # 통제: 과거 특허 모멘텀
        rr = r.dropna(subset=['gpat_past_w'])
        f2 = smf.ols('gpat_w ~ dp_w + gpat_past_w', rr).fit(cov_type='cluster', cov_kwds={'groups': rr.ksic.astype('category').cat.codes})
        # 역방향(정식): 과거 특허성장(t-3..t) -> 미래 가격변화(t..t+3)
        rv = r.dropna(subset=['dpf_w', 'gpat_past_w'])
        rho_r, pv_r = spearmanr(rv.gpat_past_w, rv.dpf_w) if len(rv) > 30 else (np.nan, np.nan)
        # 이원 고정효과(KSIC+연도): 산업 내 시계열 변동만으로 선행 확인
        cl = {'groups': rr.ksic.astype('category').cat.codes}
        f3 = smf.ols('gpat ~ dp + gpat_past + C(ksic) + C(t)', rr).fit(cov_type='cluster', cov_kwds=cl)
        res.append(dict(pat=lab, series=ser, n_obs=len(r), n_ksic=r.ksic.nunique(), spearman_yrFE=round(rho, 3), p=round(pv, 4),
                        beta_yrFE=round(f.params['dp_w'], 3), p_cl=round(f.pvalues['dp_w'], 4),
                        beta_ctrl=round(f2.params['dp_w'], 3), p_ctrl=round(f2.pvalues['dp_w'], 4),
                        beta_2wayFE=round(f3.params['dp'], 3), p_2way=round(f3.pvalues['dp'], 4),
                        rev_rho_patpast_to_pricefut=round(rho_r, 3), rev_p=round(pv_r, 4)))
res = pd.DataFrame(res)
pd.set_option('display.width', 250); print(res.to_string())
res.to_csv(W / 'demand_test_price_lead.csv', index=False, encoding='utf-8-sig')
