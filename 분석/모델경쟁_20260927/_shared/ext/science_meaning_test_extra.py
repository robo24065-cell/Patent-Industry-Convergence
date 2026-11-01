# -*- coding: utf-8 -*-
"""보조 신호 의미 검정 (science_, 2026-09-28): 기업 저자 논문·NRF 지원 논문 -> 특허 성장. 개발 원점 2006~2016."""
import os, numpy as np, pandas as pd
from science_meaning_test import *
ex = pd.read_parquet(os.path.join(EXT, 'science_openalex_ksic_extra_year.parquet'))
T = ex[ex.ksic == 'TOTAL'].set_index(['variant', 'year']).n_works
ex = ex[ex.ksic != 'TOTAL'].copy()
ex['share'] = ex.n_works / ex.set_index(['variant', 'year']).index.map(T).values
X = ex.pivot_table(index=['ksic', 'variant'], columns='year', values='share', fill_value=0)
Xn = ex.pivot_table(index=['ksic', 'variant'], columns='year', values='n_works', fill_value=0)
def g(M, t0): return lg(win(M, t0 - 2, t0), 1e-7) - lg(win(M, t0 - 5, t0 - 3), 1e-7)
def build_var(variant, offices, extra_ctl_country=None, min_n=10):
    def b(t0):
        y = outcome(P, t0); m = momentum(P, t0); xg = g(X.xs(variant, level='variant'), t0)
        nn = win(Xn.xs(variant, level='variant'), t0 - 5, t0)
        wk = g(W.xs(extra_ctl_country, level='country'), t0) if extra_ctl_country else None
        rows = []
        for k in xg.index:
            if nn[k] < min_n: continue
            for o in offices:
                r = {'ksic': k, 'y': y[(k, o)], 'x': xg[k], 'ctl_m': m[(k, o)]}
                if len(offices) > 1: r['grp'] = o
                if wk is not None: r['ctl_paper'] = wk[k]
                rows.append(r)
        return pd.DataFrame(rows)
    return b
out = [run_test('F_companyKR_paper->KR_patent', build_var('company_KR', ['KR'])),
       run_test('F2_companyKR_paper->KR_patent|KRpaper', build_var('company_KR', ['KR'], 'KR')),
       run_test('F3_companyALL_paper->5office_patent', build_var('company_ALL', ['KR','US','JP','CN','EP'])),
       run_test('F4_companyALL->5office|ALLpaper', build_var('company_ALL', ['KR','US','JP','CN','EP'], 'ALL')),
       run_test('G_NRF_paper->KR_patent', build_var('nrf_KR', ['KR'])),
       run_test('G2_NRF_paper->KR_patent|KRpaper', build_var('nrf_KR', ['KR'], 'KR'))]
R = pd.concat(out); R.to_csv(os.path.join(EXT, 'science_meaning_test_results_extra.csv'), index=False, encoding='utf-8-sig')
S = R.groupby('test').agg(n_origins=('t0','size'), mean_n=('n','mean'), mean_pcorr=('pcorr','mean'), share_pos=('pcorr', lambda s:(s>0).mean()), mean_null_sd=('null_sd','mean'), n_perm_p05=('perm_p', lambda s:(s<0.05).sum()))
print(S.round(3).to_string()); print(R.round(3).to_string())
