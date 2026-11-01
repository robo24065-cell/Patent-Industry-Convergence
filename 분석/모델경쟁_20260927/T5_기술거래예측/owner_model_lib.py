# 권리자×L 집계 테이블 (owner_model.py 와 동일 정의)
import numpy as np, pandas as pd
AGG_MEAN = ['age', 'lclaims', 'inv_is_app', 'prior_MAIN', 'cur_is_orig', 'paid_thru', 'prepaid', 'for_prio', 'pct', 'tech_exit', 'k_ipc_app3',
            'own_ipc_share', 'lic_any', 'n_inv', 'exam_yrs', 'ksic_rate3', 'multi', 'amt_per_yr', 'rgst_year']
OWN_LEVEL = ['own_port', 'own_lapse3', 'own_lapse1', 'own_lapse_rate', 'own_lapse_trend', 'own_corp3', 'own_in3', 'own_in1', 'own_app3', 'own_app_prev',
             'own_app_mom', 'own_years', 'own_n_ksic', 'k_app3', 'k_app_prev', 'k_mom', 'k_dormant', 'cur_type', 'cur_n']
SALE = ['own_out3', 'own_out3_days', 'own_out_all', 'own_out_rate', 'own_out1', 'own_out6m']


def owner_table(d, with_y=True, extra_first=(), extra_mean=()):
    g = d.groupby(['L', 'cur'])
    o = g[AGG_MEAN + list(extra_mean)].mean().add_prefix('m_')
    o = o.join(g[OWN_LEVEL + SALE + list(extra_first)].first())
    o['ksic_mode'] = g.ksic_c.agg(lambda s: s.value_counts().index[0])
    o['n'] = g.size()
    if with_y: o['rate'] = g.ytr.mean()
    return o.reset_index()


def feats(fset='A'):
    return ['m_' + c for c in AGG_MEAN] + OWN_LEVEL + ['ksic_mode'] + (SALE if fset == 'B' else [])


def ksic_global(years):
    """제공 5개국 데이터: KSIC별 국가별 출원 성장률 log((n[L-2]+1)/(n[L-5]+1)), KR·CN 점유율(L-2). L 시점엔 L-2 출원까지 대부분 공개."""
    k = pd.read_parquet('cache/ksic_cc_year.parquet'); k['ksic'] = k.ksic.astype(int)
    tot = k.groupby(['ksic', 'y']).n.sum().reset_index().assign(cc='ALL'); k = pd.concat([k, tot])
    p = k.pivot_table(index=['ksic', 'y'], columns='cc', values='n', aggfunc='sum').fillna(0)
    rows = []
    for Y in years:
        a = p.xs(Y - 2, level='y'); b = p.xs(Y - 5, level='y').reindex(a.index).fillna(0)
        r = np.log((a + 1) / (b + 1)).add_prefix('gg_')
        r['gs_KR'] = a['KR'] / a['ALL']; r['gs_CN'] = a['CN'] / a['ALL']
        rows.append(r.reset_index().rename(columns={'ksic': 'ksic_c'}).assign(L=Y))
    return pd.concat(rows, ignore_index=True)
GLOB = ['gg_ALL', 'gg_CN', 'gg_EP', 'gg_JP', 'gg_KR', 'gg_US', 'gs_KR', 'gs_CN']
