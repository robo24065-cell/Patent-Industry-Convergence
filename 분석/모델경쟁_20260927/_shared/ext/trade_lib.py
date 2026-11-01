# -*- coding: utf-8 -*-
"""trade_ 분야 의미 검정 공용 함수 (개발 구간 ≤2016만 사용)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
def patents(ccs=None):
    p = pd.read_parquet(EXT + r'\trade_patcnt_ksic_year_cc.parquet')
    if ccs: p = p[p.cc.isin(ccs)]
    return p.groupby(['ksic','year']).n.sum().unstack(fill_value=0)
def win(W, t, a, b):  # 연도 t+a..t+b 합
    cols = [c for c in range(t+a, t+b+1)]
    return W.reindex(columns=cols).sum(axis=1, min_count=len(cols))
def growth(W, t, a1, b1, a0, b0, log1p=True):
    f = (lambda s: np.log1p(s)) if log1p else (lambda s: np.log(s.where(s > 0)))
    return f(win(W, t, a1, b1)) - f(win(W, t, a0, b0))
def lead_test(X, P, origins, name, x_log1p=False, p_log1p=True, max_year=2016):
    """X: KSIC×연도 외부지표, P: KSIC×연도 특허. 원점 t에서
    전방: dX_fut(t+1..t+3 vs t-2..t) ~ dP(t-2..t vs t-5..t-3) + dX_past + 연도FE
    역방향: dP_fut ~ dX_past + dP_past + 연도FE."""
    rows = []
    for t in origins:
        assert t + 3 <= max_year
        r = pd.DataFrame({
            'dX_fut': growth(X, t, 1, 3, -2, 0, x_log1p), 'dX_past': growth(X, t, -2, 0, -5, -3, x_log1p),
            'dP_fut': growth(P, t, 1, 3, -2, 0, p_log1p), 'dP_past': growth(P, t, -2, 0, -5, -3, p_log1p)})
        r['t'] = t; rows.append(r.reset_index().rename(columns={'index': 'ksic'}))
    d = pd.concat(rows).replace([np.inf, -np.inf], np.nan).dropna()
    out = {'name': name, 'n_obs': len(d), 'n_ksic': d.ksic.nunique(), 'origins': f'{min(origins)}-{max(origins)}'}
    for tag, f, key in [('fwd', 'dX_fut ~ dP_past + dX_past + C(t)', 'dP_past'),
                        ('rev', 'dP_fut ~ dX_past + dP_past + C(t)', 'dX_past')]:
        m = smf.ols(f, d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.ksic)[0]})
        out[f'{tag}_beta'] = round(m.params[key], 4); out[f'{tag}_t'] = round(m.tvalues[key], 2); out[f'{tag}_p'] = round(m.pvalues[key], 4)
        # 원점별 편상관(잔차 스피어만)
        rs = []
        for t, g in d.groupby('t'):
            y = 'dX_fut' if tag == 'fwd' else 'dP_fut'; ctrl = 'dX_past' if tag == 'fwd' else 'dP_past'
            if len(g) < 10: continue
            ry = smf.ols(f'{y} ~ {ctrl}', g).fit().resid; rx = smf.ols(f'{key} ~ {ctrl}', g).fit().resid
            rs.append(spearmanr(rx, ry).statistic)
        out[f'{tag}_rho_mean'] = round(float(np.mean(rs)), 3); out[f'{tag}_rho_pos_share'] = round(float(np.mean(np.array(rs) > 0)), 2)
    return out, d
