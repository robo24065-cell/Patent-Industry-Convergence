# -*- coding: utf-8 -*-
"""
각도 C: 어떤 산업이 결과를 끌어내리는가 (영향점 진단)

기준 사양: dd_ship_late ~ dd_prod_early + over1,  over1 = (padi_e > 1.0)
평가: leave-one-cluster-out(mid) 예측 R², 분할연도 2013~2019 각각 계산 후 평균/최솟값

[1] leave-one-industry-out (47회)
[2] 회귀 진단 (split=2017): Cook's D, leverage, 표준화잔차
[3] 강건 추정: Huber RLM, LAD(q=0.5), 윈저화 OLS
[4] 신규 8개 산업 궤적표
[5] 판정용 요약

기존 스크립트 수정 없음. 새 파일.
"""
import os
import sys
import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.unicode.east_asian_width', True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel.csv')
PANEL39 = os.path.join(ROOT, '분석', 'fusion', 'model_search_panel_39ind.csv')
PARQ = os.path.join(ROOT, '공모전데이터', '작업데이터', 'kr_registered.parquet')
OUTDIR = os.path.join(ROOT, '분석', 'fusion')

NEW8 = [26111, 26121, 26211, 26212, 26519, 27199, 27301, 28111]
SPLITS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]


# ---------------------------------------------------------------- 평가 함수 (지정된 그대로)
def loco(X, y, g):
    """leave-one-cluster-out 예측 R². g는 클러스터 라벨(보통 mid)."""
    import numpy as np
    uq = np.unique(g); pr = np.full(len(y), np.nan)
    for c in uq:
        te = g == c; tr = ~te
        if tr.sum() < X.shape[1] + 3: return np.nan
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        try: b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
        except Exception: return np.nan
        pr[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    ok = ~np.isnan(pr)
    sse = ((y[ok]-pr[ok])**2).sum(); sst = ((y[ok]-y[ok].mean())**2).sum()
    return float(1 - sse/sst) if sst > 0 else np.nan


# ---------------------------------------------------------------- 데이터
def load_panel(path):
    d = pd.read_csv(path, encoding='utf-8-sig')
    d.columns = [c.strip().lstrip('﻿') for c in d.columns]
    d['over1'] = (d['padi_e'] > 1.0).astype(float)
    return d


def load_names():
    try:
        k = pd.read_parquet(PARQ, columns=['세세분류코드', '세세분류명'])
        k['세세분류코드'] = pd.to_numeric(k['세세분류코드'], errors='coerce')
        k = k.dropna(subset=['세세분류코드'])
        k['세세분류코드'] = k['세세분류코드'].astype(int)
        nm = k.drop_duplicates('세세분류코드').set_index('세세분류코드')['세세분류명']
        return nm.to_dict()
    except Exception as e:
        print('  [경고] 세세분류명 로드 실패:', e)
        return {}


FEATS = ['dd_prod_early', 'over1']
YCOL = 'dd_ship_late'


def eval_spec(d, feats=FEATS, ycol=YCOL, splits=SPLITS):
    """분할연도별 LOCO R² 와 OLS 계수. 반환: (r2 리스트, 계수 DataFrame, 실제 n(산업수))"""
    r2s, betas = [], []
    n_ind = None
    for s in splits:
        sub = d[d['split'] == s].dropna(subset=feats + [ycol, 'mid'])
        if len(sub) < 10:
            r2s.append(np.nan); continue
        n_ind = len(sub)
        X = sub[feats].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        g = sub['mid'].to_numpy()
        r2s.append(loco(X, y, g))
        Xt = np.column_stack([np.ones(len(y)), X])
        b = np.linalg.lstsq(Xt, y, rcond=None)[0]
        betas.append(dict(split=s, const=b[0], **{f: b[i + 1] for i, f in enumerate(feats)}))
    return np.array(r2s, float), pd.DataFrame(betas), n_ind


def summarize(r2s):
    return float(np.nanmean(r2s)), float(np.nanmin(r2s))


# ================================================================ [0] 기준선
def step0(d47, d39):
    print('=' * 100)
    print('[0] 기준선 확인 — 사양: dd_ship_late ~ dd_prod_early + over1  (over1 = padi_e>1)')
    print('=' * 100)
    rows = []
    for nm, dd in [('47산업', d47), ('39산업', d39)]:
        r2s, bt, n = eval_spec(dd)
        m, mn = summarize(r2s)
        rows.append(dict(표본=nm, n_산업=n, n_행=n * 7, LOCO평균=m, LOCO최소=mn,
                         beta_dd_prod_early=bt['dd_prod_early'].mean(),
                         beta_over1=bt['over1'].mean()))
        print(f'  {nm}: 분할별 LOCO R² = ' + ', '.join(f'{v:+.3f}' for v in r2s))
    base = pd.DataFrame(rows)
    print()
    print(base.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    return base


# ================================================================ [1] leave-one-industry-out
def step1(d47, names):
    print()
    print('=' * 100)
    print('[1] leave-one-industry-out — 47개 산업 각각 제거 후 재적합')
    print('=' * 100)
    base_r2s, base_bt, base_n = eval_spec(d47)
    base_mean, base_min = summarize(base_r2s)
    base_beta = base_bt['over1'].mean()
    base_beta_pers = base_bt['dd_prod_early'].mean()
    print(f'  기준(47산업 전체): LOCO평균={base_mean:+.4f}  LOCO최소={base_min:+.4f}  '
          f'β_over1={base_beta:+.4f}  β_persist={base_beta_pers:+.4f}   (n=47산업 × 7분할 = 329행)')

    recs = []
    for k in sorted(d47['ksic'].unique()):
        dd = d47[d47['ksic'] != k]
        r2s, bt, n = eval_spec(dd)
        m, mn = summarize(r2s)
        recs.append(dict(ksic=int(k), mid=int(d47.loc[d47.ksic == k, 'mid'].iloc[0]),
                         n_산업=n,
                         LOCO평균=m, dLOCO=m - base_mean,
                         LOCO최소=mn, dLOCO최소=mn - base_min,
                         beta_over1=bt['over1'].mean(), dbeta=bt['over1'].mean() - base_beta,
                         beta_persist=bt['dd_prod_early'].mean(),
                         padi_e=d47.loc[d47.ksic == k, 'padi_e'].mean(),
                         신규=int(k) in NEW8))
    L = pd.DataFrame(recs).sort_values('dLOCO', ascending=False).reset_index(drop=True)
    L['name'] = L['ksic'].map(lambda k: names.get(k, ''))

    top10 = L.head(10)
    n_new_in_top10 = int(top10['신규'].sum())
    print()
    print('  ▶ 제거 시 LOCO가 가장 크게 "오르는" 상위 10개 (= 결과를 끌어내리는 산업)')
    print('    (각 행은 그 산업을 뺀 n=46산업 × 7분할 = 322행 재적합 결과)')
    show = top10[['ksic', 'name', 'mid', 'padi_e', '신규', 'LOCO평균', 'dLOCO', 'beta_over1', 'dbeta']]
    print(show.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
    print(f'  ▶ 상위 10개 중 신규 8개 산업의 개수 = {n_new_in_top10}개  '
          f'(무작위 기댓값 = 10 × 8/47 = {10*8/47:.2f}개)')

    print()
    print('  ▶ 제거 시 LOCO가 가장 크게 "떨어지는" 하위 5개 (= 결과를 떠받치는 산업)')
    bot = L.tail(5).iloc[::-1][['ksic', 'name', 'mid', 'padi_e', '신규', 'LOCO평균', 'dLOCO', 'beta_over1']]
    print(bot.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

    print()
    print('  ▶ 신규 8개 산업의 leave-one-out 결과 (순위/47)')
    L['rank'] = np.arange(1, len(L) + 1)
    nw = L[L['신규']][['rank', 'ksic', 'name', 'padi_e', 'LOCO평균', 'dLOCO', 'beta_over1', 'dbeta']]
    print(nw.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
    print(f'  신규8 dLOCO 평균 = {L.loc[L["신규"],"dLOCO"].mean():+.4f} | '
          f'기존39 dLOCO 평균 = {L.loc[~L["신규"],"dLOCO"].mean():+.4f}')
    print(f'  신규8 순위 평균 = {L.loc[L["신규"],"rank"].mean():.1f} / 47 | '
          f'기존39 순위 평균 = {L.loc[~L["신규"],"rank"].mean():.1f} / 47  (1위=가장 해로움)')

    # 순위합 순열검정 (신규8의 순위합이 우연보다 낮은가)
    rs_new = L.loc[L['신규'], 'rank'].sum()
    rng = np.random.default_rng(20260813)
    perm = np.array([rng.choice(np.arange(1, 48), 8, replace=False).sum() for _ in range(20000)])
    p_perm = float((perm <= rs_new).mean())
    print(f'  신규8 순위합={rs_new} | 순열검정 p(순위합 ≤ 관측) = {p_perm:.4f}  '
          f'[n=8 부분표본 — 방향 참고용, 유의성 주장 아님]')

    # 47개 순차 제거: 상위 8개(가장 해로운) 동시 제거 효과
    worst8 = L.head(8)['ksic'].tolist()
    dd = d47[~d47['ksic'].isin(worst8)]
    r2s, bt, n = eval_spec(dd)
    m8, mn8 = summarize(r2s)
    ddn = d47[~d47['ksic'].isin(NEW8)]
    r2sn, btn, nn = eval_spec(ddn)
    mn_, mnn = summarize(r2sn)
    print()
    print('  ▶ 8개 동시 제거 비교 (모두 n=39산업 × 7분할 = 273행)')
    cmp = pd.DataFrame([
        dict(제거집합='없음(47산업)', n_산업=47, LOCO평균=base_mean, LOCO최소=base_min, beta_over1=base_beta),
        dict(제거집합='LOO 최악8', n_산업=n, LOCO평균=m8, LOCO최소=mn8, beta_over1=bt['over1'].mean()),
        dict(제거집합='신규8', n_산업=nn, LOCO평균=mn_, LOCO최소=mnn, beta_over1=btn['over1'].mean()),
    ])
    print(cmp.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print(f'  LOO 최악8 = {sorted(worst8)}')
    print(f'  신규8     = {sorted(NEW8)}')
    print(f'  교집합    = {sorted(set(worst8) & set(NEW8))} ({len(set(worst8)&set(NEW8))}개)')

    L.to_csv(os.path.join(OUTDIR, 'diag_influence_loo.csv'), index=False, encoding='utf-8-sig')
    return L, base_mean, base_beta, n_new_in_top10, cmp


# ================================================================ [2] 회귀 진단
def step2(d47, names, split=2017):
    print()
    print('=' * 100)
    print(f'[2] 회귀 진단 (split={split}, n=47산업)')
    print('=' * 100)
    import statsmodels.api as sm
    sub = d47[d47['split'] == split].dropna(subset=FEATS + [YCOL]).copy()
    X = sm.add_constant(sub[FEATS].to_numpy(float))
    y = sub[YCOL].to_numpy(float)
    res = sm.OLS(y, X).fit()
    infl = res.get_influence()
    sub['fitted'] = res.fittedvalues
    sub['resid'] = res.resid
    sub['std_resid'] = infl.resid_studentized_internal
    sub['leverage'] = infl.hat_matrix_diag
    sub['cooksD'] = infl.cooks_distance[0]
    sub['dffits'] = infl.dffits[0]
    sub['신규'] = sub['ksic'].isin(NEW8)
    sub['name'] = sub['ksic'].map(lambda k: names.get(int(k), ''))

    n, p = len(sub), X.shape[1]
    thr_cook = 4.0 / n
    thr_lev = 2.0 * p / n
    print(f'  n={n}, p={p}(상수포함).  임계: Cook D > 4/n = {thr_cook:.4f}, '
          f'leverage > 2p/n = {thr_lev:.4f}, |표준화잔차| > 2')
    print(f'  OLS: β_persist={res.params[1]:+.4f} (t={res.tvalues[1]:+.2f}), '
          f'β_over1={res.params[2]:+.4f} (t={res.tvalues[2]:+.2f}), R²={res.rsquared:.4f}')

    flag = sub[(sub.cooksD > thr_cook) | (sub.leverage > thr_lev) | (sub.std_resid.abs() > 2)]
    cols = ['ksic', 'name', 'mid', 'padi_e', 'over1', '신규', YCOL, 'fitted', 'resid',
            'std_resid', 'leverage', 'cooksD']
    print()
    print(f'  ▶ 임계 초과 산업 {len(flag)}개 (Cook D 내림차순)')
    print(flag.sort_values('cooksD', ascending=False)[cols].to_string(
        index=False, float_format=lambda v: f'{v:+.4f}'))

    print()
    print('  ▶ 신규 8개의 진단 위치 (전 47개 중 순위, 1위 = 가장 영향 큼)')
    sub['cook_rank'] = sub['cooksD'].rank(ascending=False).astype(int)
    sub['lev_rank'] = sub['leverage'].rank(ascending=False).astype(int)
    nw = sub[sub['신규']].sort_values('cooksD', ascending=False)
    print(nw[['ksic', 'name', 'padi_e', 'over1', YCOL, 'fitted', 'resid', 'std_resid',
              'leverage', 'cooksD', 'cook_rank', 'lev_rank']].to_string(
        index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
    print(f'  Cook D 평균: 신규8={nw["cooksD"].mean():.4f} vs 기존39={sub.loc[~sub["신규"],"cooksD"].mean():.4f}'
          f'  (배율 {nw["cooksD"].mean()/max(sub.loc[~sub["신규"],"cooksD"].mean(),1e-12):.2f}x)')
    print(f'  |표준화잔차| 평균: 신규8={nw["std_resid"].abs().mean():.3f} vs '
          f'기존39={sub.loc[~sub["신규"],"std_resid"].abs().mean():.3f}')
    print(f'  leverage 평균: 신규8={nw["leverage"].mean():.4f} vs '
          f'기존39={sub.loc[~sub["신규"],"leverage"].mean():.4f}')
    print(f'  Cook D 총합 중 신규8 비중 = {nw["cooksD"].sum()/sub["cooksD"].sum()*100:.1f}%  '
          f'(균등이면 {8/47*100:.1f}%)')

    # 전 분할연도 요약
    print()
    print('  ▶ 참고: 7개 분할연도 전체에서 Cook D 상위 5위 안에 신규8이 든 횟수')
    cnt = []
    for s in SPLITS:
        ss = d47[d47['split'] == s].dropna(subset=FEATS + [YCOL]).copy()
        r = sm.OLS(ss[YCOL].to_numpy(float), sm.add_constant(ss[FEATS].to_numpy(float))).fit()
        cd = r.get_influence().cooks_distance[0]
        ss['cooksD'] = cd
        t5 = ss.nlargest(5, 'cooksD')['ksic'].tolist()
        cnt.append(dict(split=s, 상위5=t5, 신규수=sum(k in NEW8 for k in t5)))
    print(pd.DataFrame(cnt).to_string(index=False))
    sub.to_csv(os.path.join(OUTDIR, f'diag_influence_reg{split}.csv'), index=False, encoding='utf-8-sig')
    return sub


# ================================================================ [3] 강건 추정
def step3(d47):
    print()
    print('=' * 100)
    print('[3] 강건 추정 — β_over1 비교 (모두 n=47산업, 분할연도별 적합 후 평균)')
    print('=' * 100)
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg

    rows = []
    for s in SPLITS:
        sub = d47[d47['split'] == s].dropna(subset=FEATS + [YCOL]).copy()
        X = sm.add_constant(sub[FEATS].to_numpy(float))
        y = sub[YCOL].to_numpy(float)
        r = dict(split=s, n=len(sub))
        o = sm.OLS(y, X).fit()
        r['OLS'] = o.params[2]; r['OLS_t'] = o.tvalues[2]; r['OLS_persist'] = o.params[1]
        try:
            h = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
            r['Huber'] = h.params[2]; r['Huber_t'] = h.tvalues[2]; r['Huber_persist'] = h.params[1]
        except Exception:
            r['Huber'] = np.nan; r['Huber_t'] = np.nan; r['Huber_persist'] = np.nan
        try:
            q = QuantReg(y, X).fit(q=0.5)
            r['LAD'] = q.params[2]; r['LAD_t'] = q.tvalues[2]; r['LAD_persist'] = q.params[1]
        except Exception:
            r['LAD'] = np.nan; r['LAD_t'] = np.nan; r['LAD_persist'] = np.nan
        lo, hi = np.percentile(y, [5, 95])
        yw = np.clip(y, lo, hi)
        w = sm.OLS(yw, X).fit()
        r['Winsor'] = w.params[2]; r['Winsor_t'] = w.tvalues[2]; r['Winsor_persist'] = w.params[1]
        rows.append(r)
    R = pd.DataFrame(rows)
    print('  ▶ 분할연도별 β_over1')
    print(R[['split', 'n', 'OLS', 'Huber', 'LAD', 'Winsor']].to_string(
        index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
    summ = pd.DataFrame([
        dict(추정량='OLS', beta_over1=R['OLS'].mean(), t평균=R['OLS_t'].mean(),
             beta_persist=R['OLS_persist'].mean(), 대OLS배율=1.0),
        dict(추정량='Huber M', beta_over1=R['Huber'].mean(), t평균=R['Huber_t'].mean(),
             beta_persist=R['Huber_persist'].mean(), 대OLS배율=R['Huber'].mean() / R['OLS'].mean()),
        dict(추정량='LAD(q=.5)', beta_over1=R['LAD'].mean(), t평균=R['LAD_t'].mean(),
             beta_persist=R['LAD_persist'].mean(), 대OLS배율=R['LAD'].mean() / R['OLS'].mean()),
        dict(추정량='윈저 5/95', beta_over1=R['Winsor'].mean(), t평균=R['Winsor_t'].mean(),
             beta_persist=R['Winsor_persist'].mean(), 대OLS배율=R['Winsor'].mean() / R['OLS'].mean()),
    ])
    print('  ▶ 7개 분할 평균 (n=47산업 고정)')
    print(summ.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print()
    print('  해석: 배율 > 1 이면 이상치가 OLS 계수를 희석 중. 배율 ≈ 1 이면 이상치 문제 아님.')

    # 39산업 대조
    d39 = load_panel(PANEL39)
    rows39 = []
    for s in SPLITS:
        sub = d39[d39['split'] == s].dropna(subset=FEATS + [YCOL]).copy()
        X = sm.add_constant(sub[FEATS].to_numpy(float)); y = sub[YCOL].to_numpy(float)
        o = sm.OLS(y, X).fit()
        h = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        rows39.append(dict(split=s, n=len(sub), OLS=o.params[2], Huber=h.params[2]))
    R39 = pd.DataFrame(rows39)
    print(f'  ▶ 대조: 39산업 패널 (n=39) β_over1  OLS={R39["OLS"].mean():+.4f}, '
          f'Huber={R39["Huber"].mean():+.4f}')
    return summ, R


# ================================================================ [4] 신규 8개 궤적
def step4(d47, names, reg2017):
    print()
    print('=' * 100)
    print('[4] 신규 8개 산업의 실제 궤적 — 전반기/후반기 낙폭, PADI, 예측, 잔차')
    print('=' * 100)
    import statsmodels.api as sm

    # 분할연도별 예측/잔차 -> 산업별 평균
    parts = []
    for s in SPLITS:
        sub = d47[d47['split'] == s].dropna(subset=FEATS + [YCOL]).copy()
        X = sm.add_constant(sub[FEATS].to_numpy(float)); y = sub[YCOL].to_numpy(float)
        o = sm.OLS(y, X).fit()
        sub['pred_OLS'] = o.fittedvalues
        sub['resid_OLS'] = o.resid
        # LOCO(mid) 표본외 예측
        g = sub['mid'].to_numpy(); pr = np.full(len(sub), np.nan)
        for c in np.unique(g):
            te = g == c; tr = ~te
            if tr.sum() < 5: continue
            b = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), sub[FEATS].to_numpy(float)[tr]]),
                                y[tr], rcond=None)[0]
            pr[te] = np.column_stack([np.ones(te.sum()), sub[FEATS].to_numpy(float)[te]]) @ b
        sub['pred_LOCO'] = pr
        sub['resid_LOCO'] = y - pr
        parts.append(sub)
    P = pd.concat(parts)

    agg = P.groupby('ksic').agg(
        mid=('mid', 'first'), padi_e=('padi_e', 'mean'),
        전반기낙폭=('dd_prod_early', 'mean'), 후반기낙폭=('dd_ship_late', 'mean'),
        예측_OLS=('pred_OLS', 'mean'), 잔차_OLS=('resid_OLS', 'mean'),
        예측_LOCO=('pred_LOCO', 'mean'), 잔차_LOCO=('resid_LOCO', 'mean'),
        절대잔차_LOCO=('resid_LOCO', lambda v: np.nanmean(np.abs(v))),
    ).reset_index()
    agg['over1'] = (agg['padi_e'] > 1.0).astype(int)
    agg['name'] = agg['ksic'].map(lambda k: names.get(int(k), ''))
    agg['신규'] = agg['ksic'].isin(NEW8)
    agg['잔차순위'] = agg['절대잔차_LOCO'].rank(ascending=False).astype(int)

    nw = agg[agg['신규']].sort_values('잔차_LOCO')
    cols = ['ksic', 'name', 'padi_e', 'over1', '전반기낙폭', '후반기낙폭',
            '예측_LOCO', '잔차_LOCO', '예측_OLS', '잔차_OLS', '잔차순위']
    print('  ▶ 신규 8개 (7개 분할연도 평균값. 낙폭 단위: %, 음수가 클수록 깊은 낙폭)')
    print(nw[cols].to_string(index=False, float_format=lambda v: f'{v:+.2f}'))
    print()
    print(f'  전체 47산업 |LOCO잔차| 평균 = {agg["절대잔차_LOCO"].mean():.2f}')
    print(f'  신규8       |LOCO잔차| 평균 = {agg.loc[agg["신규"],"절대잔차_LOCO"].mean():.2f}')
    print(f'  기존39      |LOCO잔차| 평균 = {agg.loc[~agg["신규"],"절대잔차_LOCO"].mean():.2f}')
    print()
    print('  ▶ 전체 47산업 |LOCO잔차| 상위 10 (가장 빗나간 산업)')
    top = agg.nlargest(10, '절대잔차_LOCO')[['ksic', 'name', 'padi_e', '전반기낙폭', '후반기낙폭',
                                           '예측_LOCO', '잔차_LOCO', '신규']]
    print(top.to_string(index=False, float_format=lambda v: f'{v:+.2f}'))
    print(f'  → 상위 10 중 신규8 = {int(top["신규"].sum())}개 (기댓값 {10*8/47:.2f}개)')

    # 지속성 자체의 관계가 신규8에서 다른가
    print()
    print('  ▶ 지속성(전반기낙폭)–후반기낙폭 상관: 부분표본별 [n=8 유의성 주장 아님]')
    for nm, mask in [('전체47', np.ones(len(agg), bool)), ('기존39', ~agg['신규'].to_numpy()),
                     ('신규8', agg['신규'].to_numpy())]:
        a = agg[mask]
        r = np.corrcoef(a['전반기낙폭'], a['후반기낙폭'])[0, 1]
        r2 = np.corrcoef(a['padi_e'], a['후반기낙폭'])[0, 1]
        print(f'    {nm:6s} (n={len(a):2d}): corr(전반기낙폭, 후반기낙폭) = {r:+.3f} | '
              f'corr(PADI, 후반기낙폭) = {r2:+.3f}')

    agg.to_csv(os.path.join(OUTDIR, 'diag_influence_traj.csv'), index=False, encoding='utf-8-sig')
    return agg


# ================================================================ [5] 판정
def step5(L, agg, base_mean, n_new_top10, cmp, summ):
    print()
    print('=' * 100)
    print('[5] 판정 — 소수 영향점인가 전반적 약화인가')
    print('=' * 100)
    # dLOCO 분포 집중도
    pos = L[L['dLOCO'] > 0]
    tot_pos = pos['dLOCO'].sum()
    top3 = L.nlargest(3, 'dLOCO')['dLOCO'].sum()
    print(f'  · dLOCO>0 (제거하면 개선) 산업 수 = {len(pos)}/47')
    print(f'  · 개선분 총합 대비 상위 3개 비중 = {top3/tot_pos*100:.1f}%')
    print(f'  · 최대 단일 개선폭 = {L["dLOCO"].max():+.4f} (기준 LOCO {base_mean:+.4f} 대비 '
          f'{L["dLOCO"].max()/abs(base_mean)*100:.1f}%)')
    print(f'  · 상위 10 중 신규8 = {n_new_top10}개 (기댓값 1.70개)')
    print(f'  · 신규8 전체 제거 시 LOCO = '
          f'{cmp.loc[cmp["제거집합"]=="신규8","LOCO평균"].iloc[0]:+.4f} '
          f'(47산업 {base_mean:+.4f}, 39산업 원본과 비교할 것)')
    print(f'  · 강건추정 β_over1 / OLS β_over1 배율: Huber '
          f'{summ.loc[summ["추정량"]=="Huber M","대OLS배율"].iloc[0]:.2f}x, '
          f'LAD {summ.loc[summ["추정량"]=="LAD(q=.5)","대OLS배율"].iloc[0]:.2f}x, '
          f'윈저 {summ.loc[summ["추정량"]=="윈저 5/95","대OLS배율"].iloc[0]:.2f}x')


# ================================================================ [6] 종속변수 셀 구조 (중대 발견)
def step6(d47, d39, names):
    print()
    print('=' * 100)
    print('[6] 종속변수의 해상도 점검 — 47개 KSIC가 실제로 몇 개의 서로 다른 결과를 갖는가')
    print('=' * 100)
    d47 = d47.copy()
    d47['cell'] = (d47['dd_prod_early'].round(6).astype(str) + '|' +
                   d47['dd_ship_late'].round(6).astype(str))
    for s in [2013, 2017, 2019]:
        ss = d47[d47['split'] == s]
        s39 = ss[~ss['ksic'].isin(NEW8)]
        print(f'  split={s}: 47산업 → 고유 (전반기낙폭, 후반기낙폭) 조합 {ss["cell"].nunique()}개 | '
              f'39산업 → {s39["cell"].nunique()}개  (신규8이 늘린 고유 결과 셀 = '
              f'{ss["cell"].nunique() - s39["cell"].nunique()}개)')

    s = d47[d47['split'] == 2017].copy()
    s39 = set(s.loc[~s['ksic'].isin(NEW8), 'cell'])
    print()
    print('  ▶ 신규 8개가 각각 어느 결과 셀에 들어갔는가 (split=2017)')
    rows = []
    for k in NEW8:
        r = s[s['ksic'] == k].iloc[0]
        mates = s[(s['cell'] == r['cell']) & (s['ksic'] != k)]
        rows.append(dict(ksic=k, name=names.get(k, ''),
                         후반기낙폭=r['dd_ship_late'], over1=int(r['over1']),
                         셀이_기존에_있었나=r['cell'] in s39,
                         동일셀_기존산업=','.join(str(int(x)) for x in mates['ksic']),
                         동일셀_over1=','.join(str(int(x)) for x in mates['over1'])))
    C = pd.DataFrame(rows)
    print(C.to_string(index=False, float_format=lambda v: f'{v:+.2f}'))
    n_newcell = int((~C['셀이_기존에_있었나']).sum())
    print()
    print(f'  → 신규 8개 중 완전히 새로운 결과 셀을 만든 산업 = {n_newcell}개, '
          f'기존 셀에 합류(=결과값 중복) = {8 - n_newcell}개')

    # 셀 내부 over1 분산 = 종속변수 변동 없이 설명변수만 흔드는 순수 감쇠 요인
    print()
    print('  ▶ 셀 내부 over1 불일치(같은 결과인데 PADI 깃발이 다른 쌍) 수')
    for nm, sub in [('39산업', s[~s['ksic'].isin(NEW8)]), ('47산업', s)]:
        g = sub.groupby('cell')['over1']
        mixed = int((g.nunique() > 1).sum())
        v = float(g.transform(lambda x: x - x.mean()).pow(2).sum())
        print(f'    {nm} (n={len(sub)}): 혼재 셀 {mixed}개 / 전체 {sub["cell"].nunique()}개, '
              f'셀내 over1 제곱합 = {v:.3f}  (이 부분은 y 변동이 0이므로 계수를 0쪽으로 끈다)')

    # 셀 수준으로 축약해서 재평가 (같은 결과를 1행으로)
    print()
    print('  ▶ 결과 셀 수준으로 축약 후 재평가 (중복 가중 제거). over1은 셀 평균 사용')
    out = []
    for nm, keep in [('39산업 축약', ~d47['ksic'].isin(NEW8)), ('47산업 축약', pd.Series(True, index=d47.index))]:
        dd = d47[keep]
        r2s = []
        betas = []
        for sp in SPLITS:
            ss = dd[dd['split'] == sp]
            cg = ss.groupby('cell').agg(y=('dd_ship_late', 'mean'), x=('dd_prod_early', 'mean'),
                                        o=('over1', 'mean'), mid=('mid', lambda v: v.mode().iloc[0]))
            X = cg[['x', 'o']].to_numpy(float); y = cg['y'].to_numpy(float)
            r2s.append(loco(X, y, cg['mid'].to_numpy()))
            b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
            betas.append(b[2])
        out.append(dict(표본=nm, n_셀_split2017=int(dd[dd.split == 2017]['cell'].nunique()),
                        LOCO평균=float(np.nanmean(r2s)), LOCO최소=float(np.nanmin(r2s)),
                        beta_over1=float(np.mean(betas))))
    Cc = pd.DataFrame(out)
    print(Cc.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('  (주: 두 행은 셀 수가 달라 n이 다르다 — 각 행의 n을 그대로 읽을 것)')
    return C, Cc, n_newcell


# ================================================================ [7] 규모 (H-A 점검용)
def load_size():
    import glob, re
    fs = sorted(glob.glob(os.path.join(ROOT, '공모전데이터', '외부데이터', '산업분류별', '*.csv')))
    parts = []
    for p in fs:
        try:
            d = pd.read_csv(p, encoding='cp949')
        except Exception:
            continue
        if '[B]산업별' not in d.columns: continue
        d['code'] = d['[B]산업별'].astype(str)
        m = d['code'].str.fullmatch(r'[A-Z]\d{5}')
        d = d[m].copy()
        d['ksic'] = d['code'].str[1:].astype(int)
        ycols = [c for c in d.columns if re.match(r'^\d{4}\s*년', str(c))]
        for c in ycols:
            t = d[['ksic', '항목', c]].copy(); t.columns = ['ksic', 'item', 'val']
            t['yr'] = int(str(c)[:4])
            parts.append(t)
    if not parts: return None
    R = pd.concat(parts)
    R['val'] = pd.to_numeric(R['val'], errors='coerce')
    return R


def step7(d47, names):
    print()
    print('=' * 100)
    print('[7] 규모 점검 — 신규 8개는 정말 대형인가 / 규모가 관계를 조절하는가 (H-A 보조검정)')
    print('=' * 100)
    R = load_size()
    if R is None:
        print('  [경고] 규모 데이터 로드 실패 — 생략'); return None
    items = [i for i in R['item'].dropna().unique() if '출하액' in str(i)]
    it = items[0] if items else None
    if it is None:
        print('  [경고] 출하액 항목 없음:', R['item'].dropna().unique()[:10]); return None
    sz = (R[(R['item'] == it) & (R['yr'].between(2010, 2015))]
          .groupby('ksic')['val'].mean().rename('출하액').reset_index())
    ind = d47[d47['split'] == 2017][['ksic', 'mid', 'padi_e', 'over1',
                                     'dd_prod_early', 'dd_ship_late']].copy()
    ind = ind.merge(sz, on='ksic', how='left')
    ind['신규'] = ind['ksic'].isin(NEW8)
    ind['name'] = ind['ksic'].map(lambda k: names.get(int(k), ''))
    got = ind['출하액'].notna().sum()
    print(f'  규모 지표 = {it} (2010~2015 평균).  매칭 성공 {got}/47산업')
    if got < 30:
        print('  매칭 부족 — 생략'); return None
    ind['log규모'] = np.log(ind['출하액'].clip(lower=1))
    ind['규모순위'] = ind['출하액'].rank(ascending=False)
    print()
    print(f'  중앙값 출하액(백만원): 신규8={ind.loc[ind["신규"],"출하액"].median():,.0f} | '
          f'기존39={ind.loc[~ind["신규"],"출하액"].median():,.0f} '
          f'(배율 {ind.loc[ind["신규"],"출하액"].median()/max(ind.loc[~ind["신규"],"출하액"].median(),1):.1f}x)')
    print(f'  규모순위 평균(1=최대): 신규8={ind.loc[ind["신규"],"규모순위"].mean():.1f}/47 | '
          f'기존39={ind.loc[~ind["신규"],"규모순위"].mean():.1f}/47')
    print()
    print('  ▶ 신규 8개 규모 위치')
    print(ind[ind['신규']].sort_values('규모순위')[['ksic', 'name', '출하액', '규모순위', 'padi_e']]
          .to_string(index=False, float_format=lambda v: f'{v:,.1f}'))

    # 규모 반분 LOCO (표본이 달라지므로 n 명기)
    print()
    print('  ▶ 규모 반분 LOCO (기준 사양). 각 행의 n을 반드시 함께 읽을 것')
    med = ind['출하액'].median()
    big = set(ind.loc[ind['출하액'] >= med, 'ksic'])
    small = set(ind.loc[ind['출하액'] < med, 'ksic'])
    rows = []
    for nm, ks in [('전체47', set(ind['ksic'])), ('대형 상위50%', big), ('소형 하위50%', small),
                   ('기존39 중 대형', big & set(ind.loc[~ind['신규'], 'ksic'])),
                   ('기존39 중 소형', small & set(ind.loc[~ind['신규'], 'ksic']))]:
        dd = d47[d47['ksic'].isin(ks)]
        r2s, bt, n = eval_spec(dd)
        rows.append(dict(부분표본=nm, n_산업=n, LOCO평균=float(np.nanmean(r2s)),
                         LOCO최소=float(np.nanmin(r2s)),
                         beta_over1=bt['over1'].mean() if len(bt) else np.nan,
                         유효mid수=dd[dd.split == 2017]['mid'].nunique()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

    # 규모 조절항
    print()
    print('  ▶ 규모 조절항 추가 (n=47 고정): dd_ship_late ~ 지속성 + over1 + log규모 + over1×log규모')
    d2 = d47.merge(ind[['ksic', 'log규모']], on='ksic', how='left')
    d2['lz'] = (d2['log규모'] - d2['log규모'].mean()) / d2['log규모'].std()
    d2['o_x_lz'] = d2['over1'] * d2['lz']
    for feats in [['dd_prod_early', 'over1'], ['dd_prod_early', 'over1', 'lz'],
                  ['dd_prod_early', 'over1', 'lz', 'o_x_lz']]:
        r2s, bt, n = eval_spec(d2, feats=feats)
        print(f'    {str(feats):55s} n={n}산업 LOCO평균={np.nanmean(r2s):+.4f} '
              f'최소={np.nanmin(r2s):+.4f} β_over1={bt["over1"].mean():+.3f}'
              + (f' β_교호={bt["o_x_lz"].mean():+.3f}' if 'o_x_lz' in feats else ''))
    return ind


# ================================================================ [8] 규모 vs 전자 — H-A/H-B 분리
def step8(d47, ind, names):
    print()
    print('=' * 100)
    print('[8] H-A(규모층) vs H-B(전자업 이질성) 분리 — 두 축은 교락되어 있으므로 교차 분해')
    print('=' * 100)
    if ind is None:
        print('  규모 데이터 없음 — 생략'); return
    d = d47.merge(ind[['ksic', '출하액']], on='ksic', how='left')
    d['대형'] = d['출하액'] >= ind['출하액'].median()
    d['전자'] = d['mid'].isin([26])          # 26 = 전자부품·컴퓨터·영상·통신
    d['전자기계'] = d['mid'].isin([26, 27, 28])

    print('  ▶ 교락 확인 (split=2017, 47산업)')
    s = d[d.split == 2017]
    ct = pd.crosstab(s['전자'], s['대형'])
    ct.index = ['비전자', '전자(mid26)']; ct.columns = ['소형', '대형']
    print(ct.to_string())
    print(f'  corr(전자더미, log출하액) = '
          f'{np.corrcoef(s["전자"].astype(float), np.log(s["출하액"]))[0,1]:+.3f}')

    print()
    print('  ▶ 4분면 및 축별 부분표본 LOCO (기준 사양). n과 유효 mid 클러스터 수 반드시 확인')
    rows = []
    defs = [
        ('전체47', s['ksic']),
        ('전자 mid26', s.loc[s['전자'], 'ksic']),
        ('비전자', s.loc[~s['전자'], 'ksic']),
        ('전자기계 mid26/27/28', s.loc[s['전자기계'], 'ksic']),
        ('그 외', s.loc[~s['전자기계'], 'ksic']),
        ('대형', s.loc[s['대형'], 'ksic']),
        ('소형', s.loc[~s['대형'], 'ksic']),
        ('비전자 × 대형', s.loc[~s['전자'] & s['대형'], 'ksic']),
        ('비전자 × 소형', s.loc[~s['전자'] & ~s['대형'], 'ksic']),
        ('전자 × 대형', s.loc[s['전자'] & s['대형'], 'ksic']),
        ('전자 × 소형', s.loc[s['전자'] & ~s['대형'], 'ksic']),
    ]
    for nm, ks in defs:
        ks = set(ks)
        dd = d47[d47['ksic'].isin(ks)]
        nmid = dd[dd.split == 2017]['mid'].nunique()
        r2s, bt, n = eval_spec(dd)
        rows.append(dict(부분표본=nm, n_산업=len(ks), 유효mid=nmid,
                         LOCO평균=float(np.nanmean(r2s)) if np.isfinite(r2s).any() else np.nan,
                         LOCO최소=float(np.nanmin(r2s)) if np.isfinite(r2s).any() else np.nan,
                         beta_over1=bt['over1'].mean() if len(bt) else np.nan,
                         beta_persist=bt['dd_prod_early'].mean() if len(bt) else np.nan))
    T = pd.DataFrame(rows)
    print(T.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('  (LOCO는 mid 클러스터를 하나씩 빼므로 유효mid가 3 이하면 수치를 신뢰하지 말 것)')

    print()
    print('  ▶ 신규8 제외한 39산업 안에서만 같은 분해 (신규8과 무관하게 층 차이가 있는가)')
    s39 = s[~s['ksic'].isin(NEW8)]
    rows = []
    for nm, ks in [('기존39 전체', s39['ksic']),
                   ('기존39 전자', s39.loc[s39['전자'], 'ksic']),
                   ('기존39 비전자', s39.loc[~s39['전자'], 'ksic']),
                   ('기존39 대형', s39.loc[s39['대형'], 'ksic']),
                   ('기존39 소형', s39.loc[~s39['대형'], 'ksic']),
                   ('기존39 비전자×대형', s39.loc[~s39['전자'] & s39['대형'], 'ksic']),
                   ('기존39 비전자×소형', s39.loc[~s39['전자'] & ~s39['대형'], 'ksic'])]:
        ks = set(ks)
        dd = d47[d47['ksic'].isin(ks)]
        nmid = dd[dd.split == 2017]['mid'].nunique()
        r2s, bt, n = eval_spec(dd)
        rows.append(dict(부분표본=nm, n_산업=len(ks), 유효mid=nmid,
                         LOCO평균=float(np.nanmean(r2s)) if np.isfinite(r2s).any() else np.nan,
                         beta_over1=bt['over1'].mean() if len(bt) else np.nan))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

    print()
    print('  ▶ 조절항 직접 비교 (n=47 고정, 동일 표본): 규모 조절 vs 전자 조절')
    d['lz'] = (np.log(d['출하액']) - np.log(d['출하액']).mean()) / np.log(d['출하액']).std()
    d['el'] = d['전자'].astype(float)
    d['o_lz'] = d['over1'] * d['lz']; d['o_el'] = d['over1'] * d['el']
    for nm, feats in [('기준', ['dd_prod_early', 'over1']),
                      ('+규모교호', ['dd_prod_early', 'over1', 'lz', 'o_lz']),
                      ('+전자교호', ['dd_prod_early', 'over1', 'el', 'o_el']),
                      ('+둘 다', ['dd_prod_early', 'over1', 'lz', 'o_lz', 'el', 'o_el'])]:
        r2s, bt, n = eval_spec(d, feats=feats)
        extra = ' '.join(f'β_{f}={bt[f].mean():+.2f}' for f in feats if f in ('o_lz', 'o_el'))
        print(f'    {nm:10s} n={n}산업 LOCO평균={np.nanmean(r2s):+.4f} 최소={np.nanmin(r2s):+.4f} '
              f'β_over1={bt["over1"].mean():+.2f} {extra}')
    return T


# ================================================================ [9] 유효 표본 크기 정직 보고
def step9(d47, ind):
    print()
    print('=' * 100)
    print('[9] 유효 표본 크기 — 위 부분표본 수치를 읽을 때 반드시 함께 볼 것')
    print('=' * 100)
    if ind is None:
        print('  규모 데이터 없음 — 생략'); return
    s = d47[d47['split'] == 2017].merge(ind[['ksic', '출하액']], on='ksic', how='left')
    s['cell'] = s['dd_prod_early'].round(6).astype(str) + '|' + s['dd_ship_late'].round(6).astype(str)
    s['대형'] = s['출하액'] >= s['출하액'].median()
    s['전자'] = s['mid'] == 26
    s['신규'] = s['ksic'].isin(NEW8)
    print('  종속변수는 5자리 KSIC보다 거친 단위로 관측되어 여러 산업이 같은 y를 공유한다.')
    print('  따라서 "n=47산업"은 실제 정보량이 아니다. 유효 결과셀 수가 진짜 n에 가깝다.')
    rows = []
    for nm, m in [('전체47', np.ones(len(s), bool)), ('기존39', ~s['신규'].values),
                  ('대형', s['대형'].values), ('소형', ~s['대형'].values),
                  ('비전자', ~s['전자'].values), ('전자 mid26', s['전자'].values),
                  ('비전자×대형', (~s['전자'] & s['대형']).values),
                  ('비전자×소형', (~s['전자'] & ~s['대형']).values),
                  ('기존39 비전자×대형', (~s['전자'] & s['대형'] & ~s['신규']).values),
                  ('기존39 비전자×소형', (~s['전자'] & ~s['대형'] & ~s['신규']).values),
                  ('기존39 대형', (s['대형'] & ~s['신규']).values),
                  ('기존39 소형', (~s['대형'] & ~s['신규']).values)]:
        a = s[m]
        rows.append(dict(부분표본=nm, n_산업=len(a), 유효결과셀=a['cell'].nunique(),
                         mid수=a['mid'].nunique(), over1비율=a['over1'].mean(),
                         over1_1개수=int(a['over1'].sum())))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print()
    print('  주의 1: 전자 mid26 은 13개 산업이지만 서로 다른 결과가 5개뿐 → H-B 단독 검정 사실상 불가.')
    print('  주의 2: 비전자×대형 은 over1=1 이 16개 중 4개(기존39 기준 15개 중 3개)뿐 →')
    print('          이 층의 β_over1≈0 은 "효과 없음"이 아니라 "깃발 변동 부족"일 수도 있다.')


def main():
    names = load_names()
    d47 = load_panel(PANEL)
    d39 = load_panel(PANEL39)
    base = step0(d47, d39)
    L, base_mean, base_beta, n_new_top10, cmp = step1(d47, names)
    reg = step2(d47, names, 2017)
    summ, R = step3(d47)
    agg = step4(d47, names, reg)
    C, Cc, n_newcell = step6(d47, d39, names)
    ind = step7(d47, names)
    T8 = step8(d47, ind, names)
    step9(d47, ind)
    step5(L, agg, base_mean, n_new_top10, cmp, summ)
    print()
    print('저장: diag_influence_loo.csv / diag_influence_reg2017.csv / diag_influence_traj.csv (분석/fusion/)')


if __name__ == '__main__':
    main()
