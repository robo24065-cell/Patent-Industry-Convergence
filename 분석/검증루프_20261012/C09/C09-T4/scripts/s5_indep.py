# C09-T4 s5: 기한 요일(주말·공휴일)과 레이블(T4 포기·N4 폐업)의 독립성 검정(PREREG §5).
#  T4: 평가행 특허의 과거(≤L) 갱신 납부 기한 중 주말·공휴일 비중 wkd_share, 주말·공휴일 때문에만 지연으로 잡힌 납부 수 n_fake.
#      개발 원점 O12–O18(label_full) 우선, 최종창 F22(label_np)는 진단 병기.
#  N4: 기업 단위 직전 2년(statAnnl≥4) 납부 기한의 주말·공휴일 비중과 폐업 레이블.
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
from s2_t4_feats import fdir
from scipy.stats import chi2_contingency
import statsmodels.api as sm

hol = load_hol()
p = pd.read_parquet(os.path.join(C, 't4_pay.parquet'))
p = p[p.seq > 0].copy()
due = p.due.values.astype('datetime64[D]')
p['off'] = (dow(due) >= 5) | np.isin(due, hol)
p['fake'] = (p.early < 0) & (p.earlyR2 >= 0)


def logit_ci(y, X):
    X = sm.add_constant(X.astype(float), has_constant='add')
    m = sm.GLM(y.astype(float), X, family=sm.families.Binomial()).fit()
    b = m.params.iloc[1]; se = m.bse.iloc[1]
    return dict(coef=float(b), lo=float(b - 1.96 * se), hi=float(b + 1.96 * se), p=float(m.pvalues.iloc[1]))


out = {'T4': {}, 'N4': {}}
for o in ['O12', 'O14', 'O16', 'O18', 'F22']:
    ev = pd.read_parquet(os.path.join(fdir(o), 'ev.parquet'), columns=['pid', 'L', 'y', 'age', 'n_renew', 'n_late', 'atype'])
    ev = ev[ev.y.notna()].reset_index(drop=True)
    rows = []
    for L, g in ev.groupby('L'):
        q = p[(p.pd.values <= np.datetime64(str(L))) & p.pid.isin(set(g.pid))]
        a = q.groupby('pid').agg(n=('off', 'size'), off=('off', 'sum'), fake=('fake', 'sum'))
        gg = g.join(a, on='pid')
        rows.append(gg)
    d = pd.concat(rows, ignore_index=True)
    d['wkd_share'] = d.off / d.n
    dd = d[d.n > 0].copy()
    bins = pd.cut(dd.wkd_share, [-0.01, 0.0, 0.5, 1.0], labels=['0', '(0,0.5]', '>0.5'])
    tab = pd.crosstab(bins, dd.y)
    chi = chi2_contingency(tab.values)
    X = pd.concat([dd[['wkd_share', 'age']], np.log1p(dd[['n_renew']]), pd.get_dummies(dd.atype, prefix='a', drop_first=True, dtype=float)], axis=1)
    lc = logit_ci(dd.y.values, X)
    # 정밀판: 과거 기한 수 n 을 층(더미, 15 이상 묶음)으로 통제 — wkd_share 의 분포가 n 에 따라 달라 원시 교차표는 n 구성에 혼동된다
    Xn = pd.concat([dd[['wkd_share', 'age']], pd.get_dummies(dd.n.clip(upper=15).astype(int), prefix='n', drop_first=True, dtype=float),
                    pd.get_dummies(dd.atype, prefix='a', drop_first=True, dtype=float), pd.get_dummies(dd.L.astype(str), prefix='L', drop_first=True, dtype=float)], axis=1)
    lcn = logit_ci(dd.y.values, Xn)
    # n 층 안 순위 비교: 층마다 wkd_share 상위 절반 대 하위 절반의 포기율 차이(층 크기 가중)
    hi_lo = []
    for nn, g in dd.groupby(dd.n.clip(upper=15)):
        if g.wkd_share.nunique() < 2: continue
        md_ = g.wkd_share.median(); a_ = g[g.wkd_share > md_].y.mean(); b_ = g[g.wkd_share <= md_].y.mean()
        if np.isfinite(a_) and np.isfinite(b_): hi_lo.append((len(g), a_ - b_))
    hl = float(np.average([x[1] for x in hi_lo], weights=[x[0] for x in hi_lo])) if hi_lo else None
    # n_fake: 지연 이력이 있는 특허 안에서 '가짜 지연' 비중과 레이블
    lt = dd[dd.n_late > 0].copy(); lt['fake_share'] = lt.fake / lt.n_late
    X2 = pd.concat([lt[['fake_share', 'age']], np.log1p(lt[['n_renew', 'n_late']]), pd.get_dummies(lt.atype, prefix='a', drop_first=True, dtype=float)], axis=1)
    lc2 = logit_ci(lt.y.values, X2)
    out['T4'][o] = dict(n=int(len(dd)), rate_by_bin={str(k): float(v) for k, v in dd.groupby(bins, observed=False).y.mean().items()},
                        n_by_bin={str(k): int(v) for k, v in bins.value_counts().items()}, chi2=float(chi[0]), chi2_p=float(chi[1]),
                        logit_wkd_share=lc, logit_wkd_share_nstrata=lcn, within_n_hi_minus_lo_rate=hl, n_with_late=int(len(lt)), rate_fake_any={'fake=0': float(lt[lt.fake == 0].y.mean()), 'fake>0': float(lt[lt.fake > 0].y.mean())},
                        logit_fake_share_among_late=lc2)
    print('T4', o, out['T4'][o]['rate_by_bin'], 'chi2 p', round(chi[1], 4), 'wkd coef', {k: round(v, 4) for k, v in lc.items()}, 'n층 coef', {k: round(v, 4) for k, v in lcn.items()}, 'n층 상-하', round(hl, 4), 'fake_share coef', {k: round(v, 4) for k, v in lc2.items()}, flush=True)

# ---- N4
py = pd.read_parquet(os.path.join(N4D, 'cache', 'reg_py.parquet'))
for mode, t0s in [('dev', [2017, 2018, 2019, 2020, 2021]), ('final', [2023])]:
    P = pd.read_parquet(os.path.join(N4D, 'cache', f'{mode}_panel.parquet'), columns=['applicantCd', 't0', 'y', 'log_live', 'nps_present', 'late_share_2y'])
    res = []
    for t0 in t0s:
        B = P[P.t0 == t0]
        end = t0 * 10000 + 1231; y2 = end - 20000
        w = py[(py.d <= end) & (py.d > y2) & (py.statAnnl >= 4) & py.rgstDate.notna() & py.applicantCd.isin(set(B.applicantCd))].copy()
        yy = (w.rgstDate // 10000 + w.statAnnl - 1).astype(int); md = (w.rgstDate % 10000).astype(int); md = md.where(md != 229, 228)
        dd_ = pd.to_datetime((yy * 10000 + md).astype(str), format='%Y%m%d', errors='coerce').values.astype('datetime64[D]')
        ok = ~np.isnat(dd_)
        off = np.zeros(len(w), bool); off[ok] = (dow(dd_[ok]) >= 5) | np.isin(dd_[ok], hol)
        w['off'] = off
        g = w.groupby('applicantCd').agg(n=('off', 'size'), off=('off', 'mean'))
        bb = B.join(g, on='applicantCd')
        res.append(bb[bb.n.notna()])
    d = pd.concat(res, ignore_index=True)
    X = pd.concat([d[['off', 'log_live', 'nps_present']], np.log1p(d[['n']])], axis=1)
    lc = logit_ci(d.y.values, X)
    Xn = pd.concat([d[['off', 'log_live', 'nps_present']], pd.get_dummies(pd.qcut(d.n.rank(method='first'), 10, labels=False), prefix='nq', drop_first=True, dtype=float),
                    pd.get_dummies(d.t0, prefix='t', drop_first=True, dtype=float)], axis=1)
    lcn = logit_ci(d.y.values, Xn)
    bins = pd.cut(d.off, [-0.01, 0.2, 0.4, 1.0], labels=['<=0.2', '(0.2,0.4]', '>0.4'])
    tab = pd.crosstab(bins, d.y); chi = chi2_contingency(tab.values)
    out['N4'][mode] = dict(n=int(len(d)), rate_by_bin={str(k): float(v) for k, v in d.groupby(bins, observed=False).y.mean().items()},
                           n_by_bin={str(k): int(v) for k, v in bins.value_counts().items()}, chi2_p=float(chi[1]), logit_offday_share=lc, logit_offday_share_nstrata=lcn)
    print('N4', mode, out['N4'][mode], flush=True)
json.dump(out, open(os.path.join(C, 's5_indep.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
