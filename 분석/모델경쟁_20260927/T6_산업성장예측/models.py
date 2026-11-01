# 모형 팩토리: fit_predict(train, test) 반환
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import QuantileRegressor, HuberRegressor, Ridge
warnings.filterwarnings('ignore')


def prep(tr, te, feats, clip_q=0.01):
    Xtr = tr[feats].astype(float).copy()
    Xte = te[feats].astype(float).copy()
    lo, hi = Xtr.quantile(clip_q), Xtr.quantile(1 - clip_q)
    Xtr = Xtr.clip(lo, hi, axis=1)
    Xte = Xte.clip(lo, hi, axis=1)
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    return ((Xtr - mu) / sd).values, ((Xte - mu) / sd).values


def train_rows(tr, drop2010=True, ycap=None):
    tr = tr[tr.year != 2010] if drop2010 else tr
    y = tr.y.values.copy()
    if ycap is not None:
        y = np.clip(y, -ycap, ycap)
    return tr, y


def lad(feats, alpha=0.001, ycap=None, weight_recent=None):
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=ycap)
        Xtr, Xte = prep(tr, te, feats)
        sw = None
        if weight_recent:
            sw = weight_recent ** (tr.year.max() - tr.year.values)
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver='highs').fit(Xtr, y, sample_weight=sw)
        return m.predict(Xte)
    return fp


def huber(feats, eps=1.35, alpha=1.0, ycap=None):
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=ycap)
        Xtr, Xte = prep(tr, te, feats)
        m = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=500).fit(Xtr, y)
        return m.predict(Xte)
    return fp


def ridge(feats, alpha=10.0, ycap=1.0):
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=ycap)
        Xtr, Xte = prep(tr, te, feats)
        m = Ridge(alpha=alpha).fit(Xtr, y)
        return m.predict(Xte)
    return fp


def xgb(feats, params=None, n_estimators=300, ycap=1.0):
    import xgboost as xg
    P = dict(max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
             reg_lambda=10.0, objective='reg:absoluteerror', nthread=2, verbosity=0)
    if params:
        P.update(params)

    def fp(tr, te):
        tr, y = train_rows(tr, ycap=ycap)
        m = xg.XGBRegressor(n_estimators=n_estimators, **P)
        m.fit(tr[feats].astype(float), y)
        return m.predict(te[feats].astype(float))
    return fp


def median_of(tr_col):
    def fp(tr, te):
        return np.full(len(te), 0.0)
    return fp


# ---- 연도 단면 중앙값 제거 타깃(공통 충격 제거) + 무절편 + 축소 ----
def dm_target(tr, y):
    med = pd.Series(y, index=tr.index).groupby(tr.year.values).transform('median').values
    return y - med


def lad_dm(feats, alpha=0.0, shrink=1.0, ycap=None):
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=ycap)
        y = dm_target(tr, y)
        Xtr, Xte = prep(tr, te, feats)
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver='highs', fit_intercept=False).fit(Xtr, y)
        return shrink * m.predict(Xte)
    return fp


def nested(factory, grid, inner_k=4, eval_all=True, min_inner_year=2012):
    """원점 T마다 학습구간 안에서 롤링 내부검증(최근 inner_k개 연도)으로 하이퍼파라미터 선택 후 전체 학습구간으로 재적합.
    factory(**params) -> fp. grid: list of dict. 내부평가 MAE는 488개 전체(eval_all) 또는 50개."""
    chosen = []

    def fp(tr, te):
        yrs = sorted(y for y in tr.year.unique() if y >= min_inner_year)[-inner_k:]
        best, bestv = grid[0], np.inf
        if len(yrs) >= 1:
            for g in grid:
                errs = []
                for t in yrs:
                    itr = tr[tr.year < t]
                    ite = tr[tr.year == t]
                    if not eval_all:
                        ite = ite[ite.in50 == 1]
                    if itr[itr.year != 2010].shape[0] < 100:
                        continue
                    p = factory(**g)(itr, ite)
                    errs.append(np.mean(np.abs(ite.y.values - p)))
                if errs and np.mean(errs) < bestv:
                    bestv, best = np.mean(errs), g
        chosen.append((int(te.year.iloc[0]), best))
        return factory(**best)(tr, te)
    fp.chosen = chosen
    return fp


def xgb_dm(feats, n_estimators=300, ycap=1.0, shrink=1.0, rank=False, seed=0, decay=None, **kw):
    """XGBoost(절대오차), 타깃=y-연도중앙값, 기본 예측 0에서 출발(base_score=0)."""
    import xgboost as xg
    P = dict(max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
             reg_lambda=10.0, objective='reg:absoluteerror', nthread=2, verbosity=0, base_score=0.0)
    P.update(kw)

    def fp(tr, te):
        tr, y = train_rows(tr, ycap=ycap)
        y = dm_target(tr, y)
        sw = (decay ** (tr.year.max() - tr.year.values)) if decay else None
        Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)
        if rank:  # 연도 내 백분위 순위(이상치 영향 제거)
            Xtr = Xtr.groupby(tr.year.values).rank(pct=True)
            Xte = Xte.groupby(te.year.values).rank(pct=True)
        m = xg.XGBRegressor(n_estimators=n_estimators, **P, random_state=seed)
        m.fit(Xtr, y, sample_weight=sw)
        return shrink * m.predict(Xte)
    return fp


def guard(fp, min_rows=1500, drop_years=(2010, 2011), min_year=2009):
    """학습 레이블 필터(2년 연율화 2011 제외 등) + 학습 행이 부족하면 기준선(0) 반환."""
    def g(tr, te):
        tr = tr[~tr.year.isin(drop_years) & (tr.year >= min_year)]
        if len(tr) < min_rows:
            return np.zeros(len(te))
        return fp(tr, te)
    for a in ('chosen',):
        if hasattr(fp, a):
            setattr(g, a, getattr(fp, a))
    return g


def stage2(base_version, feats, kind='lad', alpha=0.01, shrink=0.5, min_years=2, rank=True, **kw):
    """2단계 잔차 보정: base 버전의 롤링 OOF 예측(preds/<base>.parquet, 각 연도는 그 이전 자료로 예측됨)을 불러와
    50개 산업의 잔차 r=y-p 를 특허 피처로 설명. 목표연도 T의 잔차모형은 OOF 연도 < T 만 사용."""
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    B = pd.read_parquet(os.path.join(HERE, f'preds/{base_version}.parquet'))[['ksic', 'year', 'pred']].rename(columns={'pred': 'pb'})

    def fp(tr, te):
        T = int(te.year.iloc[0])
        te2 = te.merge(B, on=['ksic', 'year'], how='left')
        pb = te2.pb.fillna(0).values
        hist = tr[(tr.in50 == 1)].merge(B, on=['ksic', 'year'], how='inner')
        hist = hist[hist.year < T]
        if hist.year.nunique() < min_years:
            return pb
        r = (hist.y - hist.pb).clip(-1, 1).values
        Xh, Xt = hist[feats].astype(float), te2[feats].astype(float)
        if rank:
            Xh = Xh.groupby(hist.year.values).rank(pct=True) - 0.5
            Xt = Xt.groupby(te2.year.values).rank(pct=True) - 0.5
        Xh, Xt = Xh.fillna(0).values, Xt.fillna(0).values
        if kind == 'lad':
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver='highs', fit_intercept=False).fit(Xh, r)
        else:
            import xgboost as xg
            P = dict(max_depth=1, learning_rate=0.03, n_estimators=150, min_child_weight=20, subsample=0.8,
                     objective='reg:absoluteerror', nthread=2, verbosity=0, base_score=0.0, reg_lambda=10)
            P.update(kw)
            m = xg.XGBRegressor(**P, random_state=0).fit(Xh, r)
        return pb + shrink * m.predict(Xt)
    return fp


def xgb_z(feats, n_estimators=300, shrink=1.0, rank=True, seeds=(0,), sd_floor=0.05, zcap=4.0, **kw):
    """변동성 표준화 타깃: z=(y-연도중앙값)/sd_i (sd_i=T-1까지 확장 표준편차, 하한 sd_floor), 예측 후 sd_i 곱해 복원. seeds 여러 개면 배깅 평균."""
    import xgboost as xg
    P = dict(max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
             reg_lambda=10.0, objective='reg:absoluteerror', nthread=2, verbosity=0, base_score=0.0)
    P.update(kw)

    def fp(tr, te):
        tr, y = train_rows(tr, ycap=None)
        y = dm_target(tr, y)
        sdtr = tr.gsd_exp.fillna(tr.gsd_exp.median()).clip(lower=sd_floor).values
        sdte = te.gsd_exp.fillna(tr.gsd_exp.median()).clip(lower=sd_floor).values
        z = np.clip(y / sdtr, -zcap, zcap)
        Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)
        if rank:
            Xtr = Xtr.groupby(tr.year.values).rank(pct=True)
            Xte = Xte.groupby(te.year.values).rank(pct=True)
        ps = []
        for s in seeds:
            m = xg.XGBRegressor(n_estimators=n_estimators, **P, random_state=s)
            m.fit(Xtr, z)
            ps.append(m.predict(Xte))
        return shrink * np.mean(ps, axis=0) * sdte
    return fp


def xgb_bag(feats, seeds=(0, 1, 2, 3, 4), **kw):
    fs = [xgb_dm(feats, seed=s, **kw) for s in seeds]
    def fp(tr, te):
        return np.mean([f(tr, te) for f in fs], axis=0)
    return fp


def guard2(fp, min_rows=800, count_excl=(2010, 2011), drop_years=(2010,)):
    """문턱은 count_excl 연도를 뺀 학습행 수로 판단, 실제 학습은 drop_years만 제외(2011 연율화 레이블은 보조로 포함)."""
    def g(tr, te):
        if (~tr.year.isin(count_excl)).sum() < min_rows:
            return np.zeros(len(te))
        return fp(tr[~tr.year.isin(drop_years)], te)
    return g


def add_level(fp, kind='median_hist', k=None, w=1.0):
    """공통 수준 가산: 과거 연도별 단면 중앙값(488개)의 중앙값(또는 최근 k년) × w. 학습구간(year<T)만 사용."""
    def g(tr, te):
        t2 = tr[~tr.year.isin((2010,))]
        meds = t2.groupby('year').y.median().sort_index()
        if k:
            meds = meds.iloc[-k:]
        lvl = float(meds.median()) if len(meds) else 0.0
        return fp(tr, te) + w * lvl
    return g


def lad_rank(feats, alpha=0.001, shrink=1.0):
    """연도 내 순위(-0.5~0.5) 피처 + 중앙값 제거 타깃 LAD(무절편)."""
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=1.0)
        y = dm_target(tr, y)
        Xtr = (tr[feats].astype(float).groupby(tr.year.values).rank(pct=True) - 0.5).fillna(0).values
        Xte = (te[feats].astype(float).groupby(te.year.values).rank(pct=True) - 0.5).fillna(0).values
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver='highs', fit_intercept=False).fit(Xtr, y)
        return shrink * m.predict(Xte)
    return fp


def drift_shrink(kappa=0.3):
    """통계 시계열 족: 확률보행+드리프트, 드리프트=산업 확장평균 성장(연도중앙값 제거)×kappa(축소)."""
    def fp(tr, te):
        return kappa * (te.gmean_exp.fillna(0).values - te.cm_all1.fillna(0).values * 0)
    return fp


def blend(fps, ws):
    def fp(tr, te):
        return np.sum([w * f(tr, te) for f, w in zip(fps, ws)], axis=0)
    return fp


def hgb_rank(feats, max_iter=200, lr=0.03, max_depth=3, min_leaf=40, seeds=(0, 1, 2), l2=5.0, shrink=1.0):
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=1.0)
        y = dm_target(tr, y)
        Xtr = tr[feats].astype(float).groupby(tr.year.values).rank(pct=True).values
        Xte = te[feats].astype(float).groupby(te.year.values).rank(pct=True).values
        ps = []
        for s in seeds:
            m = H(loss='absolute_error', max_iter=max_iter, learning_rate=lr, max_depth=max_depth, min_samples_leaf=min_leaf,
                  l2_regularization=l2, random_state=s, early_stopping=False, max_features=0.8)
            ps.append(m.fit(Xtr, y).predict(Xte))
        return shrink * np.mean(ps, axis=0)
    return fp


def xgb_w(feats, w50=3.0, seeds=(0, 1, 2, 3, 4), n_estimators=300, **kw):
    """50개 산업 행 가중(w50배) XGB 순위 모형."""
    import xgboost as xg
    P = dict(max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
             reg_lambda=10.0, objective='reg:absoluteerror', nthread=2, verbosity=0, base_score=0.0)
    P.update(kw)
    def fp(tr, te):
        tr, y = train_rows(tr, ycap=1.0)
        y = dm_target(tr, y)
        Xtr = tr[feats].astype(float).groupby(tr.year.values).rank(pct=True)
        Xte = te[feats].astype(float).groupby(te.year.values).rank(pct=True)
        sw = np.where(tr.in50.values == 1, w50, 1.0)
        ps = [xg.XGBRegressor(n_estimators=n_estimators, **P, random_state=s).fit(Xtr, y, sample_weight=sw).predict(Xte) for s in seeds]
        return np.mean(ps, axis=0)
    return fp
