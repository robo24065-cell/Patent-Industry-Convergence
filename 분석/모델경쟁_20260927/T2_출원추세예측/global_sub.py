# 전역 모형 학습 표본 확대: (ksic,cc) 250계열 + (ksic,cc,IPC subclass) 하위계열을 풀링해 성장 동학 학습
# 평가·예측은 (ksic,cc) 250계열 기준. 하위계열 값은 동일 결과변수를 복제한 것이 아니라 서로 다른 부분합(가짜 표본 아님),
# 단 같은 (ksic,cc) 안의 하위계열은 상관 → 학습 표본 증강 용도로만 쓰고 평가 단위는 250계열.
import sys, json, warnings
import numpy as np, pandas as pd
from common import *

warnings.filterwarnings('ignore')
MAIN = load_panel()
MI = MAIN.index
d = pd.read_parquet(os.path.join(CACHE, 'panel_kcs.parquet'))
SUBW = d.pivot_table(index=['ksic', 'cc', 'sc'], columns='year', values='n', aggfunc='sum').reindex(columns=YEARS).fillna(0.0)
CTOT = MAIN.groupby(level='cc').sum()


def own_feats(n, c):
    L = np.log1p(n)
    F = {}
    for l in range(1, 6):
        F[f'g_l{l}'] = L[:, c - l + 1] - L[:, c - l]
    F['g3'] = (L[:, c] - L[:, c - 3]) / 3
    F['g5'] = (L[:, c] - L[:, c - 5]) / 5
    F['acc'] = F['g_l1'] - F['g3']
    F['lvl'] = L[:, c]
    return F


def feats(n, ccs, c, ctot):
    F = own_feats(n, c)
    Lt = np.log1p(ctot)
    ci = np.array([CCS.index(x) for x in ccs])
    F['c_g1'] = (Lt[:, c] - Lt[:, c - 1])[ci]
    F['c_g3'] = ((Lt[:, c] - Lt[:, c - 3]) / 3)[ci]
    for j, cj in enumerate(CCS):
        F[f'is_{cj}'] = (ci == j).astype(float)
    return pd.DataFrame(F)


def sub_fn(hist, o, kind='ridge', minavg=20, alpha=10.0, xgbp=None, tmin=2005, include_main=True, weight='mase', recent=None, sub_w=1.0, is_main=False):
    T_ = hist.shape[1]
    ctot = CTOT.values[:, :T_]
    # 하위계열 선택: 원점까지 연평균 >= minavg (원점 이전 정보만)
    S = SUBW.values[:, :T_]
    keep = S.mean(1) >= minavg
    pools = []
    if minavg is not None and minavg < 1e9:
        pools.append((S[keep], SUBW.index.get_level_values('cc')[keep], sub_w, 0.0))
    if include_main:
        pools.append((hist, MI.get_level_values('cc'), 1.0, 1.0))
    out = np.zeros((hist.shape[0], H))
    for h in range(1, H + 1):
        X, y, w = [], [], []
        t_lo = tmin if recent is None else max(tmin, o - h - recent + 1)
        for n, ccs, pw, im in pools:
            L = np.log1p(n)
            for t in range(t_lo, o - h + 1):
                c = col(t)
                f = feats(n, ccs, c, ctot)
                if is_main: f['is_main'] = im
                X.append(f); y.append(L[:, c + h] - L[:, c])
                sc = np.mean(np.abs(np.diff(n[:, :c + 1], axis=1)), axis=1)
                w.append(pw * n[:, c] / np.maximum(sc, 1))
        X = pd.concat(X, ignore_index=True); y = np.concatenate(y); w = np.concatenate(w)
        w = np.clip(w, 0, np.quantile(w, 0.99))
        sw = w if weight == 'mase' else None
        if kind == 'ridge':
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler().fit(X.values)
            m = Ridge(alpha=alpha).fit(ss.transform(X.values), y, sample_weight=sw)
            pred = lambda Z: m.predict(ss.transform(Z))
        else:
            import xgboost as xgb
            p = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                     min_child_weight=20, reg_lambda=5.0, n_jobs=2, tree_method='hist', objective='reg:absoluteerror')
            p.update(xgbp or {})
            m = xgb.XGBRegressor(**p).fit(X.values, y, sample_weight=sw)
            pred = lambda Z: m.predict(Z)
        Xo = feats(hist, MI.get_level_values('cc'), col(o), ctot)
        if is_main: Xo['is_main'] = 1.0
        g = pred(Xo.values)
        out[:, h - 1] = np.expm1(np.log1p(hist[:, -1]) + g)
    return np.maximum(out, 0)


CONFIGS = {
    's_ridge': dict(kind='ridge'),
    's_xgb': dict(kind='xgb'),
    's_ridge_main_only': dict(kind='ridge', minavg=1e12),
    's_xgb_min5': dict(kind='xgb', minavg=5),
    's_xgb_min50': dict(kind='xgb', minavg=50),
    's_xgb_now': dict(kind='xgb', weight='none'),
    's_xgb_l2': dict(kind='xgb', xgbp={'objective': 'reg:squarederror'}),
}

if __name__ == '__main__':
    for nm in sys.argv[1:]:
        cfg = CONFIGS[nm] if nm in CONFIGS else json.loads(nm)
        r = evaluate(sub_fn, **cfg)
        report(r, nm)
        save_pred(r, nm if nm in CONFIGS else 'custom')
