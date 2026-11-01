# 전역(global) 모형: 250개 (ksic,cc) 시계열을 한 모형으로 학습, 타깃 = h년 뒤 로그성장률
# 원점 o 에서 학습 표본은 기준연도 t, 목표 t+h <= o 인 것만 사용 (롤링 원점, 원점 이전 자료만)
import sys, json, warnings
import numpy as np, pandas as pd
from common import *

warnings.filterwarnings('ignore')
ALL = load_all()
IDX = ALL['n'].index
KS = sorted(set(IDX.get_level_values(0)))
CCI = np.array([CCS.index(c) for c in IDX.get_level_values(1)])
KSI = np.array([KS.index(k) for k in IDX.get_level_values(0)])


def feats_at(A, t, fs):
    """A: dict name -> ndarray(250, T) (원점까지 절단된 자료). t: 기준연도 (열 인덱스는 col(t)).
    반환 DataFrame(250 x F), 기준연도 t 까지 정보만 사용."""
    n = A['n']; L = np.log1p(n); c = col(t)
    F = {}
    for l in range(1, 6):
        F[f'g_l{l}'] = L[:, c - l + 1] - L[:, c - l]
    F['g3'] = (L[:, c] - L[:, c - 3]) / 3
    F['g5'] = (L[:, c] - L[:, c - 5]) / 5
    F['acc'] = F['g_l1'] - F['g3']
    F['lvl'] = L[:, c]
    F['cc'] = CCI.astype(float)
    if 'xc' in fs:  # 같은 KSIC 타국 성장(국가 간 선행 신호)
        for j, cj in enumerate(CCS):
            m = CCI == j
            gl1 = np.zeros(len(KS)); g3 = np.zeros(len(KS))
            gl1[KSI[m]] = F['g_l1'][m]; g3[KSI[m]] = F['g3'][m]
            F[f'x_{cj}_g1'] = gl1[KSI]; F[f'x_{cj}_g3'] = g3[KSI]
    if 'ctot' in fs:  # 같은 국가 전체(50개 KSIC 합) 성장
        tot = np.zeros((5, n.shape[1]))
        for j in range(5):
            tot[j] = n[CCI == j].sum(0)
        Lt = np.log1p(tot)
        F['c_g1'] = (Lt[:, c] - Lt[:, c - 1])[CCI]
        F['c_g3'] = ((Lt[:, c] - Lt[:, c - 3]) / 3)[CCI]
        F['rel_g3'] = F['g3'] - F['c_g3']
        share = n[:, c] / tot[CCI, c]
        share3 = n[:, c - 3] / np.maximum(tot[CCI, c - 3], 1)
        F['share_chg'] = np.log((share + 1e-6) / (share3 + 1e-6))
    if 'ktot' in fs:  # 같은 KSIC 5개국 합 성장
        kt = np.zeros((len(KS), n.shape[1]))
        for k in range(len(KS)):
            kt[k] = n[KSI == k].sum(0)
        Lk = np.log1p(kt)
        F['k_g1'] = (Lk[:, c] - Lk[:, c - 1])[KSI]
        F['k_g3'] = ((Lk[:, c] - Lk[:, c - 3]) / 3)[KSI]
    if 'comp' in fs:  # 구성 신호: 실용신안 비중, 외국우선권 비중, 신규출원인 비중, 출원인 수 성장
        nn = np.maximum(n[:, c], 1)
        F['um_sh'] = A['n_um'][:, c] / nn
        F['um_sh_chg'] = F['um_sh'] - A['n_um'][:, c - 3] / np.maximum(n[:, c - 3], 1)
        F['fprio_sh'] = A['n_fprio'][:, c] / nn
        F['fprio_chg'] = F['fprio_sh'] - A['n_fprio'][:, c - 3] / np.maximum(n[:, c - 3], 1)
        F['newapp_sh'] = A['n_newapp'][:, c] / nn
        F['app_g3'] = (np.log1p(A['n_app'][:, c]) - np.log1p(A['n_app'][:, c - 3])) / 3
    return pd.DataFrame(F)


def build_train(A, o, h, fs, tmin=2005):
    X, y, w = [], [], []
    L = np.log1p(A['n'])
    for t in range(tmin, o - h + 1):
        f = feats_at(A, t, fs)
        f['h'] = h
        X.append(f); y.append(L[:, col(t + h)] - L[:, col(t)])
        sc = np.mean(np.abs(np.diff(A['n'][:, :col(t) + 1], axis=1)), axis=1)
        w.append(A['n'][:, col(t)] / np.maximum(sc, 1))
    return pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(w)


def make_model(kind, params):
    if kind == 'xgb':
        import xgboost as xgb
        p = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=5, reg_lambda=5.0, n_jobs=2, tree_method='hist', objective='reg:absoluteerror')
        p.update(params)
        return xgb.XGBRegressor(**p)
    if kind == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=params.get('alpha', 10.0)))
    if kind == 'huber':
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=params.get('alpha', 1.0), max_iter=500))
    raise ValueError(kind)


def global_fn(hist, o, fs=(), kind='xgb', params=None, weight='mase', tmin=2005, clip=None, cc_onehot=False):
    params = params or {}
    A = {k: v.values[:, :hist.shape[1]].astype(float) for k, v in ALL.items()}
    assert np.allclose(A['n'], hist)
    out = np.zeros((hist.shape[0], H))
    Lo = np.log1p(hist[:, -1])
    for h in range(1, H + 1):
        X, y, w = build_train(A, o, h, fs, tmin)
        if cc_onehot or kind != 'xgb':
            for j, cj in enumerate(CCS):
                X[f'is_{cj}'] = (X['cc'] == j).astype(float)
        Xo = feats_at(A, o, fs); Xo['h'] = h
        if cc_onehot or kind != 'xgb':
            for j, cj in enumerate(CCS):
                Xo[f'is_{cj}'] = (Xo['cc'] == j).astype(float)
        X = X.drop(columns=['h']); Xo = Xo[X.columns]
        m = make_model(kind, params)
        sw = w if weight == 'mase' else None
        if kind == 'huber' or kind == 'ridge':
            m.fit(X.values, y, **({m.steps[-1][0] + '__sample_weight': sw} if sw is not None else {}))
        else:
            m.fit(X.values, y, sample_weight=sw)
        g = m.predict(Xo.values)
        if clip is not None:
            g = np.clip(g, -clip, clip)
        out[:, h - 1] = np.expm1(Lo + g)
    return np.maximum(out, 0)


CONFIGS = {
    'g_own_xgb': dict(fs=()),
    'g_own_ridge': dict(fs=(), kind='ridge'),
    'g_own_xgb_now': dict(fs=(), weight='none'),
    'g_xc_xgb': dict(fs=('xc',)),
    'g_xc_ctot_xgb': dict(fs=('xc', 'ctot')),
    'g_xc_ctot_ktot_xgb': dict(fs=('xc', 'ctot', 'ktot')),
    'g_all_xgb': dict(fs=('xc', 'ctot', 'ktot', 'comp')),
    'g_all_ridge': dict(fs=('xc', 'ctot', 'ktot', 'comp'), kind='ridge'),
    'g_all_huber': dict(fs=('xc', 'ctot', 'ktot', 'comp'), kind='huber'),
}

if __name__ == '__main__':
    for n in sys.argv[1:]:
        cfg = CONFIGS[n] if n in CONFIGS else json.loads(n)
        r = evaluate(global_fn, **cfg)
        report(r, n)
        save_pred(r, n if n in CONFIGS else 'custom')
