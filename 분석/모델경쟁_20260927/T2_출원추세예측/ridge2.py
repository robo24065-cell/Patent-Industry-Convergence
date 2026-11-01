# 전역 Ridge v2: v13 피처 + 우선권 흐름 선행지표 + 자국기원/해외기원 분리 성장 + (옵션) 타국 같은 KSIC 성장
import sys, json, warnings
import numpy as np, pandas as pd
from common import *
from flow import DOM, FY, OTHER, SRC, CCI, KSI, ROW, IDX

warnings.filterwarnings('ignore')
MAIN = load_panel()
CTOT = MAIN.groupby(level='cc').sum().values


def feats(n, c, T_, fs):
    L = np.log1p(n)
    F = {}
    for l in range(1, 6):
        F[f'g_l{l}'] = L[:, c - l + 1] - L[:, c - l]
    F['g3'] = (L[:, c] - L[:, c - 3]) / 3
    F['g5'] = (L[:, c] - L[:, c - 5]) / 5
    if 'long' in fs:
        F['g8'] = (L[:, c] - L[:, c - 5]) / 5 if c < 8 else (L[:, c] - L[:, c - 8]) / 8
    F['acc'] = F['g_l1'] - F['g3']
    F['lvl'] = L[:, c]
    Lt = np.log1p(CTOT[:, :T_])
    F['c_g1'] = (Lt[:, c] - Lt[:, c - 1])[CCI]
    F['c_g3'] = ((Lt[:, c] - Lt[:, c - 3]) / 3)[CCI]
    if 'long' in fs:
        F['c_g8'] = ((Lt[:, c] - Lt[:, max(c - 8, 0)]) / min(8, c))[CCI]
    for j, cj in enumerate(CCS):
        F[f'is_{cj}'] = (CCI == j).astype(float)
    nn = np.maximum(n[:, c], 1)
    if 'lead' in fs:
        # 원천국 자국기원 출원 성장(해외기원 비중 가중): 다음해 유입 선행
        num = np.zeros(len(n)); lead1 = np.zeros(len(n)); lead3 = np.zeros(len(n))
        for Y in CCS:
            S = SRC[Y]; Fy = FY[Y]
            sh = Fy[:, c] / nn
            num += sh
            lead1 += sh * (np.log1p(S[:, c]) - np.log1p(S[:, c - 1]))
            lead3 += sh * (np.log1p(S[:, c]) - np.log1p(S[:, c - 3])) / 3
        F['for_sh'] = num
        F['lead_g1'] = lead1
        F['lead_g3'] = lead3
        D = DOM
        F['dom_g3'] = (np.log1p(D[:, c]) - np.log1p(D[:, c - 3])) / 3
    if 'xc' in fs:
        for j, cj in enumerate(CCS):
            rows = [ROW[(KSI[i], j)] for i in range(len(n))]
            F[f'x_{cj}_g3'] = F['g3'][rows]
    if 'int' in fs:  # 국가별 기울기(모멘텀·평균회귀 계수의 국가 이질성)
        for j, cj in enumerate(CCS):
            m = (CCI == j).astype(float)
            for f in ['g_l1', 'g3', 'g5', 'c_g3', 'acc']:
                F[f'{f}_x_{cj}'] = F[f] * m
    return pd.DataFrame(F)


def r2_fn(hist, o, fs=('lead',), alpha=10.0, tmin=2005, kind='ridge', clipq=0.99, per_h=True, decay=None):
    T_ = hist.shape[1]
    n = hist
    L = np.log1p(n)
    out = np.zeros((n.shape[0], H))
    for h in range(1, H + 1):
        X, y, w = [], [], []
        for t in range(tmin, o - h + 1):
            c = col(t)
            X.append(feats(n, c, T_, fs)); y.append(L[:, c + h] - L[:, c])
            sc = np.mean(np.abs(np.diff(n[:, :c + 1], axis=1)), axis=1)
            w.append(n[:, c] / np.maximum(sc, 1))
        X = pd.concat(X, ignore_index=True); y = np.concatenate(y); w = np.concatenate(w)
        w = np.clip(w, 0, np.quantile(w, clipq))
        if decay is not None:  # 최근 기준연도 가중(시간 감쇠)
            ages = np.concatenate([np.full(n.shape[0], o - h - t) for t in range(tmin, o - h + 1)])
            w = w * decay ** ages
        Xo = feats(n, col(o), T_, fs)[X.columns]
        if kind == 'ridge':
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler().fit(X.values)
            m = Ridge(alpha=alpha).fit(ss.transform(X.values), y, sample_weight=w)
            g = m.predict(ss.transform(Xo.values))
        else:
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=10, reg_lambda=5.0, n_jobs=2, tree_method='hist',
                                 objective='reg:absoluteerror').fit(X.values, y, sample_weight=w)
            g = m.predict(Xo.values)
        out[:, h - 1] = np.expm1(L[:, -1] + g)
    return np.maximum(out, 0)


if __name__ == '__main__':
    runs = [('r2_base', {'fs': ()}), ('r2_lead', {}), ('r2_lead_xc', {'fs': ('lead', 'xc')}), ('r2_lead_xgb', {'kind': 'xgb'}),
            ('r2_long', {'fs': ('long',)}), ('r2_long_t2008', {'fs': ('long',), 'tmin': 2008})]
    if len(sys.argv) > 1:
        runs = [(f'r2_custom{i}', json.loads(a)) for i, a in enumerate(sys.argv[1:])]
    for nm, kw in runs:
        r = evaluate(r2_fn, **kw)
        report(r, nm, seg=True)
        save_pred(r, nm)
