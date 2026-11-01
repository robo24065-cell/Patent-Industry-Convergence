from lab2 import *
import final_run as FR
M = {k: member_cache(k, v) for k, v in BASE.items()}
# (1) 4번째 구성원: 1단계 v25 MASE공간 Huber (저장 예측; 행 순서 확인)
sc = pd.read_parquet(os.path.join(FR.T, 'preds', 'sc_huber.parquet'))
assert (sc[['ksic', 'cc', 'origin', 'h']].values == NB[['ksic', 'cc', 'origin', 'h']].values).all()
SC = {o: sc[sc.origin == o].pivot_table(index=['ksic', 'cc'], columns='h', values='f').reindex(D['idx']).values for o in FR.VAL_ORIGINS}
for o in ORIG_ALL:
    SC.setdefault(o, np.zeros((250, H)))
rep(rows_from(SC), 'sc_huber(v25)')
rep(rows_from(combine(list(M.values()) + [SC])), 'ens+sc_huber')
# (2) 전역모형 강건손실: Huber(로그성장, 표본가중) — 드론형 폭증 표본의 과대 영향 완화
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
def ridge_huber(E, o, H, alpha=10.0, tmin=2005, clipq=0.99, eps=1.35):
    n = E['n']; L = np.log1p(n); co = n.shape[1] - 1; out = np.zeros((n.shape[0], H))
    for h in range(1, H + 1):
        X, y, w = [], [], []
        for t in range(tmin, o - h + 1):
            c = t - FR.Y0
            X.append(FR._ridge_feats(E, c)); y.append(L[:, c + h] - L[:, c])
            s = np.mean(np.abs(np.diff(n[:, :c + 1], axis=1)), axis=1); w.append(n[:, c] / np.maximum(s, 1))
        X = pd.concat(X, ignore_index=True); y = np.concatenate(y); w = np.concatenate(w)
        w = np.clip(w, 0, np.quantile(w, clipq)); w = w / w.mean()
        ss = StandardScaler().fit(X.values)
        m = HuberRegressor(alpha=alpha * 1e-3, epsilon=eps, max_iter=2000).fit(ss.transform(X.values), y, sample_weight=w)
        out[:, h - 1] = np.expm1(L[:, -1] + m.predict(ss.transform(FR._ridge_feats(E, co)[X.columns].values)))
    return np.maximum(out, 0)
FR.MEMBER_FNS['ridge_huber'] = ridge_huber
for eps in [1.35, 2.0]:
    RH = member_cache('ridge_huber', {'eps': eps})
    rep(rows_from(RH), f'ridge_huber eps{eps}')
    rep(rows_from(combine([RH, M['flow_um'], M['topdown']])), f'  ens w/ ridge_huber eps{eps}')
