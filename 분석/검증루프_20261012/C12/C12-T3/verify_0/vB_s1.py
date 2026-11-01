# -*- coding: utf-8 -*-
"""verify_0 V-B: (B) S1을 원천 CSV에서 다른 코드(numpy 월 배열 → 창 평균)로 다시 만들고, 학습을 새로 작성한 함수로 직접 호출해 S1 seed 0·1 skill을 대조.
반박 시도: 단절 시점 2014-01(V1), 연 단위 걸침 판정(V2). seed 0·1. 작업 코드(scripts/*)·cache 미사용(동결 spec·C09 피처 행렬·원천 CSV만)."""
import os, sys, json, time
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, nbinom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
R3 = ROOT + r"\분석\모델경쟁_20260927\T3_기술셀부상예측\R3"
V0 = ROOT + r"\분석\검증루프_20261012\C09\C09-T3\verify_0"
CSV = ROOT + r"\공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv"
OUTD = os.path.dirname(os.path.abspath(__file__))
t = time.time()
raw = pd.read_csv(CSV, dtype={'산업코드': str})
raw = raw[raw.항목코드.isin(['T10', 'T11']) & (raw.연월 <= 202111)]
raw['mi'] = (raw.연월 // 100 - 2000) * 12 + raw.연월 % 100 - 1          # 월 번호(2000-01=0)
NM = int(raw.mi.max()) + 1
arr = {}
for (c, it), g in raw.groupby(['산업코드', '항목코드']):
    a = np.full(NM, np.nan); a[g.mi.values] = pd.to_numeric(g.값, errors='coerce').values; arr[(c, it)] = a
def wmean(c, it, wy):
    a = arr.get((c, it))
    if a is None: return np.nan
    s = (wy - 2000) * 12 - 1; e = s + 12               # (wy-1)년 12월 ~ wy년 11월
    if s < 0 or e > NM: return np.nan
    v = a[s:e]; return float(v.mean()) if np.isfinite(v).all() else np.nan
D0 = pd.read_parquet(os.path.join(V0, 'v0_D_final.parquet')); W0 = pd.read_parquet(os.path.join(V0, 'v0_W_final.parquet'))
ks = sorted(set(D0.ksic) | set(W0.ksic))
def code_of(k):
    c3 = 'C' + k[:3]
    return c3 if np.isfinite(wmean(c3, 'T10', 2010)) or (c3, 'T10') in arr and np.isfinite(arr[(c3, 'T10')][(2010 - 2000) * 12:(2010 - 2000) * 12 + 12]).any() else 'C' + k[:2]
def lr(a, b): return np.log(a / b) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else np.nan
def feat(k, t0):
    c = code_of(k); A = lambda it, y: wmean(c, it, y)
    return (lr((A('T10', t0) + A('T10', t0 - 1)) / 2, (A('T10', t0 - 2) + A('T10', t0 - 3)) / 2), lr(A('T10', t0), A('T10', t0 - 1)),
            lr((A('T11', t0) + A('T11', t0 - 1)) / 2, (A('T11', t0 - 2) + A('T11', t0 - 3)) / 2))
T0S = sorted(set(D0.t0) | set(W0.t0))
base = {(k, t0): feat(k, t0) for k in ks for t0 in T0S}
# 걸침 판정: 창 연도 wy 의 달 집합 [(wy-1)-12, wy-11]. 단절 달 B 이전·이후 달을 함께 쓰면 NaN
def straddle_set(break_ym):
    by, bm = break_ym // 100, break_ym % 100
    bi = (by - 2000) * 12 + bm - 1
    def mixed(wys):
        ms = [m for wy in wys for m in range((wy - 2000) * 12 - 1, (wy - 2000) * 12 + 11)]
        return min(ms) < bi <= max(ms)
    return mixed
def scen(rule):
    out = {}
    for (k, t0), (g2, g1, s2) in base.items():
        if rule == 'S1': mixed = straddle_set(201501); f2 = mixed([t0, t0 - 1, t0 - 2, t0 - 3]); f1 = mixed([t0, t0 - 1])
        elif rule == 'V1': mixed = straddle_set(201401); f2 = mixed([t0, t0 - 1, t0 - 2, t0 - 3]); f1 = mixed([t0, t0 - 1])
        elif rule == 'V2':   # 연 단위: 창 연도 라벨이 ≤2014 와 ≥2015 를 함께 쓰면
            f2 = (t0 - 3 <= 2014) and (t0 >= 2015); f1 = (t0 - 1 <= 2014) and (t0 >= 2015)
        else: f2 = f1 = False
        out[(k, t0)] = (np.nan if f2 else g2, np.nan if f1 else g1, np.nan if f2 else s2)
    return out
def apply(X, fe):
    X = X.copy(); v = np.array([fe[(k, t0)] for k, t0 in zip(X.ksic.values, X.t0.values)], dtype=float)
    for j, c in enumerate(['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']): X[c] = v[:, j].astype(X[c].dtype)
    return X
res = {}
chk = apply(D0, scen('S0'))
res['S0_rebuild_maxdiff'] = {c: float(np.nanmax(np.abs(chk[c].astype(float) - D0[c].astype(float)))) for c in ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']}
res['S0_rebuild_nan_agree'] = bool(all((chk[c].isna() == D0[c].isna()).all() for c in ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']))
for r in ['S1', 'V1', 'V2']:
    fl = scen(r); res[f'{r}_nan_t0'] = {c: sorted({int(t0) for (k, t0), v in fl.items() if np.isnan(v[j]) and np.isfinite(base[(k, t0)][j])}) for j, c in enumerate(['g2', 'g1', 'ship_g2'])}
# 학습(새로 작성): 동결 spec 3성분 순위평균
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8'))['cfg']['components']
def area(y, s):
    r = rankdata(s); n1 = int(y.sum()); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def fit_predict(D, W, seed):
    tr = D[D.t0.between(2005, 2018)]; te = D[D.t0 == 2021]; ww = W[W.t0.between(2005, 2018)].copy()
    cut = D[D.yb == 1].groupby('t0').y.min(); rk = []
    for comp in SPEC:
        f = comp['features']
        if comp['type'] == 'logit':
            pr = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=comp['C'], max_iter=3000)).fit(tr[f].values, tr.yb.values).predict_proba(te[f].values)[:, 1]
        else:
            prm = dict(comp['params'], seed=seed); Xa = np.vstack([tr[f].values, ww[f].values])
            if comp['type'] == 'xgbc':
                ya = np.r_[tr.yb.values, (ww.y >= ww.t0.map(cut)).astype(int).values]
                pr = xgb.train(prm, xgb.DMatrix(Xa, label=ya), comp['rounds']).predict(xgb.DMatrix(te[f].values))
            else:
                ya = np.r_[tr.kr_fut.values, ww.kr_fut.values]; dm = xgb.DMatrix(Xa, label=ya); bst = xgb.train(prm, dm, comp['rounds']); mt = bst.predict(dm)
                grid = np.exp(np.linspace(np.log(0.01), np.log(5), 60)); best = max(((nbinom.logpmf(ya, 1 / al, (1 / al) / (1 / al + mt)).sum(), al) for al in grid))[1]
                rr = 1 / best; tau = float(np.median([np.quantile(g.y.values, 0.9) for _, g in tr.groupby('t0')]))
                mu = bst.predict(xgb.DMatrix(te[f].values)); k = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9)
                pr = nbinom.sf(k - 1, rr, rr / (rr + mu))
        rk.append(rankdata(pr) / len(te))
    p = np.mean(rk, 0); pb = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values).predict_proba(te[['kr_mom']].values)[:, 1]
    a, b = area(te.yb.values, p), area(te.yb.values, pb); return float((a - b) / (1 - b))
for r in ['S1', 'V1', 'V2']:
    fl = scen(r); D, W = apply(D0, fl), apply(W0, fl)
    for sd in [0, 1]:
        res[f'{r}_seed{sd}'] = fit_predict(D, W, sd); print(r, sd, res[f'{r}_seed{sd}'], round(time.time() - t), flush=True)
res['sec'] = round(time.time() - t)
json.dump(res, open(os.path.join(OUTD, 'vB_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in res.items(): print(k, v)
