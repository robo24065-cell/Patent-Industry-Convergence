# -*- coding: utf-8 -*-
"""[⚠ 2단계 감사로 대체됨: final_run.py / final_spec.json 사용. 이 파일의 v37은 스냅샷 패밀리 누수 포함 — 홀드아웃 사용 금지]
T4 최종 후보(v37 재현 파이프라인) — 기준시점 C 로 매개화.

사용(각 호출은 9분 안에 끝나도록 구성요소별로 나눔):
  python final_model.py fit  cls   --C 2014-01-01      # XGB 3년 분류형 (v30 구성)
  python final_model.py fit  haz   --C 2014-01-01      # XGB 이산시간 위험모형 (v25 구성)
  python final_model.py fit  logit --C 2014-01-01      # 로지스틱 (v27 구성)
  python final_model.py fit  hgb   --C 2014-01-01      # sklearn HGB 분류형 (v36 구성)
  python final_model.py predict all --C 2014-01-01 --L 2014-01-01 2014-07-01 ...   # 기준일별 예측 저장
  python final_model.py score --C 2014-01-01 --L ...  # 4개 로짓 동일가중 앙상블 + (허용 시) 지표

🔒 홀드아웃(C=2019-01-01, L=2019-01-01..2021-07-01)은 3단계 평가자만:  set T4_STAGE3_HOLDOUT=1
"""
import os, sys, json, time, argparse, pickle
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd, xgboost as xgb
import t4lib as T, hazard as HZ, soc as SO
from run import train_landmarks, labels, metrics, get_frame
from versions import (add_ratio, add_fam, F23N, F6C, H12N, DEC, DRIFT, FAM, lm_w, recency_w, xgb_fit)

HERE = T.HERE
MD = os.path.join(HERE, 'final_models')
os.makedirs(MD, exist_ok=True)
XGBP = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', nthread=2, max_depth=6, eta=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20, max_cat_to_onehot=8, reg_lambda=1.0, seed=0)
CLS_N = F23N + ['ak_rate', 'ak_n']
HAZ_N = H12N + SO.SOC + FAM
LOG_N = F23N
LOG1P = ['claims', 'n_renew', 'port_ksic_n', 'n_inv', 'n_ipc', 'n_cpc', 'title_len', 'sib_n', 'aks_n', 'aipc_n', 'inv_n', 'a_lap6m', 'fam_n', 'n_transfer', 'n_lic']


def tag(C): return C.replace('-', '')


def peak_mem_gb():
    try:
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32'); ps = ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e9
    except Exception:
        return float('nan')


def enrich(d, kcol=None):
    d = add_fam(SO.add_soc(add_ratio(d)))
    if kcol: d = SO.add_ak(d, kcol)
    return d


def cats_of(d):
    return {c: sorted(d[c].astype(str).unique()) for c in F6C}


def xmat(d, num, cats):
    X = d[num].astype('float32').copy()
    for c in F6C: X[c] = pd.Categorical(d[c].astype(str), categories=cats[c])
    return X


def hazard_train_Ls(C):
    y = int(C[:4])
    return [f'{a}-01-01' for a in range(2005, y - 1)]


KEEP = set(CLS_N + HAZ_N + LOG_N + F6C + ['y', 'L', 'pid', 'acd', 'k'])


def f32(d):
    d = d[[c for c in d.columns if c in KEEP]]
    fc = d.select_dtypes('float64').columns
    d = d.astype({c: 'float32' for c in fc})
    oc = d.select_dtypes('object').columns
    d = d.astype({c: 'category' for c in oc})
    return d


# 계산예산 규칙(사전 고정): 학습행이 상한을 넘으면 기준일마다 같은 비율로 무작위 추출(seed 0).
# 검증(C=2014)은 분류 757,390행·위험 1,186,537행으로 상한 미만이라 추출이 일어나지 않는다(검증 수치 = 전량).
CAP_CLS, CAP_HAZ = 1_200_000, 1_200_000


def recat(d):
    oc = d.select_dtypes('object').columns
    return d.astype({c: 'category' for c in oc})


def cls_frame(C, kcol):
    """기준일별로 피처->레이블->보강->float32 후 결합(메모리 절약)"""
    Ls = train_landmarks(C)
    tot = sum(len(T.features(L)) for L in Ls); frac = min(1.0, CAP_CLS / tot)
    parts = []
    for L in Ls:
        f = get_frame([L], C)
        if frac < 1: f = f.sample(frac=frac, random_state=0)
        parts.append(f32(enrich(f, kcol)))
    print(f'cls rows total {tot:,} frac {frac:.3f}', flush=True)
    return recat(pd.concat(parts, ignore_index=True))


def haz_frame(C):
    Ls = hazard_train_Ls(C)
    tot = sum(len(HZ.train_decisions(L, C)) for L in Ls); frac = min(1.0, CAP_HAZ / tot)
    print(f'haz rows total {tot:,} frac {frac:.3f}', flush=True)
    parts = []
    for L in Ls:
        f = HZ.train_decisions(L, C)
        parts.append(f32(enrich(f.sample(frac=frac, random_state=0) if frac < 1 else f)))
    return recat(pd.concat(parts, ignore_index=True))


# ------------------------------------------------------------------ fit
def fit(comp, C):
    t = time.time()
    if comp in ('cls', 'hgb', 'logit'):
        tr = cls_frame(C, 'next_annl' if comp in ('cls', 'hgb') else None)
        w = lm_w(tr)
    if comp == 'cls':
        cats = cats_of(tr)
        yv = tr.y.values.copy(); X = xmat(tr, CLS_N, cats)
        del tr; import gc; gc.collect()
        dm = xgb.DMatrix(X, yv, weight=w, enable_categorical=True)
        del X; gc.collect()
        b = xgb.train(XGBP, dm, 600)
        b.save_model(os.path.join(MD, f'cls_{tag(C)}.json'))
        pickle.dump(cats, open(os.path.join(MD, f'cls_{tag(C)}_cats.pkl'), 'wb'))
    elif comp == 'haz':
        tr = haz_frame(C)
        cats = cats_of(tr)
        yv = tr.y.values.copy(); wv = recency_w(tr); X = xmat(tr, HAZ_N, cats)
        del tr; import gc; gc.collect()
        dm = xgb.DMatrix(X, yv, weight=wv, enable_categorical=True)
        del X; gc.collect()
        b = xgb.train(XGBP, dm, 600)
        b.save_model(os.path.join(MD, f'haz_{tag(C)}.json'))
        pickle.dump(cats, open(os.path.join(MD, f'haz_{tag(C)}_cats.pkl'), 'wb'))
    elif comp == 'logit':
        from sklearn.linear_model import LogisticRegression
        prep = LogPrep().fit(tr)
        X = prep.transform(tr); yv = tr.y.values.copy()
        del tr; import gc; gc.collect()
        m = LogisticRegression(C=0.5, max_iter=3000)
        m.fit(X, yv, sample_weight=w)
        pickle.dump(dict(m=m, prep=prep), open(os.path.join(MD, f'logit_{tag(C)}.pkl'), 'wb'))
    elif comp == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1, max_categories=60)
        ctr = oe.fit_transform(tr[F6C].astype(str)).astype('float32'); ctr[ctr < 0] = np.nan
        X = np.c_[tr[CLS_N].to_numpy(dtype='float32'), ctr]; yv = tr.y.values.copy()
        del tr, ctr; import gc; gc.collect()
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0,
                                           categorical_features=list(range(len(CLS_N), len(CLS_N) + len(F6C))), early_stopping=False, random_state=0)
        m.fit(X, yv, sample_weight=w)
        pickle.dump(dict(m=m, oe=oe), open(os.path.join(MD, f'hgb_{tag(C)}.pkl'), 'wb'))
    pk = peak_mem_gb()
    print(f'fit {comp} C={C} done {time.time() - t:.1f}s  peak_mem={pk:.2f}GB', flush=True)


class LogPrep:
    """v27 전처리의 메모리 절약판(float32): log1p(개수형) -> 결측지시자(결측>1%) -> 0.5/99.5% 윈저화 -> 중앙값 대치 -> 표준화 + 원핫(min_freq 50)"""
    def fit(self, d):
        self.na_cols = [c for c in LOG_N if d[c].isna().mean() > 0.01]
        X = self._num(d)
        p = X.shape[1]
        self.lo = np.empty(p, 'float32'); self.hi = np.empty(p, 'float32'); self.med = np.empty(p, 'float32')
        self.mu = np.empty(p, 'float32'); self.sd = np.empty(p, 'float32')
        for j in range(p):       # 열 단위 계산(임시 메모리 상한)
            x = X[:, j]
            lo, hi = np.nanquantile(x, [0.005, 0.995]) if np.isfinite(x).any() else (0.0, 0.0)
            np.clip(x, lo, hi, out=x)
            md = np.nanmedian(x) if np.isfinite(x).any() else 0.0
            x[np.isnan(x)] = md
            self.lo[j], self.hi[j], self.med[j] = lo, hi, md
            self.mu[j] = x.mean(dtype='float64'); s_ = x.std(dtype='float64'); self.sd[j] = s_ if s_ > 0 else 1.0
        del X
        self.cats = {}
        for c in F6C:
            vc = d[c].astype(str).value_counts()
            self.cats[c] = (list(vc[vc >= 50].index), set(vc[vc < 50].index))
        return self

    def _num(self, d):
        X = d[LOG_N].to_numpy(dtype='float32', na_value=np.nan).copy()
        for i, c in enumerate(LOG_N):
            if c in LOG1P: X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
        na = np.column_stack([np.isnan(d[c].to_numpy(dtype='float32', na_value=np.nan)) for c in self.na_cols]).astype('float32')
        return np.hstack([X, na])

    def transform(self, d, chunk=200_000):
        """행 청크별로 계산해 미리 잡은 float32 배열에 채운다(임시 메모리 상한)"""
        parts = None; out = None; r0 = 0
        for a in range(0, len(d), chunk):
            blk = self._transform(d.iloc[a:a + chunk])
            if out is None:
                out = np.empty((len(d), blk.shape[1]), dtype='float32')
            out[a:a + len(blk)] = blk
        return out

    def _transform(self, d):
        X = self._num(d)
        np.clip(X, self.lo, self.hi, out=X)
        m = np.isnan(X)
        X[m] = np.broadcast_to(self.med, X.shape)[m]
        X -= self.mu; X /= self.sd
        oh = []
        for c in F6C:
            keep, infreq = self.cats[c]
            v = d[c].astype(str).values
            M = np.zeros((len(d), len(keep) + int(len(infreq) > 0)), dtype='float32')
            pos = {k: i for i, k in enumerate(keep)}
            idx = np.array([pos.get(x, -1) for x in v])
            ok = idx >= 0
            M[np.where(ok)[0], idx[ok]] = 1
            if infreq:
                inf = np.array([x in infreq for x in v])
                M[inf, -1] = 1   # 학습 때 드문 범주 = infrequent 열, 미지 범주 = 전부 0 (sklearn handle_unknown='ignore'와 동일)
            oh.append(M)
        return np.hstack([X.astype('float32')] + oh)


def log_prep(d):
    X = d[LOG_N].astype(float).copy()
    for c in LOG1P:
        if c in X: X[c] = np.log1p(X[c].clip(lower=0))
    for c in LOG_N:
        X[c + '_na'] = X[c].isna().astype(float)
    return X


# ------------------------------------------------------------------ predict
def predict(comp, C, L):
    f = T.features(L)
    f = f.assign(acd=T.load()['pt'].acd.values[f.pid.values])
    out = pd.DataFrame({'pid': f.pid.values, 'L': L, 'ksic': f.ksic.values})
    if comp == 'cls':
        b = xgb.Booster(); b.load_model(os.path.join(MD, f'cls_{tag(C)}.json'))
        cats = pickle.load(open(os.path.join(MD, f'cls_{tag(C)}_cats.pkl'), 'rb'))
        d = enrich(f, 'next_annl')
        out['p'] = b.predict(xgb.DMatrix(xmat(d, CLS_N, cats), enable_categorical=True))
    elif comp == 'haz':
        b = xgb.Booster(); b.load_model(os.path.join(MD, f'haz_{tag(C)}.json'))
        cats = pickle.load(open(os.path.join(MD, f'haz_{tag(C)}_cats.pkl'), 'rb'))
        dec = HZ.eval_decisions(f.reset_index(drop=True))
        dec = enrich(dec)
        h = b.predict(xgb.DMatrix(xmat(dec, HAZ_N, cats), enable_categorical=True))
        out['p'] = HZ.combine(dec, h, len(f))
    elif comp == 'logit':
        o = pickle.load(open(os.path.join(MD, f'logit_{tag(C)}.pkl'), 'rb'))
        d = enrich(f)
        out['p'] = o['m'].predict_proba(o['prep'].transform(d))[:, 1]
    elif comp == 'hgb':
        o = pickle.load(open(os.path.join(MD, f'hgb_{tag(C)}.pkl'), 'rb'))
        d = enrich(f, 'next_annl')
        c = o['oe'].transform(d[F6C].astype(str)).astype('float32'); c[c < 0] = np.nan
        out['p'] = o['m'].predict_proba(np.c_[d[CLS_N].to_numpy(dtype='float32', na_value=np.nan), c])[:, 1]
    fn = os.path.join(MD, f'pred_{comp}_{tag(C)}_{tag(L)}.parquet')
    out.to_parquet(fn, index=False)
    return out


def score(C, Ls, with_labels=True):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    rows = []
    for L in Ls:
        ps = [pd.read_parquet(os.path.join(MD, f'pred_{c}_{tag(C)}_{tag(L)}.parquet')) for c in ('cls', 'haz', 'logit', 'hgb')]
        z = sum(0.25 * lg(p.p.values) for p in ps)
        d = ps[0][['pid', 'L', 'ksic']].copy()
        for c, p in zip(('cls', 'haz', 'logit', 'hgb'), ps): d['p_' + c] = p.p.values
        d['p'] = 1 / (1 + np.exp(-z))
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(os.path.join(MD, f'ens_{tag(C)}_{tag(Ls[0])}_{tag(Ls[-1])}.parquet'), index=False)
    res = {}
    if with_labels:
        y = np.concatenate([labels(L, d[d.L == L].pid.values, None) for L in Ls])
        d['y'] = y
        res = {k: metrics(y, d[k].values) for k in ['p', 'p_cls', 'p_haz', 'p_logit', 'p_hgb']}
        res['per_L'] = {L: metrics(d.y[d.L == L].values, d.p[d.L == L].values)['auc'] for L in Ls}
        print(json.dumps(res, ensure_ascii=False, indent=1))
        json.dump(res, open(os.path.join(MD, f'score_{tag(C)}.json'), 'w'), ensure_ascii=False, indent=1)
    return d, res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('comp', nargs='?', default='all')
    ap.add_argument('--C', required=True); ap.add_argument('--L', nargs='*', default=[])
    ap.add_argument('--nolabel', action='store_true')
    a = ap.parse_args()
    comps = ['cls', 'haz', 'logit', 'hgb'] if a.comp == 'all' else [a.comp]
    if a.cmd == 'fit':
        for c in comps: fit(c, a.C)
    elif a.cmd == 'predict':
        for c in comps:
            for L in a.L:
                t = time.time(); predict(c, a.C, L); print('pred', c, L, f'{time.time() - t:.1f}s', flush=True)
    elif a.cmd == 'score':
        score(a.C, a.L, with_labels=not a.nolabel)
