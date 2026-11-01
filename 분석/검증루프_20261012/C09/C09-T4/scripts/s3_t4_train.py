# C09-T4 s3: T4 R2-E03 동결 spec 구성요소 1개 학습 → 평가행 예측 저장(이 폴더 cache/t4pred 에만 씀).
#  r2lib.fit_predict_xgb_bag / fit_predict_sk / LogPrep / _ChunkIter / _prep_cat / recency_w / hazard.combine 을 그대로 옮겼다(읽기 전용 원본과 같은 계산).
#  달라지는 점: rule ∈ {orig, R1, R2, R3} 일 때 지연납부 5개 열을 s2 가 만든 npy 로 바꾼다(orig 는 프레임 값 그대로 = 재현 점검).
#  python s3_t4_train.py <O> <rule> <comp> <bag|->      comp ∈ cls, haz, logit, hgb
import os, sys, json, time, pickle, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
from s2_t4_feats import LATE, fdir, xdir
SPEC = json.load(open(os.path.join(T4D, 'R2', 'final_spec_r2.json'), encoding='utf-8'))['spec']
CAT = json.load(open(os.path.join(T4D, 'final_spec.json'), encoding='utf-8'))['categorical']


def pdir(o, rule):
    p = os.path.join(C, 't4pred', o, rule); os.makedirs(p, exist_ok=True); return p


def read(o, frame, cols, rule):
    import pyarrow.parquet as pq
    fn = os.path.join(fdir(o), frame + '.parquet')
    have = set(pq.ParquetFile(fn).schema.names)
    d = pq.read_table(fn, columns=[c for c in cols if c in have]).to_pandas()
    if rule != 'orig':
        for c in LATE:
            if c in d.columns:
                v = np.load(os.path.join(xdir(o), f'{frame}__{c}__{rule}.npy'))
                assert len(v) == len(d)
                d[c] = v
    return d


def recency_w(Lvals, spec):
    tr = spec['train']
    yr = pd.to_datetime(pd.Series(np.asarray(Lvals).astype(str))).dt.year.values
    return 1.0 + (yr - tr['weight_base_year']) / tr['weight_div']


def _ChunkIter(d, cols, y, w, chunk=150_000):
    import xgboost as xgb

    class It(xgb.DataIter):
        def __init__(self):
            self.i = 0
            super().__init__(cache_prefix=None)

        def next(self, input_data):
            if self.i >= len(d):
                return False
            a, b = self.i, min(self.i + chunk, len(d))
            input_data(data=d.iloc[a:b][cols], label=y[a:b], weight=w[a:b])
            self.i = b
            return True

        def reset(self):
            self.i = 0
    return It()


def _prep_cat(d, cmap=None):
    for c in CAT:
        d[c] = d[c].astype(str)
    if cmap is None:
        cmap = {c: sorted(d[c].unique()) for c in CAT}
    for c in CAT:
        d[c] = pd.Categorical(d[c], categories=cmap[c])
    return d, cmap


def fit_xgb(o, rule, comp, k, save_model=False):
    import xgboost as xgb
    fn = os.path.join(pdir(o, rule), f'{comp}_b{k}.npy')
    if os.path.exists(fn): return
    cfg = SPEC['components'][comp]
    t = time.time()
    frame = 'frame_cls' if comp == 'cls' else 'frame_haz'
    base_cols = cfg['num'] + CAT + ['y', 'L', 'pid', 'ksic', 'acd', 'atype', 'k', 'next_annl', 'age']
    d = read(o, frame, list(dict.fromkeys(base_cols)), rule)
    y = d.y.values.astype('float32'); w = recency_w(d.L.values, SPEC).astype('float32')
    X = d[cfg['num'] + CAT].copy(); del d; gc.collect()
    for c in cfg['num']:
        if X[c].dtype != np.float32: X[c] = X[c].astype('float32')
    X, cmap = _prep_cat(X)
    num = cfg['num'] if k == 0 else list(np.random.default_rng(k).permutation(cfg['num']))
    dm = xgb.QuantileDMatrix(_ChunkIter(X, num + CAT, y, w), enable_categorical=True, max_bin=256); gc.collect()
    P = dict(cfg['params']); P['seed'] = k
    b = xgb.train(P, dm, cfg['rounds']); del dm, X; gc.collect()
    tfit = time.time() - t
    if save_model:
        b.save_model(os.path.join(pdir(o, rule), f'{comp}_b{k}.json'))
        pickle.dump(dict(cmap=cmap, num=num), open(os.path.join(pdir(o, rule), f'{comp}_b{k}.pkl'), 'wb'))
    if k == 0:
        g = b.get_score(importance_type='total_gain')
        json.dump(g, open(os.path.join(pdir(o, rule), f'{comp}_b0_gain.json'), 'w'), indent=0)
    evf = 'ev' if comp == 'cls' else 'ev_haz'
    e = read(o, evf, list(dict.fromkeys(cfg['num'] + CAT + ['pid', 'L', 'ksic', 'acd', 'atype', 'row', 'k', 'next_annl', 'age'])), rule)
    Xe = e[num + CAT].copy()
    for c in num:
        if Xe[c].dtype != np.float32: Xe[c] = Xe[c].astype('float32')
    Xe, _ = _prep_cat(Xe, cmap)
    p = b.predict(xgb.DMatrix(Xe, enable_categorical=True))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    np.save(fn, np.log(p / (1 - p)).astype('float32'))
    print(f'{o} {rule} {comp} bag{k} fit {tfit:.0f}s total {time.time() - t:.0f}s', flush=True)


class LogPrep:
    def __init__(self, num, log1p):
        self.num, self.log1p = num, set(log1p)

    def fit(self, d):
        self.na_cols = [c for c in self.num if d[c].isna().mean() > 0.01]
        X = self._num(d); p = X.shape[1]
        self.lo, self.hi, self.med, self.mu, self.sd = [np.empty(p, 'float32') for _ in range(5)]
        for j in range(p):
            x = X[:, j]
            lo, hi = np.nanquantile(x, [0.005, 0.995]) if np.isfinite(x).any() else (0.0, 0.0)
            np.clip(x, lo, hi, out=x)
            md = np.nanmedian(x) if np.isfinite(x).any() else 0.0
            x[np.isnan(x)] = md
            self.lo[j], self.hi[j], self.med[j] = lo, hi, md
            self.mu[j] = x.mean(dtype='float64'); s_ = x.std(dtype='float64'); self.sd[j] = s_ if s_ > 0 else 1.0
        del X
        self.cats = {}
        for c in CAT:
            vc = d[c].astype(str).value_counts()
            self.cats[c] = (list(vc[vc >= 50].index), set(vc[vc < 50].index))
        return self

    def _num(self, d):
        X = d[self.num].to_numpy(dtype='float32', na_value=np.nan).copy()
        for i, c in enumerate(self.num):
            if c in self.log1p: X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
        na = np.column_stack([np.isnan(d[c].to_numpy(dtype='float32', na_value=np.nan)) for c in self.na_cols]).astype('float32') if self.na_cols else np.zeros((len(d), 0), 'float32')
        return np.hstack([X, na])

    def transform(self, d, chunk=200_000):
        out = None
        for a in range(0, len(d), chunk):
            blk = self._tr(d.iloc[a:a + chunk])
            if out is None: out = np.empty((len(d), blk.shape[1]), dtype='float32')
            out[a:a + len(blk)] = blk
        return out

    def _tr(self, d):
        X = self._num(d)
        np.clip(X, self.lo, self.hi, out=X)
        m = np.isnan(X); X[m] = np.broadcast_to(self.med, X.shape)[m]
        X -= self.mu; X /= self.sd
        oh = []
        for c in CAT:
            keep, infreq = self.cats[c]
            v = d[c].astype(str).values
            M = np.zeros((len(d), len(keep) + int(len(infreq) > 0)), dtype='float32')
            pos = {k: i for i, k in enumerate(keep)}
            idx = np.array([pos.get(x, -1) for x in v]); ok = idx >= 0
            M[np.where(ok)[0], idx[ok]] = 1
            if infreq:
                M[np.array([x in infreq for x in v]), -1] = 1
            oh.append(M)
        return np.hstack([X] + oh)


def fit_sk(o, rule, comp, save_model=False):
    fn = os.path.join(pdir(o, rule), f'{comp}.npy')
    if os.path.exists(fn): return
    t = time.time()
    cfg = SPEC['components'][comp]
    cols = list(dict.fromkeys(cfg['num'] + CAT + ['y', 'L', 'pid', 'ksic', 'acd', 'atype', 'next_annl', 'age']))
    d = read(o, 'frame_cls', cols, rule)
    if len(d) > cfg.get('max_rows', 10 ** 9):
        d = d.sample(n=cfg['max_rows'], random_state=0)
    e = read(o, 'ev', [c for c in cols if c != 'y'], rule)
    if comp == 'logit':
        from sklearn.linear_model import LogisticRegression
        prep = LogPrep(cfg['num'], cfg['log1p']).fit(d)
        X = prep.transform(d); y = d.y.values; w = recency_w(d.L.values, SPEC); del d; gc.collect()
        m = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter']).fit(X, y, sample_weight=w)
        del X; gc.collect()
        if save_model: pickle.dump(dict(m=m, prep=prep), open(os.path.join(pdir(o, rule), f'{comp}.pkl'), 'wb'))
        p = m.predict_proba(prep.transform(e))[:, 1]
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1, max_categories=60)
        c = oe.fit_transform(d[CAT].astype(str)).astype('float32'); c[c < 0] = np.nan
        X = np.c_[d[cfg['num']].to_numpy(dtype='float32', na_value=np.nan), c]; y = d.y.values; w = recency_w(d.L.values, SPEC)
        allnan = np.where(np.isnan(X[:, :len(cfg['num'])]).all(0))[0]
        X[:, allnan] = 0
        del d, c; gc.collect()
        m = HistGradientBoostingClassifier(**cfg['params'], categorical_features=list(range(len(cfg['num']), len(cfg['num']) + len(CAT))),
                                           early_stopping=False, random_state=0).fit(X, y, sample_weight=w)
        del X; gc.collect()
        if save_model: pickle.dump(dict(m=m, oe=oe, allnan=allnan), open(os.path.join(pdir(o, rule), f'{comp}.pkl'), 'wb'))
        ce = oe.transform(e[CAT].astype(str)).astype('float32'); ce[ce < 0] = np.nan
        Xe = np.c_[e[cfg['num']].to_numpy(dtype='float32', na_value=np.nan), ce]; Xe[:, allnan] = 0
        p = m.predict_proba(Xe)[:, 1]
    np.save(fn, p.astype('float64'))
    print(f'{o} {rule} {comp} {time.time() - t:.0f}s', flush=True)


def combine_haz(o, z):
    """hazard.combine: P = 1 - Π(1-h)"""
    import pyarrow.parquet as pq
    row = pq.read_table(os.path.join(fdir(o), 'ev_haz.parquet'), columns=['row']).to_pandas().row.values
    n = pq.ParquetFile(os.path.join(fdir(o), 'ev.parquet')).metadata.num_rows
    h = 1 / (1 + np.exp(-z))
    lg = pd.Series(np.log1p(-np.clip(h, 1e-6, 1 - 1e-6))).groupby(row).sum()
    P = np.zeros(n); P[lg.index.values] = 1 - np.exp(lg.values)
    return P


if __name__ == '__main__':
    o, rule, comp, k = sys.argv[1:5]
    save = (o == 'F22')
    if comp in ('cls', 'haz'):
        fit_xgb(o, rule, comp, int(k), save_model=save)
    else:
        fit_sk(o, rule, comp, save_model=save)
