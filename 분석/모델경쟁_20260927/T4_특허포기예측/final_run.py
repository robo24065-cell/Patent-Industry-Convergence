# -*- coding: utf-8 -*-
"""T4 특허 포기(등록료불납) 예측 — 동결 파이프라인 (2단계 감사·개선 후 고정).
모든 피처·하이퍼파라미터·학습범위는 final_spec.json 에서 읽는다(이 파일과 함께 동결).

사용:
  python final_run.py --mode dryrun     # 검증 설정(C=2014-01-01, L=2014-01..2015-07)으로 학습·평가 → final_models/run_dryrun/dryrun_result.json
  python final_run.py --mode holdout    # 🔒 3단계 평가자 전용. C=2019-01-01 이전 관측 레이블만으로 학습 → L=2019-01..2021-07 예측 → holdout_result.json
  python final_run.py --mode future     # 3단계 평가자 전용(홀드아웃 뒤). 전체 가용자료로 재학습 → future_forecast.csv
옵션: --step <이름> 한 단계만 실행(기본 all: 단계마다 별도 프로세스로 순차 실행, 각 단계 9분 이내·4GB 이하 목표)
      --force 이미 끝난 단계도 다시 실행.  --boot B 부트스트랩 반복수(기본 spec).
단계: feats → frame_cls → frame_haz → fit_cls → fit_haz → fit_logit → fit_hgb → fit_base → pred → score
홀드아웃 보호: dryrun 에서는 환경변수 T4_STAGE3_HOLDOUT 를 켜지 않는다 → t4lib 가 E>=2019 소멸을 메모리에 올리지 않고,
              L>=2019 피처·레이블 계산을 예외로 막는다. holdout/future 모드만 이 스크립트가 스스로 켠다.
"""
import os, sys, json, time, argparse, subprocess, pickle, gc

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['feats', 'frame_cls', 'frame_haz', 'fit_cls', 'fit_haz', 'fit_logit', 'fit_hgb', 'fit_base', 'pred', 'score']


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dryrun', 'holdout', 'future'])
    ap.add_argument('--step', default='all')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--boot', type=int, default=None)
    ap.add_argument('--spec', default='final_spec.json', help='실험용. 동결 실행은 기본값')
    ap.add_argument('--tag', default=None, help='실험용 출력 폴더 이름. 기본 run_<mode>')
    return ap.parse_args()


try:
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
A = parse()
SPEC = json.load(open(os.path.join(HERE, A.spec), encoding='utf-8'))
if A.mode in ('holdout', 'future'):
    os.environ['T4_STAGE3_HOLDOUT'] = '1'          # 3단계 전용: 전체 소멸 이벤트 적재 허용
else:
    os.environ.pop('T4_STAGE3_HOLDOUT', None)
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, HERE)

MODE = A.mode
C = SPEC['C'][MODE]
OUT = os.path.join(HERE, 'final_models', f'run_{A.tag or MODE}')
os.makedirs(OUT, exist_ok=True)
DONE = os.path.join(OUT, '_done.json')


def semi(a, b):
    out, y, m = [], int(a[:4]), int(a[5:7])
    while f'{y:04d}-{m:02d}-01' <= b:
        out.append(f'{y:04d}-{m:02d}-01')
        m += 6
        if m > 12: m -= 12; y += 1
    return out


EVAL_L = semi(*SPEC['eval_L'][MODE])


# ============================================================== 드라이버
def done_state():
    return json.load(open(DONE, encoding='utf-8')) if os.path.exists(DONE) else {}


def mark_done(step, info=None):
    s = done_state(); s[step] = dict(t=time.strftime('%Y-%m-%d %H:%M:%S'), C=C, **(info or {}))
    json.dump(s, open(DONE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def driver():
    for st in STEPS:
        if st in done_state() and not A.force:
            print(f'[skip] {st} (완료됨)', flush=True); continue
        while True:
            cmd = [sys.executable, os.path.abspath(__file__), '--mode', MODE, '--step', st, '--spec', A.spec] + (['--boot', str(A.boot)] if A.boot else []) + (['--tag', A.tag] if A.tag else [])
            t = time.time()
            r = subprocess.run(cmd)
            print(f'[{st}] exit={r.returncode} {time.time() - t:.0f}s', flush=True)
            if r.returncode == 3:      # 시간예산 때문에 나눠서 계속
                continue
            if r.returncode != 0:
                sys.exit(f'단계 {st} 실패')
            break


# ============================================================== 공통
import numpy as np, pandas as pd
import t4lib as T, soc as SO, hazard as HZ, feat2 as F2
from run import train_landmarks, labels, get_frame
from versions import add_ratio

CP = SPEC['components']
CAT = SPEC['categorical']
T0 = time.time()
BUDGET = 480   # 초: 이 이상이면 다음 호출로 넘김(exit 3)


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
        return round(c.PeakWorkingSetSize / 1e9, 2)
    except Exception:
        return float('nan')


def cls_train_Ls():
    return train_landmarks(C, SPEC['train']['first_landmark'])


def haz_train_Ls():
    y = int(C[:4])
    return [f'{a}-01-01' for a in range(int(SPEC['train']['first_landmark'][:4]), y - 1)]


def enrich(d, ak=True):
    """피처 보강 — 모두 L 이전 정보. spec 에 따라 동일하게 적용"""
    d = add_ratio(d)
    d = SO.add_soc(d)
    d = F2.add_fam_dated(d)            # [감사 수정] 시점 보정 패밀리(추정 공개연도 <= L연도-1)
    d = F2.fix_dspst(d)                # [감사 수정] 등록 후 사건 스냅샷 제거
    if ak: d = SO.add_ak(d, 'next_annl')
    d = F2.add_macro(d)
    d = F2.add_agent(d)
    d = F2.add_owner(d)
    return d


def all_cols():
    s = set()
    for c in CP.values(): s |= set(c['num'])
    s |= set(CAT) | set(SPEC['baseline']['num']) | set(SPEC['baseline']['cat'])
    return s


def slim(d):
    keep = all_cols() | {'y', 'L', 'pid', 'acd', 'k', 'd', 'row'}
    d = d[[c for c in d.columns if c in keep]]
    fc = d.select_dtypes('float64').columns
    d = d.astype({c: 'float32' for c in fc})
    return d


def to_str(d):
    return d.astype({c: str for c in d.select_dtypes(['category', 'object']).columns if c != 'L'})


# ============================================================== 단계: feats
def step_feats():
    Ls = sorted(set(cls_train_Ls() + haz_train_Ls() + EVAL_L))
    for L in Ls:
        T.features(L); SO.soc_features(L); F2.agent_feats(L); F2.owner_feats(L)
        if time.time() - T0 > BUDGET:
            print('feats: 시간예산 도달, 다음 호출에서 계속', L, flush=True); sys.exit(3)
    for L in haz_train_Ls():
        HZ.train_decisions(L, C)
        if time.time() - T0 > BUDGET: sys.exit(3)
    mark_done('feats', dict(n_L=len(Ls)))


# ============================================================== 단계: 학습 프레임
def step_frame_cls():
    Ls = cls_train_Ls()
    tot = sum(len(T.features(L)) for L in Ls)
    cap = SPEC['train']['cap_rows_cls']
    frac = min(1.0, cap / tot)
    parts = []
    for L in Ls:
        f = get_frame([L], C)
        if frac < 1: f = f.sample(frac=frac, random_state=0)
        parts.append(to_str(slim(enrich(f, ak=True))))
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(os.path.join(OUT, 'frame_cls.parquet'), index=False)
    # 기준선은 추출 없이 전량(열 5개만)
    parts = []
    for L in Ls:
        f = get_frame([L], C)
        parts.append(f[['pid', 'L', 'y'] + SPEC['baseline']['num'] + SPEC['baseline']['cat']])
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(OUT, 'frame_base.parquet'), index=False)
    print(f'frame_cls rows {len(d):,} (전체 {tot:,}, frac {frac:.3f}) Ls {Ls[0]}..{Ls[-1]} ({len(Ls)}) pos {d.y.mean():.4f} peak {peak_mem_gb()}GB', flush=True)
    mark_done('frame_cls', dict(rows=len(d), total=tot, frac=round(frac, 4), Ls=[Ls[0], Ls[-1], len(Ls)], pos_rate=round(float(d.y.mean()), 4)))


def step_frame_haz():
    Ls = haz_train_Ls()
    tot = sum(len(HZ.train_decisions(L, C)) for L in Ls)
    frac = min(1.0, SPEC['train']['cap_rows_haz'] / tot)
    parts = []
    for L in Ls:
        f = HZ.train_decisions(L, C)
        if frac < 1: f = f.sample(frac=frac, random_state=0)
        parts.append(to_str(slim(enrich(f, ak=False))))
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(os.path.join(OUT, 'frame_haz.parquet'), index=False)
    print(f'frame_haz rows {len(d):,} (전체 {tot:,}, frac {frac:.3f}) pos {d.y.mean():.4f} peak {peak_mem_gb()}GB', flush=True)
    mark_done('frame_haz', dict(rows=len(d), total=tot, frac=round(frac, 4), Ls=[Ls[0], Ls[-1], len(Ls)], pos_rate=round(float(d.y.mean()), 4)))


def recency_w(L):
    return (1.0 + (pd.to_datetime(pd.Series(L)).dt.year.values - SPEC['train']['weight_base_year']) / SPEC['train']['weight_div'])


def catmap_of(d):
    return {c: sorted(d[c].astype(str).unique()) for c in CAT}


def xmat(d, num, cmap):
    X = d[num].astype('float32').copy()
    for c in CAT: X[c] = pd.Categorical(d[c].astype(str), categories=cmap[c])
    return X


# ============================================================== 단계: 학습
def bag_num(comp, k):
    """배깅 구성원 k 의 열 순서: k=0 은 spec 순서, k>0 은 seed k 로 섞음(열 표집 무작위성 분산 감소 목적)"""
    num = CP[comp]['num']
    return num if k == 0 else list(np.random.default_rng(k).permutation(num))


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


def fit_xgb(comp, frame):
    """XGB 배깅: 구성원 k=0..n_bag-1 (seed=k, 열 순서 섞기). 구성원마다 저장하고 시간예산을 넘기면 exit 3 으로 이어서"""
    import xgboost as xgb
    cfg = CP[comp]
    nb = cfg.get('n_bag', 1)
    d = None
    for k in range(nb):
        fn = os.path.join(OUT, f'{comp}_{k}.json')
        if os.path.exists(fn) and not A.force:
            continue
        if k > 0 and time.time() - T0 > BUDGET - 60 - cfg.get('sec_hint', 300):
            print(f'{comp}: 시간예산 — 구성원 {k}부터 다음 호출', flush=True); sys.exit(3)
        tk = time.time()
        if d is None:
            import pyarrow.parquet as pq
            d = pq.read_table(os.path.join(OUT, frame), columns=cfg['num'] + CAT + ['y', 'L']).to_pandas(strings_to_categorical=True, self_destruct=True)
            y = d.y.values.astype('float32'); w = recency_w(np.asarray(d.L.astype(str)))
            d = d.drop(columns=['y', 'L'])
            for c in CAT:
                d[c] = d[c].astype(str).astype('category') if d[c].dtype.name != 'category' else d[c]
            cmap = {c: sorted(str(x) for x in d[c].cat.categories) for c in CAT}   # = catmap_of (문자열 정렬 목록)
            for c in CAT:
                d[c] = d[c].cat.rename_categories([str(x) for x in d[c].cat.categories]).cat.set_categories(cmap[c])
            for c in d.columns:
                if c not in CAT and d[c].dtype != np.float32: d[c] = d[c].astype('float32')
            gc.collect()
        num = bag_num(comp, k)
        dm = xgb.QuantileDMatrix(_ChunkIter(d, num + CAT, y, w), enable_categorical=True, max_bin=256); gc.collect()   # 청크 공급(메모리 절약, hist 256구간)
        P = dict(cfg['params']); P['seed'] = k
        b = xgb.train(P, dm, cfg['rounds']); del dm; gc.collect()
        b.save_model(fn)
        pickle.dump(dict(cmap=cmap, num=num), open(os.path.join(OUT, f'{comp}_{k}_cats.pkl'), 'wb'))
        print(f'  {comp} 구성원 {k} {time.time() - tk:.0f}s peak {peak_mem_gb()}GB', flush=True)


def xgb_predict(comp, d):
    """배깅 구성원 확률의 로짓 평균"""
    import xgboost as xgb
    zs = []
    for k in range(CP[comp].get('n_bag', 1)):
        b = xgb.Booster(); b.load_model(os.path.join(OUT, f'{comp}_{k}.json'))
        o = pickle.load(open(os.path.join(OUT, f'{comp}_{k}_cats.pkl'), 'rb'))
        p = b.predict(xgb.DMatrix(xmat(d, o['num'], o['cmap']), enable_categorical=True))
        p = np.clip(p, 1e-6, 1 - 1e-6); zs.append(np.log(p / (1 - p)))
    return 1 / (1 + np.exp(-np.mean(zs, 0)))


class LogPrep:
    """log1p(개수형) -> 결측지시자(결측>1%) -> 0.5/99.5% 윈저화 -> 중앙값 대치 -> 표준화 + 원핫(min_freq 50, 드문범주 열)"""
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


def step_fit(comp):
    t = time.time()
    if comp in ('cls', 'haz'):
        fit_xgb(comp, 'frame_cls.parquet' if comp == 'cls' else 'frame_haz.parquet')
    elif comp == 'logit':
        from sklearn.linear_model import LogisticRegression
        cfg = CP['logit']
        d = pd.read_parquet(os.path.join(OUT, 'frame_cls.parquet'), columns=cfg['num'] + CAT + ['y', 'L'])
        if len(d) > cfg.get('max_rows', 10 ** 9):        # 메모리 상한(사전 고정 규칙)
            d = d.sample(n=cfg['max_rows'], random_state=0)
        prep = LogPrep(cfg['num'], cfg['log1p']).fit(d)
        X = prep.transform(d); y = d.y.values; w = recency_w(d.L.values); del d; gc.collect()
        m = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter']).fit(X, y, sample_weight=w)
        pickle.dump(dict(m=m, prep=prep), open(os.path.join(OUT, 'logit.pkl'), 'wb'))
    elif comp == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder
        cfg = CP['hgb']
        d = pd.read_parquet(os.path.join(OUT, 'frame_cls.parquet'), columns=cfg['num'] + CAT + ['y', 'L'])
        if len(d) > cfg['max_rows']:           # 메모리 상한(HGB는 내부 float64 사본) — 사전 고정 규칙
            d = d.sample(n=cfg['max_rows'], random_state=0)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1, max_categories=60)
        c = oe.fit_transform(d[CAT].astype(str)).astype('float32'); c[c < 0] = np.nan
        X = np.c_[d[cfg['num']].to_numpy(dtype='float32', na_value=np.nan), c]; y = d.y.values; w = recency_w(d.L.values)
        del d, c; gc.collect()
        m = HistGradientBoostingClassifier(**cfg['params'], categorical_features=list(range(len(cfg['num']), len(cfg['num']) + len(CAT))),
                                           early_stopping=False, random_state=0).fit(X, y, sample_weight=w)
        pickle.dump(dict(m=m, oe=oe), open(os.path.join(OUT, 'hgb.pkl'), 'wb'))
    elif comp == 'base':
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import make_pipeline
        b = SPEC['baseline']
        d = pd.read_parquet(os.path.join(OUT, 'frame_base.parquet'))
        med = float(d.claims.median())
        X = base_X(d, med)
        ct = ColumnTransformer([('n', StandardScaler(), b['num']), ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=b['min_frequency']), b['cat'])])
        m = make_pipeline(ct, LogisticRegression(C=b['C'], max_iter=2000)).fit(X, d.y.values)
        pickle.dump(dict(m=m, med=med), open(os.path.join(OUT, 'base.pkl'), 'wb'))
    print(f'fit {comp} {time.time() - t:.0f}s peak {peak_mem_gb()}GB', flush=True)
    mark_done('fit_' + comp, dict(sec=round(time.time() - t), peak_gb=peak_mem_gb()))


def base_X(d, med):
    b = SPEC['baseline']
    X = d[b['num'] + b['cat']].copy()
    X['claims'] = np.log1p(X.claims.astype(float).fillna(med))
    for c in b['cat']: X[c] = X[c].astype(str)
    return X.fillna(0)


# ============================================================== 단계: 예측
def predict_L(L, chunk=120_000):
    """메모리 상한을 위해 위험집합을 행 청크로 나눠 예측(결과는 청크와 무관)"""
    f = T.features(L)
    f = f.assign(acd=T.load()['pt'].acd.values[f.pid.values])
    outs = []
    for a in range(0, len(f), chunk):
        outs.append(_predict_chunk(L, f.iloc[a:a + chunk].reset_index(drop=True)))
        gc.collect()
    return pd.concat(outs, ignore_index=True)


def _predict_chunk(L, f):
    out = pd.DataFrame({'pid': f.pid.values, 'L': L, 'ksic': f.ksic.values, 'acd': f.acd.values, 'atype': f.atype.values})
    d = to_str(slim(enrich(f, ak=True)))
    assert (d.pid.values == out.pid.values).all()
    # cls
    out['p_cls'] = xgb_predict('cls', d)
    # haz
    dec = HZ.eval_decisions(f.reset_index(drop=True))
    dec = to_str(slim(enrich(dec.drop(columns=['acd'], errors='ignore'), ak=False)))
    h = xgb_predict('haz', dec)
    out['p_haz'] = HZ.combine(dec, h, len(f))
    del dec, h; gc.collect()
    # logit
    o = pickle.load(open(os.path.join(OUT, 'logit.pkl'), 'rb'))
    out['p_logit'] = o['m'].predict_proba(o['prep'].transform(d))[:, 1]
    # hgb
    o = pickle.load(open(os.path.join(OUT, 'hgb.pkl'), 'rb'))
    c = o['oe'].transform(d[CAT].astype(str)).astype('float32'); c[c < 0] = np.nan
    out['p_hgb'] = o['m'].predict_proba(np.c_[d[CP['hgb']['num']].to_numpy(dtype='float32', na_value=np.nan), c])[:, 1]
    # baseline
    o = pickle.load(open(os.path.join(OUT, 'base.pkl'), 'rb'))
    out['p_base'] = o['m'].predict_proba(base_X(f, o['med']))[:, 1]
    # 앙상블(사전 고정 가중 로짓평균)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    z = sum(wt * lg(out['p_' + k].values) for k, wt in SPEC['ensemble']['weights'].items())
    out['p'] = 1 / (1 + np.exp(-z))
    return out


def step_pred():
    for L in EVAL_L:
        fn = os.path.join(OUT, f'pred_{L.replace("-", "")}.parquet')
        if os.path.exists(fn) and not A.force: continue
        t = time.time()
        predict_L(L).to_parquet(fn, index=False)
        print('pred', L, f'{time.time() - t:.0f}s peak {peak_mem_gb()}GB', flush=True)
        if time.time() - T0 > BUDGET:
            sys.exit(3)
    mark_done('pred', dict(Ls=EVAL_L))


# ============================================================== 단계: 평가
def _prep_rank(y, s):
    o = np.argsort(s, kind='mergesort'); so = s[o]
    grp = np.r_[0, np.cumsum(so[1:] != so[:-1])]
    return o, y[o].astype(float), grp


def _wauc(pr, w):
    o, yo, grp = pr
    wo = w[o]; G = grp[-1] + 1
    P = np.bincount(grp, wo * yo, G); N = np.bincount(grp, wo * (1 - yo), G)
    below = np.cumsum(N) - N
    return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


def _wap(pr, w):
    o, yo, grp = pr
    wo = w[o]; G = grp[-1] + 1
    P = np.bincount(grp, wo * yo, G)[::-1]; N = np.bincount(grp, wo * (1 - yo), G)[::-1]   # 점수 내림차순
    tp = np.cumsum(P); fp = np.cumsum(N)
    rec = tp / tp[-1]; prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    return dict(auc=float(roc_auc_score(y, p)), prauc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                base_rate=float(np.mean(y)), mean_pred=float(np.mean(p)), n=int(len(y)))


def cluster_boot(d, B, seed=0):
    """군집 = 1순위 출원인(출원인 없는 특허는 특허 단위). (특허, L) 행이 기준일마다 반복되므로 특허를 포함하는 출원인 군집으로 재표집"""
    cl = np.where(d.acd.values == 'NA', 'P' + d.pid.astype(str).values, d.acd.values)
    codes, uniq = pd.factorize(cl)
    K = len(uniq)
    y = d.y.values
    prm, prb = _prep_rank(y, d.p.values), _prep_rank(y, d.p_base.values)
    rng = np.random.default_rng(seed)
    R = {k: [] for k in ['auc', 'auc_base', 'auc_diff', 'skill', 'prauc', 'prauc_base']}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        w = cnt[codes]
        a, ab = _wauc(prm, w), _wauc(prb, w)
        R['auc'].append(a); R['auc_base'].append(ab); R['auc_diff'].append(a - ab); R['skill'].append((a - ab) / (1 - ab))
        R['prauc'].append(_wap(prm, w)); R['prauc_base'].append(_wap(prb, w))
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in R.items()}, K


def step_score():
    d = pd.concat([pd.read_parquet(os.path.join(OUT, f'pred_{L.replace("-", "")}.parquet')) for L in EVAL_L], ignore_index=True)
    if MODE == 'future':
        return future_output(d)
    y = np.concatenate([labels(L, d[d.L == L].pid.values, None) for L in EVAL_L])
    d['y'] = y
    assert np.isfinite(y).all() and set(np.unique(y)) <= {0.0, 1.0}
    res = dict(mode=MODE, C_train_label_cutoff=C, eval_L=EVAL_L, unit='(특허, 기준일 L) — L 시점 납부로 살아 있는 KR 등록특허',
               outcome='E(첫 미납 연차 기한) ∈ (L, L+3년] 인 등록료불납 소멸', cluster='1순위 출원인코드(없으면 특허)')
    res['model'] = metrics(y, d.p.values)
    res['baseline'] = metrics(y, d.p_base.values)
    res['components'] = {k: metrics(y, d['p_' + k].values)['auc'] for k in ['cls', 'haz', 'logit', 'hgb']}
    res['skill_auc'] = (res['model']['auc'] - res['baseline']['auc']) / (1 - res['baseline']['auc'])
    res['prauc_ratio_vs_baseline'] = res['model']['prauc'] / res['baseline']['prauc']
    res['per_L'] = {L: dict(auc=metrics(d.y[d.L == L].values, d.p[d.L == L].values)['auc'],
                            auc_base=metrics(d.y[d.L == L].values, d.p_base[d.L == L].values)['auc'],
                            n=int((d.L == L).sum()), rate=float(d.y[d.L == L].mean())) for L in EVAL_L}
    B = A.boot or SPEC['bootstrap']['B']
    ci, K = cluster_boot(d, B)
    res['bootstrap_95ci'] = ci; res['bootstrap_B'] = B; res['n_clusters'] = K
    # 산업 집계 타당도(트렌드 지표): 기준일별 KSIC 50개 평균 예측위험 vs 실현 3년 불납률
    from scipy.stats import spearmanr
    iv = {}
    for L in EVAL_L:
        g = d[d.L == L].groupby('ksic').agg(pred=('p', 'mean'), base=('p_base', 'mean'), real=('y', 'mean'))
        iv[L] = dict(spearman_model=round(float(spearmanr(g.pred, g.real)[0]), 3), spearman_baseline=round(float(spearmanr(g.base, g.real)[0]), 3), n_ksic=len(g))
    res['industry_level_validity'] = iv
    # 세그먼트
    seg = {}
    for col in ['atype']:
        seg[col] = {k: dict(n=int(len(g)), auc=round(metrics(g.y.values, g.p.values)['auc'], 4), auc_base=round(metrics(g.y.values, g.p_base.values)['auc'], 4))
                    for k, g in d.groupby(col) if g.y.nunique() > 1}
    res['segments'] = seg
    res['train_info'] = {k: v for k, v in done_state().items()}
    fn = os.path.join(HERE, 'holdout_result.json') if (MODE == 'holdout' and not A.tag) else os.path.join(OUT, f'{MODE}_result.json')
    json.dump(res, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ['model', 'baseline', 'components', 'skill_auc', 'bootstrap_95ci']}, ensure_ascii=False, indent=1))
    print('saved', fn)
    mark_done('score')


def future_output(d):
    """전체 자료 재학습 모형의 KSIC별 예측 포기위험 추이 + 최신 기준일 기준 향후 3년 위험 상위 산업"""
    nm = pd.read_csv(SPEC['paths']['ksic_names'], dtype={'세세분류코드': str}).drop_duplicates('세세분류코드').set_index('세세분류코드')['산업명']
    last_complete = SPEC['future']['universe_complete_until']
    obs_ok = [L for L in EVAL_L if T.add_years(np.array([np.datetime64(L)]), 3)[0] + np.timedelta64(400, 'D') <= np.datetime64(C)]
    real = {}
    for L in obs_ok:
        s = d[d.L == L]
        real[L] = pd.Series(labels(L, s.pid.values, None), index=s.index)
    d['y_realized'] = pd.concat(real.values()) if real else np.nan
    g = d.groupby(['ksic', 'L']).agg(n_alive=('pid', 'size'), pred_risk=('p', 'mean'), exp_lapses_3y=('p', 'sum'),
                                      realized_rate=('y_realized', 'mean')).reset_index()
    tot = d.groupby('L').agg(n_alive=('pid', 'size'), pred_risk=('p', 'mean'), exp_lapses_3y=('p', 'sum'), realized_rate=('y_realized', 'mean')).reset_index()
    tot['ksic'] = 'ALL'
    g = pd.concat([g, tot], ignore_index=True)
    g['industry'] = g.ksic.map(nm).fillna('전체')
    g['universe_complete'] = g.L <= last_complete
    latest = max(L for L in EVAL_L if L <= last_complete)
    base = SPEC['future']['trend_base_L']
    cur = g[g.L == latest].set_index('ksic'); b0 = g[g.L == base].set_index('ksic')
    g['rank_risk_latest'] = g.ksic.map(cur.pred_risk.rank(ascending=False))
    g['delta_vs_base'] = g.ksic.map(cur.pred_risk - b0.pred_risk)
    g['rank_rise_latest'] = g.ksic.map((cur.pred_risk - b0.pred_risk).rank(ascending=False))
    g['C_train_label_cutoff'] = C
    g = g.sort_values(['ksic', 'L'])
    fn = os.path.join(HERE, 'future_forecast.csv')
    g.to_csv(fn, index=False, encoding='utf-8-sig')
    print('saved', fn, 'latest', latest)
    print(g[g.L == latest].sort_values('pred_risk', ascending=False).head(10)[['ksic', 'industry', 'n_alive', 'pred_risk', 'delta_vs_base']].to_string())
    mark_done('score', dict(latest=latest))


# ============================================================== main
if __name__ == '__main__':
    if A.step == 'all':
        print(f'mode={MODE} C={C} eval_L={EVAL_L[0]}..{EVAL_L[-1]} ({len(EVAL_L)}) holdout_env={os.environ.get("T4_STAGE3_HOLDOUT")}', flush=True)
        driver()
        sys.exit(0)
    st = A.step
    if st == 'feats': step_feats()
    elif st == 'frame_cls': step_frame_cls()
    elif st == 'frame_haz': step_frame_haz()
    elif st.startswith('fit_'): step_fit(st[4:])
    elif st == 'pred': step_pred()
    elif st == 'score': step_score()
    else: sys.exit('unknown step')
