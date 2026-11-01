# -*- coding: utf-8 -*-
"""T4 2라운드 공용 라이브러리 — 전진 검증(walk-forward) 하네스.

- 1라운드 모듈(t4lib·soc·hazard·feat2·run)을 **읽기 전용으로 import** 해서 쓰되, 캐시 쓰기 경로를 R2/cache 로 돌린다.
- 소멸 이벤트 적재 컷(LAP_CUT): 개발 모드는 E < 2022-01-01 (최종시험 F=2022 결과 차단). 최종 모드만 전체.
- 개발 원점(PROTOCOL_R2 §2: 기준일 L <= 2019-01-01, 3년 결과 <= 2021-12):
    O12: C=2012-01-01, 평가 L 2012-01-01, 2012-07-01
    O14: C=2014-01-01, 평가 L 2014-01-01, 2014-07-01
    O16: C=2016-01-01, 평가 L 2016-01-01, 2016-07-01
    O18: C=2018-01-01, 평가 L 2018-01-01, 2018-07-01   (결과 창 끝 2021-07-01)
  원점마다 학습 레이블은 C 시점 관측분만(label_asof, d+450일 <= C) — 1라운드 규칙 그대로.
"""
import os, sys, json, time, pickle, gc, hashlib

R2 = os.path.dirname(os.path.abspath(__file__))
T4 = os.path.dirname(R2)
if T4 not in sys.path:
    sys.path.insert(0, T4)
os.environ.pop('T4_STAGE3_HOLDOUT', None)
os.environ.setdefault('OMP_NUM_THREADS', '2')
try:
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np, pandas as pd

MODE = os.environ.get('T4R2_MODE', 'dev')          # dev | final | future (final_run_r2.py 가 설정)
F_START = np.datetime64('2022-01-01')               # 🔒 최종시험 첫 결과연도
DEV_LAST_L = np.datetime64('2018-07-01')

import t4lib as T
T.CACHE = os.path.join(R2, 'cache')
os.makedirs(T.CACHE, exist_ok=True)
if MODE == 'dev':
    T.HOLDOUT_START = F_START                        # label_full: L+3y > 2022-01-01 이면 예외, guard_L: L >= 2022 이면 예외
    LAP_CUT = F_START
elif MODE == 'final':
    # 최종시험: 학습 레이블(label_asof, 소멸등록 cd<=C=2022-01-01)과 피처(cd<=L)는 E<2022 소멸만으로 충분 → 같은 컷 유지.
    # 평가 레이블은 label_np(납부기록)로만 만든다. guard_L 은 L=2022 피처 계산을 위해 해제.
    T.HOLDOUT_START = np.datetime64('2100-01-01')
    LAP_CUT = F_START
else:                                                # future: 전체 가용 자료
    T.HOLDOUT_START = np.datetime64('2100-01-01')
    LAP_CUT = None

_D = {}


def r2_load():
    """1라운드 적재 캐시(loaded_all.pkl, 읽기 전용)를 읽어 소멸 이벤트를 LAP_CUT 이전 실효분만 남긴다."""
    if _D:
        return _D
    tag = 'dev' if LAP_CUT is not None else 'all'
    pk = os.path.join(T.CACHE, f'loaded_r2{tag}.pkl')
    if os.path.exists(pk):
        _D.update(pickle.load(open(pk, 'rb')))
        return _D
    src = os.path.join(T4, 'cache', 'loaded_all.pkl')
    D = pickle.load(open(src, 'rb'))
    if LAP_CUT is not None:
        D['lap'] = D['lap'][D['lap'].E < LAP_CUT].reset_index(drop=True)     # 🔒 F 이후 실효 소멸은 메모리에 올리지 않음
    _D.update(D)
    pickle.dump(dict(_D), open(pk, 'wb'), protocol=4)
    return _D


T.load = r2_load
T._D = _D

import soc as SO, hazard as HZ, feat2 as F2       # feat2.CACHE 는 import 시점의 T.CACHE(R2/cache)에 묶인다
from run import train_landmarks, labels, get_frame, semi
assert F2.CACHE == T.CACHE

SPEC0 = json.load(open(os.path.join(T4, 'final_spec.json'), encoding='utf-8'))
CAT = SPEC0['categorical']
ORIGINS = [('O12', '2012-01-01', ['2012-01-01', '2012-07-01']),
           ('O14', '2014-01-01', ['2014-01-01', '2014-07-01']),
           ('O16', '2016-01-01', ['2016-01-01', '2016-07-01']),
           ('O18', '2018-01-01', ['2018-01-01', '2018-07-01'])]
if MODE == 'final':
    ORIGINS = [('F22', '2022-01-01', ['2022-01-01'])]          # 🔒 최종시험(1회). 결과 창 2022-01~2025-01, 유예 경과 2025-07 < DATA_END
elif MODE == 'future':
    ORIGINS = [('FUT', '2025-08-22', ['2025-07-01'])]          # 전체 자료 재학습. 추이 예측은 predict_saved 로 기준일마다
ORIG = {o: (c, ls) for o, c, ls in ORIGINS}
RUNS = {'dev': 'runs', 'final': 'runs_final', 'future': 'runs_future'}[MODE]
if MODE == 'dev':
    for _o, _c, _ls in ORIGINS:
        for _L in _ls:
            _end = T.add_years(np.array([np.datetime64(_L)]), 3)[0]
            assert _end < F_START, '개발 원점 결과 구간이 F 이후와 겹침'
TREND = ['ksic_lapse_1y', 'ksic_lapse_trend', 'ipc_lapse_2y']
EXTRA_KEEP = TREND + ['ipc_sub', 'ipc_cls', 'rgst_year', 'port_n', 'a_decisions', 'age_total', 'last_amt_per_yr',
                      'ag_rate', 'ag_rate1y', 'ag_alive', 'ag_n', 'ow_n', 'log_port', 'exp_d', 'acd', 'atype', 'ksic']


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


# ================================================================ 피처 보강(1라운드와 동일)
def add_ratio(d):
    d = d.copy()
    d['port_alive_r'] = d.port_alive / d.port_n.replace(0, np.nan)
    d['port_new3_r'] = d.port_new3 / d.port_n.replace(0, np.nan)
    d['log_port'] = np.log1p(d.port_n)
    d['exp_near'] = ((d.exp_d if 'exp_d' in d else d.to_expiry) < 3).astype(int)
    return d


def enrich(d, ak=True):
    d = add_ratio(d)
    d = SO.add_soc(d)
    d = F2.add_fam_dated(d)
    d = F2.fix_dspst(d)
    if ak: d = SO.add_ak(d, 'next_annl')
    d = F2.add_macro(d)
    d = F2.add_agent(d)
    d = F2.add_owner(d)
    return d


def all_cols(spec=SPEC0):
    s = set()
    for c in spec['components'].values(): s |= set(c['num'])
    s |= set(CAT) | set(spec['baseline']['num']) | set(spec['baseline']['cat']) | set(EXTRA_KEEP)
    return s


def slim(d):
    keep = all_cols() | {'y', 'L', 'pid', 'acd', 'k', 'd', 'row'}
    d = d[[c for c in d.columns if c in keep]]
    fc = d.select_dtypes('float64').columns
    return d.astype({c: 'float32' for c in fc})


def to_str(d):
    return d.astype({c: str for c in d.select_dtypes(['category', 'object']).columns if c != 'L'})


def odir(o):
    p = os.path.join(R2, RUNS, o)
    os.makedirs(os.path.join(p, 'pred'), exist_ok=True)
    return p


SAVE = os.environ.get('T4R2_SAVE') == '1'          # final/future 모드: 모형 저장(재현·기준일별 예측용)


def mdir(o):
    p = os.path.join(odir(o), 'models'); os.makedirs(p, exist_ok=True)
    return p


def haz_train_Ls(C, first='2005-01-01', semi=False):
    a = [f'{y}-01-01' for y in range(int(first[:4]), int(C[:4]) - 1)]
    if semi:                                     # 반기 기준일 추가(7월) — 마지막은 (C연도-2)-07-01, 결정 d+450일<=C 규칙은 동일
        a = sorted(a + [f'{y}-07-01' for y in range(int(first[:4]), int(C[:4]) - 1)])
    return a


def haz_stem(spec):
    return 'frame_hazs' if spec['train'].get('haz_semi') else 'frame_haz'


# ================================================================ 프레임(원점별 1회, 캐시)
def build_frames(o, cap_cls=1_000_000, cap_haz=1_000_000, first='2005-01-01', which=('cls', 'haz', 'ev')):
    C, EL = ORIG[o]
    out = odir(o)
    pt_acd = T.load()['pt'].acd.values
    if MODE != 'dev':
        for L in set(train_landmarks(C, first) + haz_train_Ls(C, first) + list(EL)):
            seed_cache_from_r1(L)
    t = time.time()
    fn = os.path.join(out, 'frame_cls.parquet')
    if 'cls' in which and not os.path.exists(fn):
        Ls = train_landmarks(C, first)
        tot = sum(len(T.features(L)) for L in Ls)
        frac = min(1.0, cap_cls / tot)
        parts, bparts = [], []
        for L in Ls:
            f = get_frame([L], C)
            bparts.append(f[['pid', 'L', 'y'] + SPEC0['baseline']['num'] + SPEC0['baseline']['cat']])
            if frac < 1: f = f.sample(frac=frac, random_state=0)
            parts.append(to_str(slim(enrich(f, ak=True))))
        d = pd.concat(parts, ignore_index=True)
        d.to_parquet(fn, index=False)
        pd.concat(bparts, ignore_index=True).to_parquet(os.path.join(out, 'frame_base.parquet'), index=False)
        json.dump(dict(Ls=Ls, total=tot, frac=frac, rows=len(d), pos=float(d.y.mean())), open(os.path.join(out, 'frame_cls.json'), 'w'))
        print(o, 'frame_cls', len(d), f'frac {frac:.3f}', f'{time.time() - t:.0f}s', flush=True)
    for hv, stem in (('haz', 'frame_haz'), ('hazs', 'frame_hazs')):
        fn = os.path.join(out, stem + '.parquet')
        if hv in which and not os.path.exists(fn):
            Ls = haz_train_Ls(C, first, semi=(hv == 'hazs'))
            tot = sum(len(HZ.train_decisions(L, C)) for L in Ls)
            frac = min(1.0, cap_haz / tot)
            parts = []
            for L in Ls:
                f = HZ.train_decisions(L, C)
                if frac < 1: f = f.sample(frac=frac, random_state=0)
                f = f.assign(acd=pt_acd[f.pid.values])
                parts.append(to_str(slim(enrich(f, ak=False))))
            d = pd.concat(parts, ignore_index=True)
            d.to_parquet(fn, index=False)
            json.dump(dict(Ls=Ls, total=tot, frac=frac, rows=len(d), pos=float(d.y.mean())), open(os.path.join(out, stem + '.json'), 'w'))
            print(o, stem, len(d), f'frac {frac:.3f}', f'{time.time() - t:.0f}s', flush=True)
    fn = os.path.join(out, 'ev.parquet')
    for i, L in enumerate(EL):                          # 평가 기준일마다 별도 프로세스(메모리 4GB 이하)
        pf = os.path.join(out, f'ev_part{i}.parquet')
        if f'ev{i}' in which and not os.path.exists(pf) and not os.path.exists(fn):
            f = T.features(L)
            f = f.assign(acd=pt_acd[f.pid.values]).reset_index(drop=True)
            y = label_fn(L, f.pid.values)
            d = to_str(slim(enrich(f, ak=True)))
            assert (d.pid.values == f.pid.values).all()
            d = d.assign(y=y)
            d.to_parquet(pf, index=False)
            n = len(d); del d; gc.collect()
            dec = HZ.eval_decisions(f)
            dec = to_str(slim(enrich(dec.drop(columns=['acd']), ak=False)))
            dec = dec.assign(acd=pt_acd[dec.pid.values])
            dec.to_parquet(os.path.join(out, f'evh_part{i}.parquet'), index=False)
            print(o, 'ev', L, n, f'{time.time() - t:.0f}s', flush=True)
    if 'ev' in which and not os.path.exists(fn):
        evs, hzs, off = [], [], 0
        for i, L in enumerate(EL):
            d = pd.read_parquet(os.path.join(out, f'ev_part{i}.parquet'))
            h = pd.read_parquet(os.path.join(out, f'evh_part{i}.parquet'))
            h['row'] = h['row'].values + off
            off += len(d)
            evs.append(d); hzs.append(h)
        ev = pd.concat(evs, ignore_index=True)
        ev.to_parquet(fn, index=False)
        pd.concat(hzs, ignore_index=True).to_parquet(os.path.join(out, 'ev_haz.parquet'), index=False)
        for i in range(len(EL)):
            os.remove(os.path.join(out, f'ev_part{i}.parquet')); os.remove(os.path.join(out, f'evh_part{i}.parquet'))
        print(o, 'ev', len(ev), 'pos', round(float(ev.y.mean()), 4) if ev.y.notna().all() else 'NA', f'{time.time() - t:.0f}s', flush=True)


def label_fn(L, pids):
    """평가 레이블. dev: 1라운드 완전 레이블(label_full, 결과 창 끝 < F). final: label_np(납부기록 기반, DATA_END 기준)."""
    if MODE == 'dev':
        return labels(L, pids, None)
    if MODE == 'final':
        return label_np(L, pids)
    return np.full(len(pids), np.nan)


# ================================================================ 외부/추가 피처 결합
def _frame_cols(o, stem):
    import pyarrow.parquet as pq
    return set(pq.ParquetFile(os.path.join(odir(o), stem + '.parquet')).schema.names)


def xpath(o, stem, col):
    p = os.path.join(odir(o), 'x'); os.makedirs(p, exist_ok=True)
    return os.path.join(p, f'{stem}__{col}.npy')


def missing_extra(o, stem, cols):
    have = _frame_cols(o, stem)
    return [c for c in cols if c not in have and not os.path.exists(xpath(o, stem, c))]


def build_extra(o, stem, cols):
    """frame 에 없는 열을 r2ext 로 계산해 행 순서대로 npy 저장(별도 프로세스에서 1회)"""
    import r2ext
    idc = ['pid', 'L', 'ksic', 'next_annl', 'k', 'acd', 'age', 'atype']
    d = _read_frame(o, stem + '.parquet', idc)
    d = r2ext.add(d, cols)
    for c in cols:
        np.save(xpath(o, stem, c), d[c].values.astype('float32'))
    print(o, stem, 'extra', cols, 'nan', {c: round(float(np.isnan(d[c].values).mean()), 3) for c in cols}, flush=True)


def join_extra(d, cols, o, stem):
    miss = [c for c in cols if c not in d.columns]
    for c in miss:
        v = np.load(xpath(o, stem, c))
        assert len(v) == len(d), (o, stem, c)
        d[c] = v
    return d


# ================================================================ 구성요소 학습·예측
def recency_w(Lvals, spec):
    tr = spec['train']
    yr = pd.to_datetime(pd.Series(np.asarray(Lvals).astype(str))).dt.year.values
    if tr.get('weight_kind', 'linear') == 'exp':
        return np.exp((yr - yr.max()) / tr['weight_tau'])
    return 1.0 + (yr - tr['weight_base_year']) / tr['weight_div']


def comp_key(spec, comp):
    cfg = spec['components'][comp]
    drop = {'cls_label_rule', 'haz_label_rule', 'cap_rule'} | (set() if comp.startswith('haz') else {'haz_semi'})
    blob = dict(comp=comp, cfg=cfg, CAT=CAT, train={k: v for k, v in spec['train'].items() if k not in drop},
                ext_ver=spec.get('ext_ver', {}))
    return comp + '_' + hashlib.md5(json.dumps(blob, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def _read_frame(o, frame, cols):
    import pyarrow.parquet as pq
    fn = os.path.join(odir(o), frame)
    have = set(pq.ParquetFile(fn).schema.names)
    need = [c for c in cols if c in have]
    d = pq.read_table(fn, columns=need).to_pandas()
    return d


def _filter_window(d, spec):
    w = spec['train'].get('window_years')
    if w:
        yr = pd.to_datetime(d.L.astype(str)).dt.year
        d = d[yr >= yr.max() - w + 1].reset_index(drop=True)
    return d


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


def fit_predict_xgb_bag(o, spec, comp, k):
    """XGB 구성요소의 배깅 구성원 k 1개 학습 → 평가행 예측 로짓 저장"""
    import xgboost as xgb
    key = comp_key(spec, comp)
    fn = os.path.join(odir(o), 'pred', f'{key}_b{k}.npy')
    if os.path.exists(fn):
        return fn
    cfg = spec['components'][comp]
    t = time.time()
    frame = 'frame_cls.parquet' if comp.startswith('cls') else haz_stem(spec) + '.parquet'
    base_cols = cfg['num'] + CAT + ['y', 'L', 'pid', 'ksic', 'acd', 'atype', 'k', 'next_annl', 'age']
    d = _read_frame(o, frame, list(dict.fromkeys(base_cols)))
    d = join_extra(d, cfg['num'], o, frame[:-8])
    d = _filter_window(d, spec)
    y = d.y.values.astype('float32'); w = recency_w(d.L.values, spec).astype('float32')
    X = d[cfg['num'] + CAT].copy(); del d; gc.collect()
    for c in cfg['num']:
        if X[c].dtype != np.float32: X[c] = X[c].astype('float32')
    X, cmap = _prep_cat(X)
    num = cfg['num'] if k == 0 else list(np.random.default_rng(k).permutation(cfg['num']))
    dm = xgb.QuantileDMatrix(_ChunkIter(X, num + CAT, y, w), enable_categorical=True, max_bin=256); gc.collect()
    P = dict(cfg['params']); P['seed'] = k
    b = xgb.train(P, dm, cfg['rounds']); del dm, X; gc.collect()
    tfit = time.time() - t
    if SAVE:
        b.save_model(os.path.join(mdir(o), f'{key}_b{k}.json'))
        pickle.dump(dict(cmap=cmap, num=num), open(os.path.join(mdir(o), f'{key}_b{k}.pkl'), 'wb'))
    # 예측
    evf = 'ev.parquet' if comp.startswith('cls') else 'ev_haz.parquet'
    e = _read_frame(o, evf, list(dict.fromkeys(cfg['num'] + CAT + ['pid', 'L', 'ksic', 'acd', 'atype', 'row', 'k', 'next_annl', 'age'])))
    e = join_extra(e, cfg['num'], o, evf[:-8])
    Xe = e[num + CAT].copy()
    for c in num:
        if Xe[c].dtype != np.float32: Xe[c] = Xe[c].astype('float32')
    Xe, _ = _prep_cat(Xe, cmap)
    p = b.predict(xgb.DMatrix(Xe, enable_categorical=True))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).astype('float32')
    np.save(fn, z)
    print(f'  {o} {comp} bag{k} fit {tfit:.0f}s total {time.time() - t:.0f}s peak {peak_mem_gb()}GB', flush=True)
    return fn


def combine_xgb(o, spec, comp):
    """배깅 로짓 평균 → (cls) 평가행 확률 / (haz) 결정 hazard 결합 P"""
    key = comp_key(spec, comp)
    fn = os.path.join(odir(o), 'pred', f'{key}.npy')
    if os.path.exists(fn):
        return np.load(fn)
    nb = spec['components'][comp].get('n_bag', 1)
    z = np.mean([np.load(os.path.join(odir(o), 'pred', f'{key}_b{k}.npy')) for k in range(nb)], 0)
    p = 1 / (1 + np.exp(-z))
    if comp.startswith('haz'):
        import pyarrow.parquet as pq
        row = pq.read_table(os.path.join(odir(o), 'ev_haz.parquet'), columns=['row']).to_pandas().row.values
        n = pq.ParquetFile(os.path.join(odir(o), 'ev.parquet')).metadata.num_rows
        p = HZ.combine(pd.DataFrame({'row': row}), p, n)
    np.save(fn, p.astype('float64'))
    return p


class LogPrep:
    """1라운드 final_run.LogPrep 과 동일"""
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


def fit_predict_sk(o, spec, comp):
    """logit / hgb / base 구성요소 학습 → 평가행 확률 저장"""
    key = comp_key(spec, comp) if comp != 'base' else 'base_' + hashlib.md5(json.dumps(spec['baseline'], sort_keys=True).encode()).hexdigest()[:10]
    fn = os.path.join(odir(o), 'pred', f'{key}.npy')
    if os.path.exists(fn):
        return np.load(fn)
    t = time.time()
    if comp == 'base':
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import make_pipeline
        b = spec['baseline']
        d = pd.read_parquet(os.path.join(odir(o), 'frame_base.parquet'))
        med = float(d.claims.median())
        ct = ColumnTransformer([('n', StandardScaler(), b['num']), ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=b['min_frequency']), b['cat'])])
        m = make_pipeline(ct, LogisticRegression(C=b['C'], max_iter=2000)).fit(base_X(d, med, b), d.y.values)
        if SAVE: pickle.dump(dict(m=m, med=med), open(os.path.join(mdir(o), f'{key}.pkl'), 'wb'))
        e = pd.read_parquet(os.path.join(odir(o), 'ev.parquet'), columns=b['num'] + b['cat'])
        p = m.predict_proba(base_X(e, med, b))[:, 1]
    else:
        cfg = spec['components'][comp]
        cols = list(dict.fromkeys(cfg['num'] + CAT + ['y', 'L', 'pid', 'ksic', 'acd', 'atype', 'next_annl', 'age']))
        d = _read_frame(o, 'frame_cls.parquet', cols)
        d = join_extra(d, cfg['num'], o, 'frame_cls')
        d = _filter_window(d, spec)
        if len(d) > cfg.get('max_rows', 10 ** 9):
            d = d.sample(n=cfg['max_rows'], random_state=0)
        e = _read_frame(o, 'ev.parquet', [c for c in cols if c not in ('y',)])
        e = join_extra(e, cfg['num'], o, 'ev')
        if comp.startswith('logit'):
            from sklearn.linear_model import LogisticRegression
            prep = LogPrep(cfg['num'], cfg['log1p']).fit(d)
            X = prep.transform(d); y = d.y.values; w = recency_w(d.L.values, spec); del d; gc.collect()
            m = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter']).fit(X, y, sample_weight=w)
            del X; gc.collect()
            if SAVE: pickle.dump(dict(m=m, prep=prep), open(os.path.join(mdir(o), f'{key}.pkl'), 'wb'))
            p = m.predict_proba(prep.transform(e))[:, 1]
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1, max_categories=60)
            c = oe.fit_transform(d[CAT].astype(str)).astype('float32'); c[c < 0] = np.nan
            X = np.c_[d[cfg['num']].to_numpy(dtype='float32', na_value=np.nan), c]; y = d.y.values; w = recency_w(d.L.values, spec)
            allnan = np.where(np.isnan(X[:, :len(cfg['num'])]).all(0))[0]      # 학습 구간에 값이 전혀 없는 열(예: O12 의 3년 경기지표) → 상수 0(정보 없음)
            X[:, allnan] = 0
            del d, c; gc.collect()
            m = HistGradientBoostingClassifier(**cfg['params'], categorical_features=list(range(len(cfg['num']), len(cfg['num']) + len(CAT))),
                                               early_stopping=False, random_state=0).fit(X, y, sample_weight=w)
            del X; gc.collect()
            if SAVE: pickle.dump(dict(m=m, oe=oe, allnan=allnan), open(os.path.join(mdir(o), f'{key}.pkl'), 'wb'))
            ce = oe.transform(e[CAT].astype(str)).astype('float32'); ce[ce < 0] = np.nan
            Xe = np.c_[e[cfg['num']].to_numpy(dtype='float32', na_value=np.nan), ce]; Xe[:, allnan] = 0
            p = m.predict_proba(Xe)[:, 1]
    np.save(fn, p.astype('float64'))
    print(f'  {o} {comp} {time.time() - t:.0f}s peak {peak_mem_gb()}GB', flush=True)
    return p


def base_X(d, med, b):
    X = d[b['num'] + b['cat']].copy()
    X['claims'] = np.log1p(X.claims.astype(float).fillna(med))
    for c in b['cat']: X[c] = X[c].astype(str)
    return X.fillna(0)


# ================================================================ 앙상블·채점
def comp_pred(o, spec, comp):
    if comp.startswith('cls') or comp.startswith('haz'):
        return combine_xgb(o, spec, comp)
    return fit_predict_sk(o, spec, comp)


def ensemble(o, spec):
    ws = spec['ensemble']['weights']
    ps = {k: comp_pred(o, spec, k) for k in ws if ws[k] != 0}
    how = spec['ensemble'].get('how', 'logit')
    if how == 'rank':
        from scipy.stats import rankdata
        z = sum(w * rankdata(ps[k]) / len(ps[k]) for k, w in ws.items() if w != 0)
        return z / sum(ws.values()), ps
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    z = sum(w * lg(ps[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
    return 1 / (1 + np.exp(-z)), ps


_EV = {}


def ev_meta(o):
    if o not in _EV:
        _EV[o] = pd.read_parquet(os.path.join(odir(o), 'ev.parquet'), columns=['pid', 'L', 'ksic', 'acd', 'atype', 'y'])
    return _EV[o]


def score_origin(o, spec):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import spearmanr
    ev = ev_meta(o)
    p, ps = ensemble(o, spec)
    pb = fit_predict_sk(o, spec, 'base')
    y = ev.y.values
    a, ab = roc_auc_score(y, p), roc_auc_score(y, pb)
    sp = []
    for L in ev.L.unique():
        m = (ev.L == L).values
        g = pd.DataFrame({'k': ev.ksic.values[m], 'p': p[m], 'y': y[m]}).groupby('k').mean()
        sp.append(spearmanr(g.p, g.y)[0])
    return dict(auc=float(a), auc_base=float(ab), skill=float((a - ab) / (1 - ab)), prauc=float(average_precision_score(y, p)),
                ind_sp=float(np.mean(sp)), comps={k: round(float(roc_auc_score(y, v)), 4) for k, v in ps.items()}, n=int(len(y)), rate=float(y.mean()))


# ================================================================ 작업 목록
def jobs_for(spec, origins=None):
    """필요한(캐시에 없는) 학습 작업 목록: (origin, comp, bag|None)"""
    out = []
    for o, _, _ in ORIGINS:
        if origins and o not in origins: continue
        for comp, w in spec['ensemble']['weights'].items():
            if w == 0: continue
            key = comp_key(spec, comp)
            if os.path.exists(os.path.join(odir(o), 'pred', f'{key}.npy')): continue
            if comp.startswith('cls') or comp.startswith('haz'):
                for k in range(spec['components'][comp].get('n_bag', 1)):
                    if not os.path.exists(os.path.join(odir(o), 'pred', f'{key}_b{k}.npy')):
                        out.append((o, comp, k))
            else:
                out.append((o, comp, None))
        bkey = 'base_' + hashlib.md5(json.dumps(spec['baseline'], sort_keys=True).encode()).hexdigest()[:10]
        if not os.path.exists(os.path.join(odir(o), 'pred', f'{bkey}.npy')):
            out.append((o, 'base', None))
    # 큰 작업 먼저
    rank = {'O18': 0, 'O16': 1, 'O14': 2, 'O12': 3}
    out.sort(key=lambda j: (0 if (j[1].startswith('cls') or j[1].startswith('haz')) else 1, rank.get(j[0], 9)))
    return out


# ================================================================ 최종시험용 레이블(납부기록 기반)
DATA_END = np.datetime64('2025-08-22')          # KR 등록원부 최종일(README). 납부·등록 사건이 이 날까지 반영
GRACE_DAYS = 183 + 30                           # 추가납부 6개월 + 반영 여유 30일


def label_np(L, pids, data_end=None):
    """y=1: data_end 시점 납부로 본 납부완료일 pth ∈ (L, L+3y], pth < 존속만료, pth+유예 <= data_end, 창 안 말소 없음
       y=0: pth > L+3y, 또는 존속만료까지 납부, 또는 창 안 말소.  나머지(유예기간 미경과) NaN.
    소멸등록(right) 기록을 쓰지 않으므로 등록원부 반영 지연(소멸등록은 실효 뒤 약 321일)에 영향받지 않는다."""
    D = T.load(); pt = D['pt']
    Ld = np.datetime64(L); de = DATA_END if data_end is None else np.datetime64(data_end)
    end = T.add_years(np.array([Ld]), 3)[0]
    pth, _ = T.paid_through(de, asof=de)
    p = pth[pids]; ex = pt.exp.values[pids]
    y = np.full(len(pids), np.nan)
    ml = D['ev']['malso']
    mz = np.isin(pids, ml[(ml.cd > Ld) & (ml.cd <= min(end, de))].pid.unique())
    zero = (p > end) | ((ex <= end) & (p >= ex)) | mz
    one = (p > Ld) & (p <= end) & (p < ex) & (p + np.timedelta64(GRACE_DAYS, 'D') <= de) & ~mz
    y[zero] = 0
    y[one & ~zero] = 1
    return y


# ================================================================ 저장 모형으로 임의 기준일 예측(future 모드 추이용)
def seed_cache_from_r1(L):
    """1라운드 캐시에 같은 기준일 피처가 있으면 복사(피처는 cd<=L 정보만이라 적재 모드와 무관하게 같다)"""
    import shutil
    tag = str(L).replace('-', '')
    for pre in ('feat_', 'soc_', 's2ag_', 's2ow_'):
        src = os.path.join(T4, 'cache', f'{pre}{tag}.parquet'); dst = os.path.join(T.CACHE, f'{pre}{tag}.parquet')
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)


def predict_saved(o, spec, L):
    import xgboost as xgb
    seed_cache_from_r1(L)
    pt_acd = T.load()['pt'].acd.values
    f = T.features(L)
    f = f.assign(acd=pt_acd[f.pid.values]).reset_index(drop=True)
    d = to_str(slim(enrich(f, ak=True)))
    out = pd.DataFrame({'pid': d.pid.values, 'L': L, 'ksic': d.ksic.values, 'atype': d.atype.values, 'acd': d.acd.values})
    import r2ext
    # [실행 오류 수정 2026-09-28] 위험모형 전용 열(lead·j·k·age_d·tier_d)은 eval_decisions 뒤에 생기므로 특허 단위 d 에서 찾지 않는다
    allnum = set().union(*[set(spec['components'][c]['num']) for c in spec['ensemble']['weights'] if not c.startswith('haz')])
    miss = [c for c in allnum if c not in d.columns]
    if miss: d = r2ext.add(d, miss)
    # [메모리 수정 2026-09-28] 위험모형 결정 프레임을 먼저 만들고 원본 피처 f 를 해제(예측값은 동일, 4GB 규칙)
    eh = None
    if any(c.startswith('haz') and w != 0 for c, w in spec['ensemble']['weights'].items()):
        eh = HZ.eval_decisions(f)
        parts = []
        for a in range(0, len(eh), 250_000):                 # 행 단위 보강이라 분할해도 값 동일
            c_ = eh.iloc[a:a + 250_000]
            parts.append(to_str(slim(enrich(c_.drop(columns=['acd']), ak=False))).assign(acd=pt_acd[c_.pid.values]))
            gc.collect()
        eh = pd.concat(parts, ignore_index=True); del parts
    del f; gc.collect()
    ps = {}
    for comp, wt in spec['ensemble']['weights'].items():
        if wt == 0: continue
        key = comp_key(spec, comp); cfg = spec['components'][comp]
        if comp.startswith('cls') or comp.startswith('haz'):
            e = d
            if comp.startswith('haz'):
                e = eh
                hm = [c for c in cfg['num'] if c not in e.columns]
                if hm: e = r2ext.add(e, hm)
            zs = []
            for k in range(cfg.get('n_bag', 1)):
                b = xgb.Booster(); b.load_model(os.path.join(mdir(o), f'{key}_b{k}.json'))
                mm = pickle.load(open(os.path.join(mdir(o), f'{key}_b{k}.pkl'), 'rb'))
                X = e[mm['num'] + CAT].copy()
                for c in mm['num']:
                    if X[c].dtype != np.float32: X[c] = X[c].astype('float32')
                X, _ = _prep_cat(X, mm['cmap'])
                p = np.clip(b.predict(xgb.DMatrix(X, enable_categorical=True)), 1e-6, 1 - 1e-6)
                zs.append(np.log(p / (1 - p)))
            p = 1 / (1 + np.exp(-np.mean(zs, 0)))
            if comp.startswith('haz'):
                p = HZ.combine(e, p, len(d))
        else:
            mm = pickle.load(open(os.path.join(mdir(o), f'{key}.pkl'), 'rb'))
            if comp.startswith('logit'):
                p = mm['m'].predict_proba(mm['prep'].transform(d))[:, 1]
            else:
                ce = mm['oe'].transform(d[CAT].astype(str)).astype('float32'); ce[ce < 0] = np.nan
                Xe = np.c_[d[cfg['num']].to_numpy(dtype='float32', na_value=np.nan), ce]; Xe[:, mm.get('allnan', [])] = 0
                p = mm['m'].predict_proba(Xe)[:, 1]
        ps[comp] = p; out['p_' + comp] = p
    ws = spec['ensemble']['weights']
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if spec['ensemble'].get('how', 'logit') == 'rank':
        from scipy.stats import rankdata
        out['p'] = sum(w * rankdata(ps[k]) / len(ps[k]) for k, w in ws.items() if w != 0) / sum(ws.values())
    else:
        z = sum(w * lg(ps[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
        out['p'] = 1 / (1 + np.exp(-z))
    bkey = 'base_' + hashlib.md5(json.dumps(spec['baseline'], sort_keys=True).encode()).hexdigest()[:10]
    mm = pickle.load(open(os.path.join(mdir(o), f'{bkey}.pkl'), 'rb'))
    out['p_base'] = mm['m'].predict_proba(base_X(d, mm['med'], spec['baseline']))[:, 1]
    return out
