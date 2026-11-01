# T2 공용 평가 모듈 — 검증 원점 2012–2015, h=1..3, 목표연도 ≤2018 (SPLITS.md 고정)
# 패널은 build_panel.py 에서 출원연도 2000–2018 만 집계됨 (🔒 2019–2021 미계산)
import os, csv, datetime
import numpy as np, pandas as pd

T = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(T, 'cache')
Y0, Y1 = 2000, 2018
YEARS = np.arange(Y0, Y1 + 1)
CCS = ['CN', 'EP', 'JP', 'KR', 'US']
ORIGINS = [2012, 2013, 2014, 2015]
H = 3
assert max(ORIGINS) + H <= 2018


def load_panel(col='n'):
    d = pd.read_parquet(os.path.join(CACHE, 'panel_kc.parquet'))
    assert d.year.max() <= 2018
    ks = sorted(d.ksic.unique())
    idx = pd.MultiIndex.from_product([ks, CCS], names=['ksic', 'cc'])
    w = d.pivot_table(index=['ksic', 'cc'], columns='year', values=col, aggfunc='sum')
    w = w.reindex(idx).reindex(columns=YEARS).fillna(0.0)
    return w  # 250 x 19 DataFrame


def load_all():
    d = pd.read_parquet(os.path.join(CACHE, 'panel_kc.parquet'))
    ks = sorted(d.ksic.unique())
    idx = pd.MultiIndex.from_product([ks, CCS], names=['ksic', 'cc'])
    out = {}
    for c in ['n', 'n_um', 'n_fprio', 'n_prio', 'n_newapp', 'n_app']:
        w = d.pivot_table(index=['ksic', 'cc'], columns='year', values=c, aggfunc='sum')
        out[c] = w.reindex(idx).reindex(columns=YEARS).fillna(0.0)
    return out


def col(year):
    return int(year - Y0)


def evaluate(forecast_fn, Yw=None, verbose=True, **kw):
    """forecast_fn(hist: ndarray (250, t) up to origin inclusive, origin, **kw) -> ndarray (250, H)
    반환: 평가 DataFrame (series x origin x h)"""
    if Yw is None:
        Yw = load_panel()
    Y = Yw.values.astype(float)
    rows = []
    for o in ORIGINS:
        hist = Y[:, :col(o) + 1].copy()
        F = np.asarray(forecast_fn(hist, o, **kw), dtype=float)
        assert F.shape == (Y.shape[0], H)
        # MASE 척도: 학습구간(2000..o) 1-step naive 평균절대차
        scale = np.mean(np.abs(np.diff(hist, axis=1)), axis=1)
        scale = np.where(scale <= 0, 1.0, scale)
        for h in range(1, H + 1):
            yt = Y[:, col(o + h)]
            for i in range(Y.shape[0]):
                rows.append((Yw.index[i][0], Yw.index[i][1], o, h, o + h, yt[i], F[i, h - 1],
                             hist[i, -1], scale[i]))
    r = pd.DataFrame(rows, columns=['ksic', 'cc', 'origin', 'h', 'target', 'y', 'f', 'ylast', 'scale'])
    r['ae'] = (r.y - r.f).abs()
    r['ase'] = r.ae / r.scale
    return r


def summarize(r):
    out = {'MASE': r.ase.mean(), 'WAPE': r.ae.sum() / r.y.abs().sum()}
    for h in range(1, H + 1):
        out[f'MASE_h{h}'] = r[r.h == h].ase.mean()
    return out


_NAIVE = None


def naive_fn(hist, o):
    return np.repeat(hist[:, -1:], H, axis=1)


def naive_ref():
    global _NAIVE
    if _NAIVE is None:
        _NAIVE = evaluate(naive_fn)
    return _NAIVE


def report(r, name='', seg=True):
    s = summarize(r)
    b = summarize(naive_ref())
    skill = 1 - s['MASE'] / b['MASE']
    print(f"[{name}] MASE={s['MASE']:.4f} (naive {b['MASE']:.4f}) skill={skill:.4f} WAPE={s['WAPE']:.4f} "
          + ' '.join(f"h{h}={s[f'MASE_h{h}']:.3f}" for h in range(1, H + 1)))
    if seg:
        nb = naive_ref()
        for g in ['cc', 'h', 'origin']:
            a = r.groupby(g).ase.mean(); bb = nb.groupby(g).ase.mean()
            print('  by', g, ' '.join(f"{k}:{a[k]:.3f}/{bb[k]:.3f}" for k in a.index))
    return s, b, skill


def log_leaderboard(version, change, hypothesis, value, baseline_value, kept, note, metric='MASE(val 2012-15,h1-3)'):
    p = os.path.join(T, 'leaderboard.csv')
    new = not os.path.exists(p)
    skill = 1 - value / baseline_value
    with open(p, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['version', 'date', 'change', 'hypothesis', 'metric', 'value', 'baseline_value', 'skill', 'kept', 'note'])
        w.writerow([version, datetime.date.today().isoformat(), change, hypothesis, metric,
                    f'{value:.4f}', f'{baseline_value:.4f}', f'{skill:.4f}', kept, note])


def save_pred(r, name):
    os.makedirs(os.path.join(T, 'preds'), exist_ok=True)
    r.to_parquet(os.path.join(T, 'preds', f'{name}.parquet'))
