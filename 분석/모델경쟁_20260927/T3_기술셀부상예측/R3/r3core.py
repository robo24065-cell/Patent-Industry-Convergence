"""T3 3라운드 실험 코어 (2라운드 r2core 사본: 경로·변형 스위치·기록 형식만 변경, 모형 코드는 동일).
변형 T3R3_VAR=app(출원연도, 개발 원점 2010–2016, F=2020) | pub(공개연도, 개발 원점 2010–2018, 🔒 F'=2022).
"""
import os, sys, json, hashlib, time, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import r3lib as L
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import nbinom, rankdata, spearmanr
import xgboost as xgb

DEV_T0 = L.DEV_T0      # 개발 원점 (결과창 t0+1..t0+3 < F)
FIRST_T0 = 2005
MIN_TOTAL = 50; TOP = 0.10
COMP_DIR = os.path.join(HERE, f'comp_{L.VAR}'); os.makedirs(COMP_DIR, exist_ok=True)
EXP_CSV = os.path.join(HERE, 'experiments.csv')
CHAMP_JSON = os.path.join(HERE, f'champion_{L.VAR}.json')

# ---------------- 레이블 (1라운드 중립 동률깨기 그대로) ----------------
def hkey(cells):
    return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cells], dtype=np.int64)
def breakout_neutral(y, tie, key, top=TOP):
    k = max(1, int(round(top * len(y)))); o = np.lexsort((key, -tie, -y))
    yb = np.zeros(len(y), dtype=int); yb[o[:k]] = 1; return yb
def add_labels(df):
    d = df[df.tot_W >= MIN_TOTAL].copy(); d['yb'] = 0
    for t, g in d.groupby('t0'):
        if 'y' not in g or g.y.isna().all(): continue
        d.loc[g.index, 'yb'] = breakout_neutral(g.y.values, (g.kr_fut - g.kr_W).values, hkey(g.cell.values))
    return d.reset_index(drop=True)

# ---------------- 파생 피처 (1라운드 enrich 동결 사본) ----------------
def enrich(D):
    P = L.load(); kr = P['A']['n'][:, 0, :]; tot = P['A']['n'].sum(1)
    f2 = pd.read_parquet(os.path.join(L.CACHE, 'feat2.parquet')).drop(columns=['sub'])
    n0 = len(D); D = D.merge(f2, on=['ksic', 'mg', 't0'], how='left'); assert len(D) == n0
    idx = np.searchsorted(P['cells'], D.cell.values)
    long = np.full(len(D), np.nan); dd = np.full(len(D), np.nan); tl = np.full(len(D), np.nan)
    for t0, g in D.groupby('t0'):
        a = max(L.Y0, t0 - 8); ii = idx[g.index]
        long[g.index] = np.log1p(kr[ii, a - L.Y0:t0 - 3 - L.Y0 + 1].mean(1))
        dd[g.index] = np.log1p(kr[ii, a - L.Y0:t0 - L.Y0 + 1].max(1)) - np.log1p(kr[ii, t0 - 2 - L.Y0:t0 - L.Y0 + 1].mean(1))
        tl[g.index] = np.log1p(tot[ii, a - L.Y0:t0 - 3 - L.Y0 + 1].mean(1))
    D['kr_long'] = long; D['kr_drawdown'] = dd; D['kr_vs_long'] = D.kr_lvl - np.log(3) - D.kr_long
    D['tot_vs_long'] = D.tot_lvl - np.log(3) - tl
    D['kr_share_rel_ksic'] = D.kr_share - D.groupby(['t0', 'ksic']).kr_share.transform('median')
    D['kr_share_rel_sub'] = D.kr_share - D.sub_krsh
    D['kr_share_rel_mg'] = D.kr_share - D.groupby(['t0', 'mg']).kr_share.transform('median')
    return D

# ---------------- 2라운드 신규 파생 피처 (모두 t0 이하 자료) ----------------
MOMS = ['kr_mom', 'kr_mom_l1', 'kr_mom_l2', 'US_mom', 'CN_mom', 'JP_mom', 'EP_mom', 'fx_mom', 'fx_mom_l1', 'tot_mom',
        'gap_fx_kr', 'ksic_kr_mom', 'mg_kr_mom', 'mg_tot_mom', 'mg_fx_mom', 'ksic_tot_mom', 'nb_kmom', 'nb_tmom', 'sub_kmom', 'sub_fmom']
def extras(D):
    P = L.load(); n = P['A']['n']; kr = n[:, 0, :]; tot = n.sum(1)
    idx = np.searchsorted(P['cells'], D.cell.values)
    # (a) 원점 단면 중앙값 차감(전역 경기순환 제거) 모멘텀
    for c in MOMS:
        D[c + '_dm'] = D[c] - D.groupby('t0')[c].transform('median')
    # (b) IPC 신설코드 보정: 셀(5개국)·메인그룹의 첫 출원연도로부터 경과연수 (2000 이전은 절단 → 상한)
    age = np.full(len(D), np.nan); mage = np.full(len(D), np.nan); kage = np.full(len(D), np.nan)
    totmg = pd.DataFrame(tot).groupby(P['mg']).sum()
    mg_first = {}
    for t0, g in D.groupby('t0'):
        ii = idx[g.index]; T = t0 - L.Y0 + 1
        pos = tot[ii, :T] > 0; fy = np.where(pos.any(1), pos.argmax(1), T)
        age[g.index] = np.minimum(t0 - (L.Y0 + fy), 10)
        pk = kr[ii, :T] > 0; fk = np.where(pk.any(1), pk.argmax(1), T)
        kage[g.index] = np.where(pk.any(1), np.minimum(t0 - (L.Y0 + fk), 10), -1)
        M = totmg.values[:, :T] > 0; fm = np.where(M.any(1), M.argmax(1), T)
        s = pd.Series(np.minimum(t0 - (L.Y0 + fm), 10), index=totmg.index)
        mage[g.index] = s.reindex(g.mg.values).values
    D['cell_age'] = age; D['mg_age'] = mage; D['kr_age'] = kage
    # (c) 선행 시차: 해외 4개국 동일셀의 시차 3·4 성장, 4개국 합 시차 3
    fx = n[:, 1:, :].sum(1)
    for t0, g in D.groupby('t0'):
        ii = idx[g.index]
        for lag in (3, 4):
            v = L._g(fx, t0, lag)
            D.loc[g.index, f'fx_mom_l{lag}'] = v[ii] if not np.isscalar(v) else np.nan
        for j, c in enumerate(L.CCS[1:], start=1):
            v = L._g(n[:, j, :], t0, 3); D.loc[g.index, f'{c}_mom_l3'] = v[ii]
    # (d) KR 0건 셀 지시자·해외 규모 대비 KR 부재
    D['kr_zero'] = (D.kr_W == 0).astype(int)
    D['fx_lvl_krzero'] = np.where(D.kr_W == 0, np.log1p(D.tot_W), 0.0)
    return D

# ---------------- 외부 데이터 (KSIC 연결) ----------------
EXT_SUFFIX = os.environ.get('T3R3_EXT_SUFFIX', '')
EXT_FILES = ([os.path.join(HERE, '..', 'R2', 'ext_ksic_t0.parquet')] if L.VAR == 'app' else []) + [os.path.join(HERE, f'ext_ksic_t0_{L.VAR}{EXT_SUFFIX}.parquet')]
def add_ext(D):
    for fn in EXT_FILES:
        if not os.path.exists(fn): continue
        E = pd.read_parquet(fn); E = E[['ksic', 't0'] + [c for c in E.columns if c not in D.columns and c not in ('ksic', 't0')]]
        n0 = len(D); D = D.merge(E, on=['ksic', 't0'], how='left'); assert len(D) == n0
    return D
CELL_EXT_FILE = os.path.join(HERE, f'ext_cell_t0_{L.VAR}{EXT_SUFFIX}.parquet')
def add_cell_ext(D):
    if not os.path.exists(CELL_EXT_FILE): return D
    E = pd.read_parquet(CELL_EXT_FILE)
    n0 = len(D); D = D.merge(E, on=['cell', 't0'], how='left'); assert len(D) == n0
    return D

KS_FEATS = ['ksic_kr_mom', 'ksic_tot_mom', 'pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2', 'pi_inv_ship', 'pi_util', 'pi_util_chg',
            'io_up_kmom', 'io_dn_kmom', 'io_up_fmom', 'io_dn_fmom', 'tr_imp_g', 'tr_cn_sh', 'tr_exp_g', 'tr_net_exp']
def add_ksic_hier(D, feats=KS_FEATS, alpha=10.0):
    """계층 결합: KSIC×원점 부상률을 KSIC 수준 피처(특허 모멘텀·생산지수·산업연관·무역)로 예측하는 릿지.
    원점 t 의 예측값은 결과창이 t 이전에 끝난 원점(≤t-3)의 KSIC 부상률만으로 학습 → 인과적(누수 없음)."""
    from sklearn.linear_model import Ridge
    fs = [f for f in feats if f in D.columns]
    K = D.groupby(['t0', 'ksic']).agg(rate=('yb', 'mean'), **{f: (f, 'first') for f in fs}).reset_index()
    K['ksic_hier'] = np.nan
    for t in sorted(K.t0.unique()):
        trk = K[(K.t0 <= t - 3) & K.rate.notna()]
        if trk.t0.nunique() < 3: continue
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))
        m.fit(trk[fs].values, trk.rate.values)
        K.loc[K.t0 == t, 'ksic_hier'] = m.predict(K.loc[K.t0 == t, fs].values)
    n0 = len(D); D = D.drop(columns=[c for c in ['ksic_hier'] if c in D]).merge(K[['t0', 'ksic', 'ksic_hier']], on=['t0', 'ksic'], how='left')
    assert len(D) == n0; return D

def _cell_arr(df, col, cc=None, year='year'):
    P = L.load(); T = L.Y1 - L.Y0 + 1
    d = df if cc is None else df[df.cc == cc]
    d = d.assign(cell=d.ksic + '|' + d.mg); d = d[d.cell.isin(set(P['cells'])) & (d[year] >= L.Y0) & (d[year] <= L.Y1)]
    g = d.groupby(['cell', year])[col].sum().reset_index()
    arr = np.zeros((len(P['cells']), T), dtype=np.float32)
    arr[np.searchsorted(P['cells'], g.cell.values), g[year].values - L.Y0] = g[col].values
    return arr
_R2P = {}
def add_r2panel(D):
    """출원인 유형(대학·공공연구기관·KR 개인) 비중과 KR 등록 속도 (t0 이하 출원·등록만)."""
    fn = os.path.join(L.CACHE, 'feat_r2_panel.parquet'); fr = os.path.join(L.CACHE, 'kr_reg_panel.parquet')
    if not (os.path.exists(fn) and os.path.exists(fr)): return D
    P = L.load(); n = P['A']['n']; kr = n[:, 0, :]; fx = n[:, 1:, :].sum(1)
    if 'A' not in _R2P:
        f = pd.read_parquet(fn); r = pd.read_parquet(fr).dropna(subset=['mg'])
        A = dict(ku=_cell_arr(f, 'n_univ', 'KR'), kg=_cell_arr(f, 'n_gri', 'KR'), ki=_cell_arr(f, 'n_indiv', 'KR'),
                 fu=sum(_cell_arr(f, 'n_univ', c) for c in ['US', 'JP', 'EP', 'CN']))
        r['cell'] = r.ksic + '|' + r.mg; r = r[r.cell.isin(set(P['cells']))]
        _R2P['A'] = A; _R2P['reg'] = r
    A = _R2P['A']; r = _R2P['reg']
    idx = np.searchsorted(P['cells'], D.cell.values)
    cols = ['kr_univ_sh', 'kr_gri_sh', 'kr_pub_mom', 'kr_indiv_sh', 'kr_indiv_sh_chg', 'fx_univ_sh', 'kr_reg_rate', 'kr_reg_fast', 'kr_reg_mom']
    for c in cols: D[c] = np.nan
    W = lambda x, t, a, b: L.wsum(x, t, a, b)
    for t0, g in D.groupby('t0'):
        ii = idx[g.index]; kw = W(kr, t0, -2, 0)[ii]; fw = W(fx, t0, -2, 0)[ii]
        D.loc[g.index, 'kr_univ_sh'] = W(A['ku'], t0, -2, 0)[ii] / np.maximum(kw, 1)
        D.loc[g.index, 'kr_gri_sh'] = W(A['kg'], t0, -2, 0)[ii] / np.maximum(kw, 1)
        pub = A['ku'] + A['kg']
        if t0 - 5 >= L.Y0: D.loc[g.index, 'kr_pub_mom'] = np.log((W(pub, t0, -2, 0)[ii] + 1) / (W(pub, t0, -5, -3)[ii] + 1))
        ish = W(A['ki'], t0, -2, 0)[ii] / np.maximum(kw, 1); D.loc[g.index, 'kr_indiv_sh'] = ish
        if t0 - 5 >= L.Y0: D.loc[g.index, 'kr_indiv_sh_chg'] = ish - W(A['ki'], t0, -5, -3)[ii] / np.maximum(W(kr, t0, -5, -3)[ii], 1)
        D.loc[g.index, 'fx_univ_sh'] = W(A['fu'], t0, -2, 0)[ii] / np.maximum(fw, 1)
        # 등록: t0 까지 등록된 것만 (rgst_year ≤ t0)
        rr = r[(r.appl_year >= t0 - 6) & (r.appl_year <= t0 - 3)]
        tot_ = rr.groupby('cell').n.sum(); reg_ = rr[rr.rgst_year <= t0].groupby('cell').n.sum()
        D.loc[g.index, 'kr_reg_rate'] = (reg_.reindex(g.cell.values).fillna(0).values + 0.5) / (tot_.reindex(g.cell.values).fillna(0).values + 1)
        rf = r[(r.appl_year >= t0 - 3) & (r.appl_year <= t0 - 1)]
        tf = rf.groupby('cell').n.sum(); gf = rf[rf.rgst_year <= t0].groupby('cell').n.sum()
        D.loc[g.index, 'kr_reg_fast'] = (gf.reindex(g.cell.values).fillna(0).values + 0.5) / (tf.reindex(g.cell.values).fillna(0).values + 1)
        rn = r[r.rgst_year.notna()]
        a0 = rn[(rn.rgst_year >= t0 - 2) & (rn.rgst_year <= t0)].groupby('cell').n.sum(); a1 = rn[(rn.rgst_year >= t0 - 5) & (rn.rgst_year <= t0 - 3)].groupby('cell').n.sum()
        D.loc[g.index, 'kr_reg_mom'] = np.log((a0.reindex(g.cell.values).fillna(0).values + 1) / (a1.reindex(g.cell.values).fillna(0).values + 1))
    return D

def add_word(D):
    fn = os.path.join(L.CACHE, 'word_expo.parquet')
    if not os.path.exists(fn): return D
    W = pd.read_parquet(fn); n0 = len(D); D = D.merge(W, on=['ksic', 'mg', 't0'], how='left'); assert len(D) == n0; return D

_D = {}
DATA_FIRST_T0 = int(os.environ.get('T3R3_DATA_FIRST', '2005'))
def get_D(t0s=None, refresh=False):
    t0s = range(DATA_FIRST_T0, max(DEV_T0) + 1) if t0s is None else t0s
    key = tuple(t0s)
    if key in _D and not refresh: return _D[key]
    df = pd.concat([L.get_feats(t) for t in t0s], ignore_index=True)
    d = add_labels(df); d = enrich(d).reset_index(drop=True); d = extras(d); d = add_ext(d); d = add_cell_ext(d); d = d if os.environ.get('T3R3_HIER') != '1' else add_ksic_hier(d); d = add_r2panel(d); d = add_word(d)
    if 'pi_prod_g2' in d:   # 수요-특허 괴리: 원점 내 z(생산 2년 성장) − z(KSIC KR 특허 모멘텀)
        z = lambda c: (d[c] - d.groupby('t0')[c].transform('mean')) / d.groupby('t0')[c].transform('std')
        d['pi_gap'] = z('pi_prod_g2') - z('ksic_kr_mom')
    if 'ks_ship_g' in d and 'pi_ship_g2' in d:   # 산업 성장 최선값: KOSIS 5자리 출하 연율 성장(2012~) 없으면 생산지수 3자리 출하 2년 성장 연율
        d['ind_ship_best'] = d.ks_ship_g.fillna(d.pi_ship_g2 / 2)
    _D[key] = d; return d

# ---------------- 성분 모형 ----------------
def _w(tr, half):
    if not half: return None
    return (0.5 ** ((tr.t0.max() - tr.t0) / half)).values
def comp_logit(c, tr, te):
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=c.get('C', 1.0), max_iter=3000))
    w = _w(tr, c.get('half'))
    kw = {} if w is None else {'logisticregression__sample_weight': w}
    m.fit(tr[c['features']].values, tr.yb.values, **kw); return m.predict_proba(te[c['features']].values)[:, 1]
def comp_xgbc(c, tr, te):
    ps = []
    for s in range(c.get('seeds', 1)):
        p = dict(c['params']); p['seed'] = s
        d = xgb.DMatrix(tr[c['features']].values, label=tr.yb.values, weight=_w(tr, c.get('half')))
        m = xgb.train(p, d, c['rounds']); ps.append(m.predict(xgb.DMatrix(te[c['features']].values)))
    return np.mean([rankdata(p) for p in ps], 0) if len(ps) > 1 else ps[0]
def fit_alpha(F, mu):
    best = None
    for a in np.exp(np.linspace(np.log(0.01), np.log(5), 60)):
        r = 1 / a; ll = nbinom.logpmf(F, r, r / (r + mu)).sum()
        if best is None or ll > best[0]: best = (ll, a)
    return best[1]
def comp_nb(c, tr, te):
    outs = []
    for s in range(c.get('seeds', 1)):
        p = dict(c['params']); p['seed'] = s
        d = xgb.DMatrix(tr[c['features']].values, label=tr.kr_fut.values, weight=_w(tr, c.get('half')))
        m = xgb.train(p, d, c['rounds'])
        a = fit_alpha(tr.kr_fut.values, m.predict(d)); mu = m.predict(xgb.DMatrix(te[c['features']].values))
        trq = tr[tr['_elig'] == 1] if '_elig' in tr else tr
        tau = float(np.median([np.quantile(g.y.values, 1 - TOP) for _, g in trq.groupby('t0')]))
        need = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9); r = 1 / a
        outs.append(nbinom.sf(need - 1, r, r / (r + mu)))
    return np.mean([rankdata(o) for o in outs], 0) if len(outs) > 1 else outs[0]
def comp_logit_spline(c, tr, te):
    from sklearn.preprocessing import SplineTransformer
    from sklearn.compose import ColumnTransformer
    fs = c['features']; sp = [fs.index(f) for f in c['spline'] if f in fs]; rest = [i for i in range(len(fs)) if i not in sp]
    ct = ColumnTransformer([('sp', SplineTransformer(n_knots=c.get('knots', 5), degree=3), sp), ('lin', 'passthrough', rest)])
    m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), ct, StandardScaler(), LogisticRegression(C=c.get('C', 1.0), max_iter=5000))
    m.fit(tr[fs].values, tr.yb.values); return m.predict_proba(te[fs].values)[:, 1]

def comp_xgbreg(c, tr, te):
    p = dict(c['params']); d = xgb.DMatrix(tr[c['features']].values, label=tr.y.values, weight=_w(tr, c.get('half')))
    m = xgb.train(p, d, c['rounds']); return m.predict(xgb.DMatrix(te[c['features']].values))
def comp_rf(c, tr, te):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    K = ExtraTreesClassifier if c.get('extra') else RandomForestClassifier
    X = tr[c['features']].values; Xt = te[c['features']].values
    if c.get('extra'):
        med = np.nanmedian(X, 0); X = np.where(np.isnan(X), med, X); Xt = np.where(np.isnan(Xt), med, Xt)
    m = K(n_estimators=c.get('n', 300), min_samples_leaf=c.get('leaf', 20), max_features=c.get('mf', 0.3), n_jobs=2, random_state=0)
    m.fit(X, tr.yb.values, sample_weight=_w(tr, c.get('half'))); return m.predict_proba(Xt)[:, 1]
COMP = dict(logit_spline=comp_logit_spline, logit=comp_logit, xgbc=comp_xgbc, nb=comp_nb, xgbreg=comp_xgbreg, rf=comp_rf)

TRAIN_MAX_T0 = int(os.environ.get('T3R3_TRAIN_MAX_T0', '9999'))   # 🔒 final: 2016 (결과창 ≤2019 < F=2020), future: 2019
def train_rows(D, v, cfg):
    tr = D[(D.t0 <= min(v - 3, TRAIN_MAX_T0)) & (D.t0 >= cfg.get('first_t0', FIRST_T0))]
    if cfg.get('window'): tr = tr[tr.t0 >= v - 3 - cfg['window'] + 1]
    assert len(tr) and tr.t0.max() + 3 <= v
    return tr

_WIDE = {}
def get_wide(D, min_total):
    """음이항 성분 확대 학습용: 5개국 t0-2..t0 합 ≥min_total 셀(적격 ≥50 밖 포함). kr_fut·y 는 셀 자체 값(레이블 yb 불필요)."""
    if min_total in _WIDE: return _WIDE[min_total]
    df = pd.concat([L.get_feats(t) for t in sorted(D.t0.unique())], ignore_index=True)
    w = df[(df.tot_W >= min_total) & (df.tot_W < MIN_TOTAL)].copy(); w['yb'] = 0
    w = enrich(w.reset_index(drop=True)); w = extras(w); w = add_ext(w); w = add_cell_ext(w); w = add_r2panel(w); w = add_word(w)
    if 'ksic_hier' in D and os.environ.get('T3R3_HIER') == '1': w = w.merge(D[['t0', 'ksic', 'ksic_hier']].drop_duplicates(), on=['t0', 'ksic'], how='left')
    if 'ks_ship_g' in w and 'pi_ship_g2' in w: w['ind_ship_best'] = w.ks_ship_g.fillna(w.pi_ship_g2 / 2)
    _WIDE[min_total] = w; return w

def comp_pred(c, D, v, cfg, tag='dev'):
    sub = D.loc[D.t0 <= v, c['features'] + ['yb', 'kr_fut', 'y', 'kr_W', 't0']]
    dsig = hashlib.md5(np.ascontiguousarray(sub.values.astype(np.float64)).tobytes()).hexdigest()[:12]   # 데이터 서명(외부자료 갱신 시 캐시 무효화)
    key = hashlib.md5(json.dumps(dict(c=c, v=v, first=cfg.get('first_t0', FIRST_T0), win=cfg.get('window'), tag=tag, d=dsig),
                                 sort_keys=True).encode()).hexdigest()[:16]
    fn = os.path.join(COMP_DIR, f'{key}.npy')
    te = D[D.t0 == v]
    if os.path.exists(fn):
        p = np.load(fn)
        if len(p) == len(te): return p
    tr = train_rows(D, v, cfg)
    if c.get('train_min_total'):   # 음이항: 확대 표본 추가 (τ 는 적격 표본 y 분위로 유지)
        W = get_wide(D, c['train_min_total']); W = train_rows(W, v, cfg)
        if c['type'] != 'nb':   # 분류 성분: 확대 셀의 레이블 = 같은 원점 적격 표본의 부상 경계(y 하한) 이상이면 1
            thr = D[D.yb == 1].groupby('t0').y.min(); W = W.copy(); W['yb'] = (W.y >= W.t0.map(thr)).astype(int)
        tr = pd.concat([tr, W[tr.columns.intersection(W.columns)]], ignore_index=True)
        tr['_elig'] = np.r_[np.ones(len(tr) - len(W)), np.zeros(len(W))]
    p = np.asarray(COMP[c['type']](c, tr, te), dtype=float); np.save(fn, p); return p

def score(cfg, D, v, tag='dev'):
    comps = cfg['components']; w = cfg.get('weights', [1] * len(comps))
    R = [rankdata(comp_pred(c, D, v, cfg, tag)) / (D.t0 == v).sum() for c in comps]
    return np.average(R, axis=0, weights=w)

def baseline_pred(D, v):
    tr = D[(D.t0 <= min(v - 3, TRAIN_MAX_T0)) & (D.t0 >= FIRST_T0)]; te = D[D.t0 == v]
    assert tr.t0.max() + 3 <= v
    m = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values)
    return m.predict_proba(te[['kr_mom']].values)[:, 1]

def metrics(y, yb, p):
    k = max(1, int(round(0.10 * len(y)))); top = np.argsort(-p)[:k]; pr = average_precision_score(yb, p)
    return dict(auc=float(roc_auc_score(yb, p)), prauc=float(pr), p_at10=float(yb[top].mean()),
                rho=float(spearmanr(y, p).statistic), n=int(len(y)), n_evt=int(yb.sum()))

_BASE = {}
def evaluate(cfg, D=None, origins=DEV_T0):
    D = get_D() if D is None else D
    out = {}
    for v in origins:
        te = D[D.t0 == v]
        p = score(cfg, D, v)
        if v not in _BASE: _BASE[v] = metrics(te.y.values, te.yb.values, baseline_pred(D, v))
        m = metrics(te.y.values, te.yb.values, p); b = _BASE[v]
        m['base_auc'] = b['auc']; m['skill'] = (m['auc'] - b['auc']) / (1 - b['auc'])
        out[v] = m
    return out

def summarize(res):
    sk = np.array([res[v]['skill'] for v in res]); au = np.array([res[v]['auc'] for v in res])
    return dict(mean_skill=float(sk.mean()), mean_auc=float(au.mean()),
                mean_base_auc=float(np.mean([res[v]['base_auc'] for v in res])),
                mean_prauc=float(np.mean([res[v]['prauc'] for v in res])),
                mean_p10=float(np.mean([res[v]['p_at10'] for v in res])),
                mean_rho=float(np.mean([res[v]['rho'] for v in res])))

# ---------------- 챔피언/도전자 ----------------
def load_champ():
    return json.load(open(CHAMP_JSON, encoding='utf-8')) if os.path.exists(CHAMP_JSON) else None
def save_champ(exp_id, cfg, res):
    json.dump(dict(exp_id=exp_id, cfg=cfg, res={str(k): v for k, v in res.items()}, **summarize(res)),
              open(CHAMP_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

def record(exp_id, parent, change, res, decision, note='', delta=np.nan, win=np.nan):
    s = summarize(res)
    row = dict(exp_id=exp_id, variant=L.VAR, parent=parent, change=change)
    for v in res: row[f'auc_{v}'] = round(res[v]['auc'], 4)
    for v in res: row[f'skill_{v}'] = round(res[v]['skill'], 4)
    row.update(mean_auc=round(s['mean_auc'], 4), mean_base_auc=round(s['mean_base_auc'], 4), mean_skill=round(s['mean_skill'], 4),
               delta_vs_champ=(round(delta, 4) if delta == delta else ''), win_rate=(round(win, 3) if win == win else ''),
               mean_prauc=round(s['mean_prauc'], 4), mean_p10=round(s['mean_p10'], 4), mean_rho=round(s['mean_rho'], 4),
               decision=decision, note=note, time=time.strftime('%Y-%m-%d %H:%M'))
    lock = EXP_CSV + '.lock'
    for _ in range(600):
        try: fd = os.open(lock, os.O_CREAT | os.O_EXCL); os.close(fd); break
        except FileExistsError: time.sleep(0.5)
    try:
        df = pd.read_csv(EXP_CSV, encoding='utf-8-sig') if os.path.exists(EXP_CSV) else pd.DataFrame()
        if len(df): df = df[~((df.exp_id == exp_id) & (df.variant == L.VAR))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        oc = sorted([c for c in df.columns if c.startswith('auc_20')]) + sorted([c for c in df.columns if c.startswith('skill_20')])
        head = ['exp_id', 'variant', 'parent', 'change']; rest = [c for c in df.columns if c not in head + oc]
        df[head + oc + rest].to_csv(EXP_CSV, index=False, encoding='utf-8-sig')
    finally:
        os.remove(lock)

def challenge(exp_id, change, cfg, note='', decide=True, D=None):
    """챔피언 대비 채택 판단: 평균 skill +0.003 이상 AND 원점 60% 이상 우세."""
    ch = load_champ(); t = time.time()
    res = evaluate(cfg, D)
    s = summarize(res)
    cres = {int(k): v for k, v in ch['res'].items()}
    delta = s['mean_skill'] - ch['mean_skill']
    wins = np.mean([res[v]['skill'] > cres[v]['skill'] for v in res])
    ok = (delta >= 0.003) and (wins >= 0.6)
    dec = ('ACCEPT' if ok else 'ROLLBACK') if decide else 'INFO'
    record(exp_id, ch['exp_id'], change, res, dec, note, delta, wins)
    print(f"[{exp_id}] {change[:70]} | skill {s['mean_skill']:.4f} (챔피언 {ch['mean_skill']:.4f}, Δ{delta:+.4f}) 승률 {wins:.2f} → {dec} "
          f"| AUC {s['mean_auc']:.4f} | {' '.join(f'{res[v]['skill']:.3f}' for v in res)} | {time.time()-t:.0f}s", flush=True)
    if dec == 'ACCEPT': save_champ(exp_id, cfg, res)
    return dec, res, delta, wins

# ---------------- 1라운드 동결 모형 v30 ----------------
def v30_cfg():
    spec = json.load(open(os.path.join(HERE, '..', 'final_spec.json'), encoding='utf-8'))['model']
    return dict(components=[
        dict(type='logit', features=spec['logit']['features'], C=spec['logit']['C']),
        dict(type='xgbc', features=spec['xgb_clf']['features'], params=spec['xgb_clf']['params'], rounds=spec['xgb_clf']['rounds']),
        dict(type='nb', features=spec['nb_struct']['features'], params=spec['nb_struct']['params'], rounds=spec['nb_struct']['rounds'])],
        weights=[1, 1, 1], first_t0=FIRST_T0)

def with_feats(cfg, add=(), drop=(), comps=('logit', 'xgbc', 'nb')):
    c2 = copy.deepcopy(cfg)
    for c in c2['components']:
        if c['type'] in comps:
            c['features'] = [f for f in c['features'] if f not in drop] + [f for f in add if f not in c['features']]
    return c2


def rescore(exp_id, change, cfg, note='', D=None, parent='-'):
    """챔피언 설정 재채점(기록 + 이 변형의 챔피언으로 저장)."""
    t = time.time(); res = evaluate(cfg, D); s = summarize(res)
    record(exp_id, parent, change, res, 'CHAMPION', note); save_champ(exp_id, cfg, res)
    print(f"[{exp_id}] {L.VAR} 재채점 skill {s['mean_skill']:.4f} AUC {s['mean_auc']:.4f} 기준선 {s['mean_base_auc']:.4f} | "
          f"{' '.join(f'{v}:{res[v]['skill']:.3f}' for v in res)} | {time.time()-t:.0f}s", flush=True)
    return res
