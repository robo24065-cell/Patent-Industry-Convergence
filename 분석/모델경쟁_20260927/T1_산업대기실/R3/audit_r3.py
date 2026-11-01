# T1 R3 최종시험 전 감사 — 피처 시점 교란 시험(개발 원점 c=2012에서만, 최종시험 결과변수 미사용)
#  원점 연말 이후에 공개·등록된 KR 출원(또는 출원연도 > c)의 KSIC·IPC·출원인 연결을 무작위로 바꾸고, 원점 이후 물가 자료를 무작위로 바꿔도
#  동결 모형의 새 피처(ipcx·ipcm·entfit·peer·price)가 비트 단위로 같은지 확인한다. 같으면 원점 이후 정보가 그 피처에 들어가지 않는다.
#  디자인·실용신안은 분류기 캐시(P_*)에 원점 cap 이후 등록분이 아예 없는지 확인한다.
import sys, json, pathlib, time
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
import numpy as np, pandas as pd
import r3
from r3 import fc_r3, peak_gb
import r3ext, r3ext2, r3ext3
t0 = time.time(); C = 2012; CE = int(f'{C}1231')
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

te0 = pd.read_parquet(fc_r3(C)/'prep_te.parquet', columns=['firm', 'k', 'cut', 't_lsum'])
rs = np.random.RandomState(7); te0 = te0.iloc[np.sort(rs.choice(len(te0), 150000, replace=False))].reset_index(drop=True)
tr0 = te0.iloc[:2000].copy()
COLS = ['ipcx_fit', 'ipcx_fit_rank', 'ipcm_fit', 'ipcm_fit_rank', 'ent_fit', 'ent_fit_rank', 'peer_k', 'peer_k_rank']


def run_feats():
    tr, te = tr0.copy(), te0.copy()
    r3ext2._PPROF.clear()
    for g, f in (('ipcx', r3ext2.EXT['ipcx']), ('ipcm', r3ext3.EXT['ipcm']), ('entfit', r3ext3.EXT['entfit']), ('peer', r3ext3.EXT['peer'])):
        tr, te = f(tr, te, C, False)
    return te[COLS].copy()


A = run_feats(); print('기준 피처 계산', round(time.time() - t0), f'{peak_gb():.2f}GB', flush=True)

# ---- 교란: 원점 이후 가시 출원 ----
X = r3ext2.ipcx_src(); Y = r3ext2.ipcx_src_mg()
fut = ~((X['ay'] <= C) & (X['vis'] <= CE)); print('원점 이후 가시 출원 수', int(fut.sum()), '/', len(fut), flush=True)
orig = {k: X[k].copy() for k in ('k', 'ej', 'fa_firm')}; origY = Y['ej'].copy()
rng = np.random.RandomState(11)
X['k'] = np.where(fut, rng.randint(0, 50, len(fut)), X['k']).astype(X['k'].dtype)
m = fut[X['ea']]; ej = X['ej'].copy(); ej[m] = rng.randint(0, X['nsc'], int(m.sum())); X['ej'] = ej
m2 = fut[Y['ea']]; ej2 = Y['ej'].copy(); ej2[m2] = rng.randint(0, len(Y['mgs']), int(m2.sum())); Y['ej'] = ej2
m3 = fut[X['fa_a']]; ff = X['fa_firm'].copy(); pool = pd.unique(te0.firm.astype(str).values); ff[m3] = rng.choice(pool, int(m3.sum())); X['fa_firm'] = ff
Bf = run_feats(); print('교란 피처 계산', round(time.time() - t0), f'{peak_gb():.2f}GB', flush=True)
diff = {c: float(np.nanmax(np.abs(A[c].values - Bf[c].values))) if np.isfinite(A[c].values).any() else 0.0 for c in COLS}
nan_eq = {c: bool((np.isnan(A[c].values) == np.isnan(Bf[c].values)).all()) for c in COLS}
X.update(orig); Y['ej'] = origY

# ---- 물가: 원점−1년 이후 자료 교란 ----
_rp = pd.read_parquet
def rp(p, *a, **kw):
    d = _rp(p, *a, **kw)
    if 'demand_ksic_price_panel' in str(p):
        d = d.copy(); m = d.year >= C; d.loc[m, 'lidx'] = np.random.RandomState(5).randn(int(m.sum()))
    return d
PC = ['pr_ppi_g3', 'pr_ppi_g1', 'pr_exp_g3', 'pr_imp_g3', 'pr_gap']
import r2
r2.light_load()
pa_ = r3ext.ext_price(tr0.copy(), te0.copy(), C)[1][PC]
r3ext.pd.read_parquet = rp
try: pb_ = r3ext.ext_price(tr0.copy(), te0.copy(), C)[1][PC]
finally: r3ext.pd.read_parquet = _rp
pdiff = {c: float(np.nanmax(np.abs(pa_[c].values - pb_[c].values))) if np.isfinite(pa_[c].values).any() else 0.0 for c in PC}

# ---- 디자인·실용신안 분류기 캐시 ----
dz = {}
for kind in ('design', 'um'):
    ix = pd.read_parquet(fc_r3(C)/f'P_{kind}_index.parquet', columns=['rgst_date'])
    dz[kind] = dict(n=int(len(ix)), max_rgst=str(ix.rgst_date.max()), after_cap=int((ix.rgst_date > f'{C}1231').sum()))

out = dict(origin=C, rows=int(len(te0)), future_visible_apps=int(fut.sum()), tech_feature_max_abs_diff=diff, tech_feature_nan_pattern_equal=nan_eq,
           price_max_abs_diff=pdiff, design_um_cache=dz, peak_gb=round(peak_gb(), 2), sec=round(time.time() - t0))
json.dump(out, open(R3/'audit_r3.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
