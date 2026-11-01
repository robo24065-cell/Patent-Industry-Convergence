"""C09-T3 verify_3: 원천에서 독립 재계산 (PREREG.txt 판정 1–3). 쓰기는 이 폴더만."""
import os, re, json, time, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr, rankdata
t = time.time()
MY = os.path.dirname(os.path.abspath(__file__)); WD = os.path.dirname(MY)
ROOT = os.path.abspath(os.path.join(WD, '..', '..', '..'))
SRC = os.path.join(ROOT, '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3', 'cache_pub_final')
log = open(os.path.join(MY, 'v3_log.txt'), 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

# 1) US 행 원천 읽기
tb = pq.read_table(os.path.join(SRC, 'base_rows.parquet'), columns=['ksic', 'cc', 'year', 'appl_year', 'appl_no', 'ipc_raw', 'app1'], filters=[('cc', '=', 'US')])
U = tb.to_pandas(); del tb
P('US base rows', len(U), round(time.time() - t))
R1 = re.compile(r'^([A-H][0-9]{2}[A-Z])([0-9]{1,4})/'); R2 = re.compile(r'^([A-H][0-9]{2}[A-Z])([0-9]{3,6})$')
def mgn(s):
    if s is None: return None
    s = s.replace(' ', '')
    m = R1.match(s)
    if m: return m.group(1) + ' ' + str(int(m.group(2)))
    if R2.match(s): return s[:4] + ' ' + str(int(s[4:len(s) - 2]))
    return None
uq = pd.Series(U.ipc_raw.dropna().unique())
mp = dict(zip(uq, uq.map(mgn)))
U['mg'] = U.ipc_raw.map(mp); U = U[U.mg.notna()].drop(columns=['ipc_raw', 'cc'])
U = U.groupby(['ksic', 'mg', 'year', 'appl_no'], sort=False).agg(app1=('app1', 'first'), appl_year=('appl_year', 'min')).reset_index()
U['miss'] = U.app1.isna() | (U.app1 == '')
U['usold'] = U.appl_year <= 2011
P('US dedup rows', len(U), 'miss', int(U.miss.sum()), 'usold', int(U.usold.sum()), round(time.time() - t))
key = ['ksic', 'mg', 'app1']
fa0 = U[~U.miss].groupby(key).year.min().rename('fy0')
fa1 = U[~U.miss & ~U.usold].groupby(key).year.min().rename('fy1')
U = U.join(fa0, on=key).join(fa1, on=key)
U['new0'] = (~U.miss) & (U.fy0 == U.year)
U['new1'] = (~U.miss) & (~U.usold) & (U.fy1 == U.year)
G = U.groupby(['ksic', 'mg', 'year']).agg(n_us=('appl_no', 'size'), new0=('new0', 'sum'), new1=('new1', 'sum'), usold=('usold', 'sum')).reset_index()
del U, fa0, fa1
# 2) 공식 panel 대조 (US n, n_newapp_apps)
PC = pd.read_parquet(os.path.join(SRC, 'panel_ccy.parquet'), columns=['ksic', 'mg', 'cc', 'year', 'n', 'n_newapp_apps'])
po = PC[PC.cc == 'US'].merge(G, on=['ksic', 'mg', 'year'], how='outer')
chk = dict(us_rows_official=int((PC.cc == 'US').sum()), us_rows_mine=len(G),
           maxdiff_n=float(np.nanmax(np.abs(po.n.fillna(0) - po.n_us.fillna(0)))),
           maxdiff_newapp=float(np.nanmax(np.abs(po.n_newapp_apps.fillna(0) - po.new0.fillna(0)))))
P('panel check', chk)
# 3) 창 집계
PC['cell'] = PC.ksic + '|' + PC.mg; G['cell'] = G.ksic + '|' + G.mg
D = pd.read_parquet(os.path.join(WD, 'verify_0', 'v0_D_final.parquet'), columns=['cell', 'mg', 't0', 'y', 'yb', 'tot_W', 'all_newapp_sh'])
V1 = pd.read_csv(os.path.join(WD, 'verify_1', 'v1_by_origin.csv'))
def win(df, cols, t0):
    w = df[(df.year >= t0 - 2) & (df.year <= t0)]
    return w.groupby('cell')[cols].sum()
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(0)
rows = []; feats = []
for t0, g in D.groupby('t0'):
    g = g.copy()
    allw = win(PC, ['n', 'n_newapp_apps'], t0)
    epw = win(PC[PC.cc == 'EP'], ['n', 'n_newapp_apps'], t0).add_prefix('ep_')
    usw = win(G, ['n_us', 'new0', 'new1', 'usold'], t0)
    g = g.join(allw, on='cell').join(usw, on='cell').join(epw, on='cell').fillna({'n_us': 0, 'new0': 0, 'new1': 0, 'usold': 0, 'ep_n': 0, 'ep_n_newapp_apps': 0})
    g['S0'] = g.n_newapp_apps / np.maximum(g.n, 1)
    g['S1'] = (g.n_newapp_apps - g.new0 + g.new1) / np.maximum(g.n - g.usold, 1)
    g['us_sh'] = g.n_us / np.maximum(g.n, 1)
    g['dsh'] = g.S0 - g.S1
    g['r_us_cell'] = np.where(g.n_us > 0, g.new0 / np.maximum(g.n_us, 1), np.nan)
    feats.append(g)
    yb = g.yb.values.astype(int)
    # 군집 부트스트랩 AUC (mg 단위)
    mgs = g.mg.values; um, inv = np.unique(mgs, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(um))]
    bs = []
    for _ in range(200):
        pick = rng.integers(0, len(um), len(um)); ii = np.concatenate([idx_by[k] for k in pick])
        if yb[ii].sum() in (0, len(ii)): continue
        bs.append(auc(yb[ii], g.dsh.values[ii]))
    rows.append(dict(t0=int(t0), n=len(g), maxdiff_S0_vs_D=float(np.abs(g.S0 - g.all_newapp_sh).max()), maxdiff_totW=float(np.abs(g.n - g.tot_W).max()),
                     mean_us_sh=g.us_sh.mean(), mean_abs_dsh=g.dsh.abs().mean(), rho_ussh_dsh=spearmanr(g.us_sh, g.dsh).statistic,
                     rank_corr_S0_S1=spearmanr(g.S0, g.S1).statistic, auc_dsh_yb=auc(yb, g.dsh.values),
                     auc_dsh_lo=np.percentile(bs, 2.5), auc_dsh_hi=np.percentile(bs, 97.5),
                     r_US=g.new0.sum() / max(g.n_us.sum(), 1), r_EP=g.ep_n_newapp_apps.sum() / max(g.ep_n.sum(), 1)))
R = pd.DataFrame(rows)
R = R.merge(V1[['t0', 'rho_ussh_dsh', 'rank_corr_S0_S1', 'auc_dsh_for_yb', 'mean_abs_dsh']].rename(columns=lambda c: c if c == 't0' else 'v1_' + c), on='t0')
R['d_rho'] = R.rho_ussh_dsh - R.v1_rho_ussh_dsh
# 판정 3: 차이의 차이 excess, 대안 왜곡량
rus21 = R.loc[R.t0 == 2021, 'r_US'].item(); rep21 = R.loc[R.t0 == 2021, 'r_EP'].item()
R['excess'] = (R.r_US - rus21) - (R.r_EP - rep21)
ext = []
for g in feats:
    t0 = int(g.t0.iloc[0]); ex = R.loc[R.t0 == t0, 'excess'].item()
    dss = ex * g.us_sh; Sss = g.S0 - dss
    yb = g.yb.values.astype(int)
    ext.append(dict(t0=t0, mean_dsh_ss=dss.mean(), rank_corr_S0_Sss=spearmanr(g.S0, Sss).statistic,
                    auc_dsh_ss_yb=auc(yb, dss.values) if ex != 0 else np.nan, rho_ussh_y=spearmanr(g.us_sh, g.y).statistic,
                    mean_r_us_cell=np.nanmean(g.r_us_cell)))
R = R.merge(pd.DataFrame(ext), on='t0')
R.round(4).to_csv(os.path.join(MY, 'v3_by_origin.csv'), index=False, encoding='utf-8-sig')
tr = R[R.t0 <= 2018]
out = dict(panel_check=chk,
           j1_n_train_absrho_ge_0_3=int((tr.rho_ussh_dsh.abs() >= 0.3).sum()), j1_max_abs_d_rho=float(R.d_rho.abs().max()),
           j1_max_abs_d_rankcorr=float((R.rank_corr_S0_S1 - R.v1_rank_corr_S0_S1).abs().max()),
           j1_max_abs_d_auc=float((R.auc_dsh_yb - R.v1_auc_dsh_for_yb).abs().max()),
           j2_auc_range=[float(R.auc_dsh_yb.min()), float(R.auc_dsh_yb.max())],
           j2_n_origins_ci_excl_0_5=int(((R.auc_dsh_lo > 0.5) | (R.auc_dsh_hi < 0.5)).sum()),
           j3_excess_2016=float(R.loc[R.t0 == 2016, 'excess'].item()), j3_excess_2017=float(R.loc[R.t0 == 2017, 'excess'].item()),
           j3_mean_dsh_ss_2016=float(R.loc[R.t0 == 2016, 'mean_dsh_ss'].item()), j3_mean_dsh_ss_2017=float(R.loc[R.t0 == 2017, 'mean_dsh_ss'].item()),
           secs=round(time.time() - t))
json.dump(out, open(os.path.join(MY, 'v3_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
P(R.round(3).to_string()); P(json.dumps(out, ensure_ascii=False, default=float))
