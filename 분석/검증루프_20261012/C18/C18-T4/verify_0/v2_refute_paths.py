# C18-T4 verify_0 v2: 반박 — '가장 유리한 경로'(PREREG §7).
#  (A) K2 D2: 최소 SE 추정량(영향함수 KSIC/jackknife)으로 G1 통과 → 같은 피처 경량 LR 재학습으로 P0(b) 확인(+ IF KSIC SE).
#  (B) K1 LR P0(b) 실패의 견고성: 다른 LR 학습 창(v−3..v−1, 음성 1/3 = C12 XGB 학습 설계)으로 2024 재학습.
# C13·C12 캐시는 읽기만. 출력 verify_0/v2_out.json
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
from lib18 import C, C12C, C13C, AUCPrep, out, load_frame, Z80
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')
t = time.time()
res = {}


def auc_if(s, y, w):
    y = y.astype(bool)
    o = np.argsort(s, kind='mergesort'); ss = s[o]
    grp = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
    g = np.empty(len(s), np.int64); g[o] = grp
    ng = grp[-1] + 1
    wp = np.bincount(g[y], weights=w[y], minlength=ng); wn = np.bincount(g[~y], weights=w[~y], minlength=ng)
    Wp, Wn = wp.sum(), wn.sum()
    v10 = (np.cumsum(wn) - wn + 0.5 * wn) / Wn
    v01 = (Wp - np.cumsum(wp) + 0.5 * wp) / Wp
    th = float((wp * v10).sum() / Wp)
    return th, np.where(y, w * (v10[g] - th) / Wp, w * (v01[g] - th) / Wn)


def lrfit(A, y, Bm, sw=None):
    mu = np.nanmean(A, 0); A = np.where(np.isnan(A), mu, A); Bm = np.where(np.isnan(Bm), mu, Bm)
    sc = StandardScaler().fit(A)
    return LogisticRegression(C=1.0, solver='lbfgs', max_iter=500).fit(sc.transform(A), y, sample_weight=sw).decision_function(sc.transform(Bm))


# ---------- (A) K2 ----------
BASE = ['b_n3', 'b_n1', 'b_gr', 'b_npart', 'b_new3', 'b_new1', 'b_cosh', 'b_nsc', 'b_iscn']
SX = ['x_nk', 'x_ent', 'x_msh', 'x_xpart', 'x_xpart_na', 'x_xnew3', 'x_indr', 'x_indn', 'x_imp', 'x_imp_na', 'x_modal_na']
TR = list(range(2016, 2023)); EV = 2023
ld = lambda s: pd.concat([pd.read_parquet(C13C + rf'\u_{s}_{t}.parquet').assign(t0=t) for t in TR + [EV]], ignore_index=True)
P = ld('prov'); Fs = {'prov': P, 'free': ld('free'), 'hsc': ld('hsc')}
for s in Fs:
    assert (Fs[s].u.values == P.u.values).all()
y = P.y.values.astype(int); tr = P.t0.isin(TR).values; ev = (P.t0 == EV).values
fr = pd.read_parquet(C + r'\fr_K2_2023.parquet')
assert (fr.y.values == y[ev]).all()
sc = {'base': lrfit(P[BASE].values[tr].astype(float), y[tr], P[BASE].values[ev].astype(float))}
for s in Fs:
    X = np.c_[P[BASE].values, Fs[s][SX].values].astype(float)
    sc[s] = lrfit(X[tr], y[tr], X[ev])
ye = y[ev]; w1 = np.ones(len(ye))
th, ps = {}, {}
for m, s in sc.items():
    th[m], ps[m] = auc_if(s, ye, w1)
best = 'free' if th['free'] >= th['hsc'] else 'hsc'
psi = ps['prov'] - ps[best]; k = fr.k.values
Sk = np.bincount(pd.factorize(k)[0], weights=psi); G = len(Sk)
se_k = float(np.sqrt(G / (G - 1) * (Sk ** 2).sum())); fL = 1981.51 / ye.sum()
res['A_K2_lr'] = {'auc': th, 'best_public': best, 'd_prov_best': th['prov'] - th[best], 'd_prov_free': th['prov'] - th['free'], 'd_prov_hsc': th['prov'] - th['hsc'],
                  'SE_IF_k_full': se_k, 'SE_IF_k_lock_iid_scaled': se_k / np.sqrt(fL),
                  'power_0.005_lock': float(norm.cdf(0.005 / (se_k / np.sqrt(fL)) - 1.959964)),
                  'P0b_lr_pass': bool(th['prov'] - th['free'] > 0 and th['prov'] - th['hsc'] > 0)}
out('A', json.dumps(res['A_K2_lr']), round(time.time() - t, 1))

# ---------- (B) K1 LR 다른 학습 창(2024) ----------
SEED12 = 20261013
PUBF = ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc', 'is_cn', 's_n5', 's_grow', 's_nfirm', 's_newrate',
        'p_sec', 'p_relmean', 'p_relmax', 'p_cls', 'p_sect', 'p_cf', 'pi_rfirm', 'pi_rsc']
CTX = ['dot', 'cos', 'adopt', 'entry', 'qmain', 'mom_c', 'sind_c', 'mom_kr', 'sind_kr', 'mom_5', 'sind_5']
ARMS = ['prov', 'free', 'hsc', 'scagg']
cols = ['cc', 'y', 'w', 'Y'] + PUBF + [f'{a}_{x}' for a in ARMS for x in CTX]


def load(Yr, sub):
    f = pd.concat([pd.read_parquet(C12C + rf'\u_{cc}_{Yr}.parquet', columns=cols) for cc in ['CN', 'EP']], ignore_index=True)
    if sub:
        rng = np.random.default_rng(SEED12 + Yr + 7)
        keep = (f.y.values == 1) | (rng.random(len(f)) < 1 / 3)
        f = f[keep].copy(); f.loc[f.y == 0, 'w'] *= 3
    return f


v = 2024
trd = pd.concat([load(Yr, True) for Yr in [v - 3, v - 2, v - 1]], ignore_index=True); evd = load(v, False)
Fk = load_frame('K1', v)
assert (Fk.y.values == evd.y.values).all()
th = {}
for arm in ['base'] + ARMS:
    cs = PUBF + ([f'{arm}_{x}' for x in CTX] if arm != 'base' else [])
    s = lrfit(trd[cs].values.astype(float), trd.y.values, evd[cs].values.astype(float), trd.w.values)
    th[arm] = AUCPrep(s, evd.y.values).auc(evd.w.values.astype(float))
res['B_K1_lr_trainwin3'] = {'auc': th, 'd_prov_scagg': th['prov'] - th['scagg'], 'd_prov_free': th['prov'] - th['free'], 'd_prov_hsc': th['prov'] - th['hsc']}
out('B', json.dumps(res['B_K1_lr_trainwin3']), round(time.time() - t, 1))
res['sec'] = round(time.time() - t, 1)
json.dump(res, open(os.path.join(HERE, 'v2_out.json'), 'w'), indent=1, default=float)
