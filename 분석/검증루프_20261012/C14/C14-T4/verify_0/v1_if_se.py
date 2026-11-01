# C14-T4 verify_0 v1: 다른 SE 경로(AUC 영향함수 기반 해석적 CGM/dyad 분산)와 다른 학습기(표준화 로지스틱 재학습)
#  (A) C12-T4 후보: 저장 예측(원점 2020–24)의 단일 원점 ΔAUC 해석적 CGM SE → 검정력. + 원점 2024 로지스틱 재학습(학습 2021–23) Δ·SE·검정력
#  (B) U6×Eb: b3 저장 예측의 해석적 dyad SE + 로지스틱 재학습 Δ·SE·검정력
# 작업 스크립트(a1·a2·b3)의 부트스트랩 코드를 쓰지 않는다. 출력 verify_0/v1_out.json
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
from lib14 import C, C12C, W, SEED, E_LOCK_A, TARGET, EVAL_Y, CCS, CUT, out, guard_parquet_pub
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')
t = time.time()


def auc_if(s, y, w):
    """가중 AUC와 단위별 영향(ψ). 동률 0.5."""
    y = y.astype(bool)
    o = np.argsort(s, kind='mergesort'); ss = s[o]
    grp = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
    g = np.empty(len(s), np.int64); g[o] = grp
    ng = grp[-1] + 1
    wp = np.bincount(g[y], weights=w[y], minlength=ng); wn = np.bincount(g[~y], weights=w[~y], minlength=ng)
    Wp, Wn = wp.sum(), wn.sum()
    below_n = np.cumsum(wn) - wn; above_p = Wp - np.cumsum(wp)
    v10 = (below_n + 0.5 * wn) / Wn      # 양성 i: 점수 낮은 음성 몫
    v01 = (above_p + 0.5 * wp) / Wp      # 음성 j: 점수 높은 양성 몫
    th = float((wp * v10).sum() / Wp)
    psi = np.where(y, w * (v10[g] - th) / Wp, w * (v01[g] - th) / Wn)
    return th, psi


def csum2(psi, cl):
    return float((np.bincount(cl, weights=psi) ** 2).sum())


res = {}
# ---------- (A) 저장 예측 ----------
rows = []
for v in EVAL_Y:
    P = pd.read_parquet(C12C + rf'\pred_{v}_base.parquet'); Q = pd.read_parquet(C12C + rf'\pred_{v}_pubarms.parquet')
    assert (P.aid.values == Q.aid.values).all()
    y = P.y.values.astype(int); w = P.w.values.astype(float)
    th, ps = {}, {}
    for m, s in [('prov', P.prov.values), ('free', Q.free.values), ('hsc', Q.hsc.values)]:
        th[m], ps[m] = auc_if(s.astype(float), y, w)
    best = 'free' if th['free'] >= th['hsc'] else 'hsc'
    d = th['prov'] - th[best]; psi = ps['prov'] - ps[best]
    f = pd.factorize(P.cc.astype(str) + ':' + P.aid.astype(str))[0]; s_ = pd.factorize(P.s)[0]
    Vf, Vs, Vc = csum2(psi, f), csum2(psi, s_), float((psi ** 2).sum())
    Vcgm = Vf + Vs - Vc
    se = np.sqrt(max(Vcgm, Vf, Vs)); npos = int(y.sum())
    sc = np.sqrt(npos / E_LOCK_A)
    rows.append({'v': v, 'dAUC': d, 'best': best, 'SE_if_firm': np.sqrt(Vf), 'SE_if_sc': np.sqrt(Vs), 'SE_if_iid': np.sqrt(Vc), 'SE_if_CGM': np.sqrt(max(Vcgm, 0)),
                 'SE_if_primary': se, 'power_lock_primary': float(norm.cdf(TARGET / (se * sc) - 1.96)),
                 'power_lock_at_observed': float(norm.cdf(d / (se * sc) - 1.96))})
    out(rows[-1], round(time.time() - t, 1))
res['A_stored_if'] = rows
res['A_stored_if_min_power'] = float(min(r['power_lock_primary'] for r in rows))

# ---------- (A) 다른 학습기: 로지스틱 재학습(평가 2024, 학습 2021–23 저장 행, 나라·원점별 음성 상한 100,000) ----------
PUBF = ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc', 'is_cn', 's_n5', 's_grow', 's_nfirm', 's_newrate',
        'p_sec', 'p_relmean', 'p_relmax', 'p_cls', 'p_sect', 'p_cf', 'pi_rfirm', 'pi_rsc']
CTX = ['dot', 'cos', 'adopt', 'entry', 'qmain', 'mom_c', 'sind_c', 'mom_kr', 'sind_kr', 'mom_5', 'sind_5']
cols = PUBF + [f'{L}_{x}' for L in ['prov', 'free', 'hsc'] for x in CTX]
TR = []
for Y in [2021, 2022, 2023]:
    for cc in CCS:
        u = pd.read_parquet(C12C + rf'\u_{cc}_{Y}.parquet', columns=['y', 'w'] + cols)
        rng = np.random.default_rng(SEED + 500 + Y + (0 if cc == 'CN' else 1))
        neg = np.flatnonzero(u.y.values == 0)
        if len(neg) > 100000:
            keep = np.r_[np.flatnonzero(u.y.values == 1), rng.choice(neg, 100000, replace=False)]
            fac = len(neg) / 100000
            u = u.iloc[np.sort(keep)].copy(); u.loc[u.y == 0, 'w'] = u.loc[u.y == 0, 'w'] * fac
        TR.append(u)
TR = pd.concat(TR, ignore_index=True)
EV = pd.concat([pd.read_parquet(C12C + rf'\u_{cc}_2024.parquet', columns=['cc', 'aid', 's', 'y', 'w'] + cols) for cc in CCS], ignore_index=True)
mu = TR[cols].mean()
Xtr_all = TR[cols].fillna(mu); Xev_all = EV[cols].fillna(mu)
ye = EV.y.values.astype(int); we = EV.w.values.astype(float)
th, ps = {}, {}
for arm in ['pub', 'prov', 'free', 'hsc']:
    cs = PUBF + ([f'{arm}_{x}' for x in CTX] if arm != 'pub' else [])
    scl = StandardScaler().fit(Xtr_all[cs])
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500).fit(scl.transform(Xtr_all[cs]), TR.y.values, sample_weight=TR.w.values)
    s = lr.decision_function(scl.transform(Xev_all[cs]))
    th[arm], ps[arm] = auc_if(s, ye, we)
    out('logit', arm, th[arm], round(time.time() - t, 1))
best = 'free' if th['free'] >= th['hsc'] else 'hsc'
d = th['prov'] - th[best]; psi = ps['prov'] - ps[best]
f = pd.factorize(EV.cc.astype(str) + ':' + EV.aid.astype(str))[0]; s_ = pd.factorize(EV.s)[0]
Vf, Vs, Vc = csum2(psi, f), csum2(psi, s_), float((psi ** 2).sum()); se = np.sqrt(max(Vf + Vs - Vc, Vf, Vs))
sc = np.sqrt(int(ye.sum()) / E_LOCK_A)
res['A_logit_2024'] = {'auc': th, 'best': best, 'dAUC': d, 'SE_CGM_primary': float(se), 'SE_firm': float(np.sqrt(Vf)), 'SE_sc': float(np.sqrt(Vs)),
                       'power_lock_target': float(norm.cdf(TARGET / (se * sc) - 1.96)), 'power_lock_at_observed': float(norm.cdf(d / (se * sc) - 1.96)) if d > 0 else 0.0}
out('A_logit', res['A_logit_2024'])

# ---------- (B) U6×Eb ----------
b1 = json.load(open(C + r'\b1_out.json', encoding='utf-8'))
E_POS, E_NEG = b1['Eb_pos']['lock_exp'], b1['Eb_neg']['lock_exp']
pr = pd.read_parquet(C + r'\b3_pred_2023.parquet')
y = pr.y.values.astype(int); w1 = np.ones(len(pr))


def dyad_se(psi, df):
    nodes = pd.Index(pd.concat([df.key, df.key_g]).unique())
    fi, gi = nodes.get_indexer(df.key), nodes.get_indexer(df.key_g)
    Fn = float(((np.bincount(fi, weights=psi, minlength=len(nodes)) + np.bincount(gi, weights=psi, minlength=len(nodes))) ** 2).sum())
    Vf = Fn - float((psi ** 2).sum())
    kn = pd.Index(pd.concat([df.mf, df.mg]).astype(str).unique())
    ki, kj = kn.get_indexer(df.mf.astype(str)), kn.get_indexer(df.mg.astype(str))
    Kn = float(((np.bincount(ki, weights=psi, minlength=len(kn)) + np.bincount(kj, weights=psi, minlength=len(kn))) ** 2).sum())
    kc = pd.factorize(np.where(df.mf.astype(str) < df.mg.astype(str), df.mf.astype(str) + '|' + df.mg.astype(str), df.mg.astype(str) + '|' + df.mf.astype(str)))[0]
    Vk = Kn - csum2(psi, kc)
    return {'SE_iid': float(np.sqrt((psi ** 2).sum())), 'SE_Fdyad': float(np.sqrt(max(Vf, 0))), 'SE_Kdyad': float(np.sqrt(max(Vk, 0))), 'SE_primary': float(np.sqrt(max(Vf, Vk, 0)))}


scale = np.sqrt((1 / E_POS + 1 / E_NEG) / (1 / y.sum() + 1 / (1 - y).sum()))
th, ps = {}, {}
for m in ['prov', 'free', 'hsc']:
    th[m], ps[m] = auc_if(pr['p_' + m].values, y, w1)
best = 'free' if th['free'] >= th['hsc'] else 'hsc'
SEb = dyad_se(ps['prov'] - ps[best], pr)
res['B_U6Eb_stored_if'] = {'auc': th, 'best': best, 'dAUC': th['prov'] - th[best], **SEb, 'power_lock_primary': float(norm.cdf(TARGET / (SEb['SE_primary'] * scale) - 1.96))}
out('B stored', res['B_U6Eb_stored_if'])
# 로지스틱 재학습
U = pd.concat([pd.read_parquet(C + rf'\b2_u_{yy}.parquet') for yy in range(2016, 2024)], ignore_index=True)
BASE = [c for c in U.columns if c.startswith('b_')]
tr = U.t0 < 2023; ev = U.t0 == 2023
Ev = U[ev].reset_index(drop=True); Ev = Ev.rename(columns={'mf_prov': 'mf', 'mg_prov': 'mg'})
ye = Ev.y.values.astype(int)
th, ps = {}, {}
for L in ['prov', 'free', 'hsc']:
    cs = BASE + [c for c in U.columns if c.startswith(L + '_')]
    scl = StandardScaler().fit(U.loc[tr, cs])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(scl.transform(U.loc[tr, cs]), U.y.values[tr])
    th[L], ps[L] = auc_if(lr.decision_function(scl.transform(U.loc[ev, cs])), ye, np.ones(len(ye)))
best = 'free' if th['free'] >= th['hsc'] else 'hsc'
SEb = dyad_se(ps['prov'] - ps[best], Ev)
res['B_U6Eb_logit'] = {'auc': th, 'best': best, 'dAUC': th['prov'] - th[best], **SEb, 'power_lock_primary': float(norm.cdf(TARGET / (SEb['SE_primary'] * scale) - 1.96))}
out('B logit', res['B_U6Eb_logit'])
res['secs'] = round(time.time() - t, 1)
json.dump(res, open(os.path.join(HERE, 'v1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
out('done', res['secs'])
