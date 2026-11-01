# P0(검정력)·P1(규모). 사용: python -I p0_power.py CAND
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lib11 import *
from evalkit import AUCPrep, skill

CAND = sys.argv[1]
t_start = time.time()
NK = 50
cols = ['cc', 'aid', 'k', 'risk_p', 'y_p', 'f_log3', 'p_nk3', 'p_ck_log3', 'p_own_n1']
U = {}
for Y in ALL_Y:
    d = pd.read_parquet(C + rf'\u_{CAND}_{Y}.parquet', columns=cols)
    d = d[d.risk_p].copy()
    d['fid'] = d.aid + np.where(d.cc == 'EP', 10_000_000, 0)
    U[Y] = d
res = {'cand': CAND, 'P1': {}}
for Y, d in U.items():
    pf = d[d.y_p].fid.nunique()
    res['P1'][Y] = dict(units=len(d), pos=int(d.y_p.sum()), base=float(d.y_p.mean()), risk_firms=int(d.fid.nunique()), pos_firms=int(pf),
                        pos_firm_share=float(pf / d.fid.nunique()),
                        by_cc={cc: dict(units=int((d.cc == cc).sum()), pos=int(d[d.cc == cc].y_p.sum())) for cc in d.cc.unique()})
    out(Y, res['P1'][Y]['units'], res['P1'][Y]['pos'], round(res['P1'][Y]['base'], 5), res['P1'][Y]['risk_firms'], pf)

# ---- 잠금 위험집합(2025-01-01 이전 문서만) ----
CCS = ['KR'] if CAND == 'B' else ['CN', 'EP']
t0 = ymd(2025)
lock = {}
for cc in CCS:
    dd = pd.read_parquet(C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k'])
    a = pd.read_parquet(C + rf'\app_{cc}.parquet')
    if cc == 'KR':
        a = a[a.corp]
    m = a.merge(dd, on='did')
    assert m.pub.max() < CUT
    keys = pd.Index(m.key.unique()); m['aid'] = keys.get_indexer(m.key)
    y3 = m[(m.pub >= t0 - 30000) & (m.pub < t0)]
    n3 = y3.groupby('aid').size()
    if CAND == 'B':
        act = n3[n3 >= 3].index
    else:
        n1 = y3[y3.pub >= t0 - 10000].groupby('aid').size()
        act = n3[n3 >= 10].index.intersection(n1[n1 >= 1].index)
    mm = m[m.aid.isin(act)]
    def cnt(lo, hi):
        s = mm[(mm.pub >= lo) & (mm.pub < hi)]
        return s.groupby(['aid', 'k']).size()
    if CAND == 'A':
        c5 = cnt(t0 - 50000, t0)
        held = c5.groupby('aid').size().reindex(act).fillna(0)
        units = int((NK - held).sum()); firms = int(((NK - held) > 0).sum())
    elif CAND == 'B':
        past = cnt(t0 - 100000, t0 - 30000); now = cnt(t0 - 30000, t0)
        r = past.index.difference(now.index)
        units = len(r); firms = r.get_level_values(0).nunique()
    else:
        c1 = cnt(t0 - 10000, t0)
        r = c1[c1 >= 4]
        units = len(r); firms = r.index.get_level_values(0).nunique()
    lock[cc] = dict(active=int(len(act)), units=int(units), firms=int(firms))
    out('lock', cc, lock[cc])
L_units = sum(v['units'] for v in lock.values()); L_firms = sum(v['firms'] for v in lock.values())
pfs = np.mean([res['P1'][y]['pos_firm_share'] for y in (2023, 2024)])
br = np.mean([res['P1'][y]['base'] for y in (2023, 2024)])
res['P0'] = dict(lock=lock, lock_units=L_units, lock_firms=L_firms, exp_pos_firms=float(L_firms * pfs), exp_pos_units=float(L_units * br),
                 a_pass=bool(L_firms >= 300), b_pass=bool(L_firms * pfs >= 100))
out('P0 a/b', res['P0'])

# ---- P0(c) 검정력 ----
X = ['f_log3', 'p_nk3', 'p_ck_log3'] + (['p_own_n1'] if CAND == 'C' else [])
orig = []
for v in EVAL_Y:
    tr = pd.concat([U[y] for y in ALL_Y if y < v])
    te = U[v]
    def fx(d):
        Z = d[X].astype(float).copy()
        Z['p_nk3'] = np.log1p(Z.p_nk3)
        if 'p_own_n1' in Z:
            Z['p_own_n1'] = np.log1p(Z.p_own_n1)
        return Z.values
    sc = StandardScaler().fit(fx(tr))
    lr = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(fx(tr)), tr.y_p.values)
    s0 = lr.decision_function(sc.transform(fx(te)))
    orig.append(dict(s0=s0, y=te.y_p.values, f=te.fid.values, k=te.k.values))
allf = pd.Index(np.unique(np.concatenate([o['f'] for o in orig])))
for o in orig:
    o['fi'] = allf.get_indexer(o['f'])
ua = np.random.default_rng(SEED).standard_normal(len(allf))
for i, o in enumerate(orig):
    e = np.random.default_rng(SEED + 1 + i).standard_normal(len(o['y']))
    o['noise'] = np.sqrt(0.5) * ua[o['fi']] + np.sqrt(0.5) * e
    o['p0'] = AUCPrep(o['s0'], o['y'])


def dsk(mu):
    return np.mean([skill(AUCPrep(o['s0'] + mu * o['y'] + o['noise'], o['y']).auc(), o['p0'].auc()) for o in orig])


lo, hi = 0.0, 10.0
for it in range(30):
    mid = (lo + hi) / 2
    if dsk(mid) < 0.02:
        lo = mid
    else:
        hi = mid
mu = hi; d_obs = dsk(mu)
for o in orig:
    o['p1'] = AUCPrep(o['s0'] + mu * o['y'] + o['noise'], o['y'])
B = 200
out('mu', mu, 'dskill', d_obs, round(time.time() - t_start, 1))
pw = {}
for kind in ['firm', 'twoway']:
    Wf = np.random.default_rng(SEED).poisson(1.0, size=(B, len(allf))).astype(float)
    Wk = np.random.default_rng(SEED + 7).poisson(1.0, size=(B, NK)).astype(float)
    bs = []
    for b in range(B):
        vals = []
        for o in orig:
            w = Wf[b, o['fi']]
            if kind == 'twoway':
                w = w * Wk[b, o['k']]
            vals.append(skill(o['p1'].auc(w), o['p0'].auc(w)))
        bs.append(np.mean(vals))
    bs = np.array(bs); se = float(bs.std(ddof=1))
    pw[kind] = dict(se=se, power=float(norm.cdf(0.02 / se - 1.96)), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    out(kind, pw[kind], round(time.time() - t_start, 1))
res['P0']['c'] = dict(mu=mu, dskill_synth=float(d_obs), base_auc=[o['p0'].auc() for o in orig], **pw)
res['P0']['c_pass'] = bool(pw['firm']['power'] >= 0.8)
res['P0']['pass'] = bool(res['P0']['a_pass'] and res['P0']['b_pass'] and res['P0']['c_pass'])
ev = [res['P1'][y] for y in EVAL_Y]
res['P1_pass'] = bool(all(e['pos'] >= 200 and e['pos_firms'] >= 100 for e in ev))
json.dump(res, open(C + rf'\p0_{CAND}.json', 'w'), indent=1, default=float)
out('P0 pass', res['P0']['pass'], 'P1 pass', res['P1_pass'], round(time.time() - t_start, 1))
