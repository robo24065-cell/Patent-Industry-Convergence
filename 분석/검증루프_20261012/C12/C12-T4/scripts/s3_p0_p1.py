# P0(검정력·규모) + P1(기저율·양성 수). 출력 cache/p0p1.json
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feat12 import *
from evalkit12 import *
from scipy.stats import norm

t = time.time()
res = {}
# ---- P1 ----
info = {f'{cc}_{Y}': json.load(open(C + rf'\info_{cc}_{Y}.json')) for cc in CCS for Y in ALL_Y}
p1 = {}
for Y in ALL_Y:
    r = {k: sum(info[f'{cc}_{Y}'][k] for cc in CCS) for k in ['n_active', 'n_risk', 'n_pos', 'n_pos_firms']}
    r['base'] = r['n_pos'] / r['n_risk']
    r['by_cc'] = {cc: {k: info[f'{cc}_{Y}'][k] for k in ['n_active', 'n_cand', 'n_risk', 'n_pos', 'n_pos_firms']} for cc in CCS}
    p1[Y] = r
res['P1'] = {'table': p1, 'pass': bool(all(p1[v]['n_pos'] >= 1000 and p1[v]['n_pos_firms'] >= 300 for v in EVAL_Y))}
out('P1', res['P1']['pass'])
# ---- P0(a) 잠금 위험집합(2025-01-01 이전 문서만) ----
lock = {}
for cc in CCS:
    Cn = Country(cc)
    lock[cc] = Cn.count(2025)
    out(cc, lock[cc], round(time.time() - t, 1))
    del Cn
base_2324 = float(np.mean([p1[2023]['base'], p1[2024]['base']]))
nrisk = sum(lock[cc]['n_risk'] for cc in CCS)
res['P0a'] = {'lock': lock, 'risk_pairs': nrisk, 'risk_firms': sum(lock[cc]['n_active'] for cc in CCS),
              'base_2023_2024': base_2324, 'exp_pos': nrisk * base_2324, 'pass': bool(nrisk * base_2324 >= 1000)}
out('P0a', res['P0a'])
# ---- P0(b) 검정력 ----
fr = [pd.read_parquet(C + rf'\pred_{v}_sb.parquet') for v in EVAL_Y]
bt = Boot(fr, B=200)
rng = np.random.default_rng(SEED)
U = rng.standard_normal(len(bt.fidx)); V = rng.standard_normal(len(bt.sidx))
Z0 = []
for k, f in enumerate(fr):
    fi, si = bt.maps[k]
    e = rng.standard_normal(len(f))
    Z0.append((U[fi] + V[si] + e) / np.sqrt(3))
prep0 = [AUCPrep(f.SB.values, f.y.values) for f in fr]
a0 = [p.auc(f.w.values.astype(np.float64)) for p, f in zip(prep0, fr)]


def ds_mu(mu):
    out_ = []
    for k, f in enumerate(fr):
        s1 = f.SB.values + mu * f.y.values + Z0[k]
        a1 = AUCPrep(s1, f.y.values).auc(f.w.values.astype(np.float64))
        out_.append(dskill(a1, a0[k]))
    return float(np.mean(out_)), out_


lo, hi = 0.0, 3.0
for it in range(40):
    mid = (lo + hi) / 2
    m, _ = ds_mu(mid)
    if abs(m - 0.005) <= 0.0005 and it > 5:
        break
    if m < 0.005:
        lo = mid
    else:
        hi = mid
mu = mid; mval, per = ds_mu(mu)
out('mu', mu, mval, round(time.time() - t, 1))
S1 = [f.SB.values + mu * f.y.values + Z0[k] for k, f in enumerate(fr)]
prep1 = [AUCPrep(s, f.y.values) for s, f in zip(S1, fr)]
bs = {'firm': [], 'two': []}; b24 = []
for b in range(bt.B):
    for kind in ['firm', 'two']:
        vals = []
        for k, f in enumerate(fr):
            w = f.w.values * bt.weights(k, b, kind)
            vals.append(dskill(prep1[k].auc(w), prep0[k].auc(w)))
        bs[kind].append(np.mean(vals))
        if kind == 'two':
            b24.append(vals[-1])
se = {k: float(np.std(v, ddof=1)) for k, v in bs.items()}
se24 = float(np.std(b24, ddof=1))
pw = {k: float(norm.cdf(0.005 / se[k] - 1.96)) for k in se}
res['P0b'] = {'mu': mu, 'dskill_mean': mval, 'dskill_per_origin': per, 'SE': se, 'power': pw,
              'single_2024_two_SE': se24, 'single_2024_two_power': float(norm.cdf(0.005 / se24 - 1.96)),
              'auc_SB': a0, 'pass': bool(pw['two'] >= 0.80)}
res['P0'] = {'pass': bool(res['P0a']['pass'] and res['P0b']['pass'])}
out('P0b', json.dumps(res['P0b']))
res['secs'] = round(time.time() - t, 1)
json.dump(res, open(C + r'\p0p1.json', 'w'), indent=1, default=float)
out('done', res['secs'])
