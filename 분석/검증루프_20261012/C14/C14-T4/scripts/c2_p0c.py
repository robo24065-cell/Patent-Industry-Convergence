# C14-T4 c2: P0(c) 판정(PREREG §1.7). (c-i) se_struct.json의 포함 관계, (c-ii) c1 귀무 40회. 출력 cache/c2_p0c.json
import sys, os, json, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
se = json.load(open(W + r'\se_struct.json', encoding='utf-8'))
p = se['point']
e1, e2 = p['E_prov_to_pub_mean'], p['E_pub_to_prov_mean']
se1 = se['E_prov_to_pub_mean5']['SE_primary'][0]
ci1 = [e1 - 1.96 * se1, e1 + 1.96 * se1]
n_nonpos = int(sum(d <= 0 for d in p['dAUC_v']))
ci_ = {'E_prov_to_pub': e1, 'E_pub_to_prov': e2, 'SE_primary_E_prov_to_pub': se1, 'ci_normal': ci1, 'n_origins_dAUC_le0': n_nonpos,
       'reject': bool(e2 >= e1 or ci1[0] <= 0 or n_nonpos >= 3)}
# (c-ii)
real = None; nulls = {}
for f in sorted(glob.glob(C + r'\c1_null_*.json')):
    d = json.load(open(f))
    if 'G_real' in d:
        real = d
    for r in d.get('nulls', []):
        nulls[r['b']] = r
bs = sorted(nulls)
assert bs == list(range(40)), bs
G = np.array([nulls[b]['G'] for b in bs]); Gper = np.array([nulls[b]['per'] for b in bs])
Greal = real['G_real'][0]; Gfree = real['G_free'][0]; Ghsc = real['G_hsc'][0]; Gpb = max(Gfree, Ghsc)
Rctx = float(G.mean() / Greal)
RD = float((G.mean() - Gpb) / (Greal - Gpb))
per_real = np.array(real['G_real'][1]); per_pb = np.maximum(real['G_free'][1], real['G_hsc'][1])
# C12 귀무 200회(일관성)
c12 = {}
for f in glob.glob(C12C + r'\p5null_*.json'):
    for r in json.load(open(f)).get('nulls', []):
        c12[r['b']] = r['G']
g12 = np.array([c12[b] for b in sorted(c12)])
ii = {'n_null': len(G), 'G_real': Greal, 'G_free': Gfree, 'G_hsc': Ghsc, 'G_null_mean': float(G.mean()), 'G_null_sd': float(G.std(ddof=1)),
      'G_null_q95': float(np.percentile(G, 95)), 'G_null_max': float(G.max()), 'z_real': float((Greal - G.mean()) / G.std(ddof=1)),
      'R_ctx': Rctx, 'R_ctx_per_origin': [float(x) for x in Gper.mean(0) / per_real],
      'R_D_secondary': RD, 'R_D_per_origin': [float(x) for x in (Gper.mean(0) - per_pb) / (per_real - per_pb)],
      'C12_null_n': int(len(g12)), 'C12_null_mean': float(g12.mean()), 'C12_null_sd': float(g12.std(ddof=1)),
      'consistent_with_C12': bool(abs(G.mean() - g12.mean()) < 2 * g12.std(ddof=1)),
      'reject': bool(Rctx >= 0.50)}
res = {'c_i': ci_, 'c_ii': ii, 'P0c_reject': bool(ci_['reject'] or ii['reject']),
       'first_fail': 'c-i' if ci_['reject'] else ('c-ii' if ii['reject'] else None)}
json.dump(res, open(C + r'\c2_p0c.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out(json.dumps(res, ensure_ascii=False, indent=1))
