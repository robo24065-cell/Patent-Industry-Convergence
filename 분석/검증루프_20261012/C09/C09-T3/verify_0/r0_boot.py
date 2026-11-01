"""반박 검사 1: preds 파일에서 skill·짝 Δ CI 독립 재계산 (mg 군집 부트스트랩)."""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
MY = os.path.dirname(os.path.abspath(__file__)); WD = os.path.dirname(MY); CA = os.path.join(WD, 'cache')
SC = ['S0', 'S1', 'S2', 'S3', 'S4', 'S2b', 'S4b']
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def load(s, mode):
    fs = sorted(glob.glob(os.path.join(CA, f'preds_{s}_{mode}*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.drop_duplicates(['cell', 't0'], keep='last').sort_values(['t0', 'cell']).reset_index(drop=True)
out = {}
F = {s: load(s, 'final') for s in SC}
b = F['S0']
for s, d in F.items():
    assert (d.cell.values == b.cell.values).all() and (d.yb.values == b.yb.values).all()
y = b.yb.values; p0b = b.p_base.values
sk = {s: (auc(y, d.p.values) - auc(y, p0b)) / (1 - auc(y, p0b)) for s, d in F.items()}
out['final_skill'] = sk; out['final_delta'] = {s: sk[s] - sk['S0'] for s in SC}
out['n_final'] = int(len(b)); out['pos_final'] = int(y.sum())
mg = b.mg.values; ums = np.unique(mg)
grp = pd.Series(np.arange(len(mg))).groupby(mg).apply(lambda x: x.values).to_dict()
def boot(nb, seed):
    rng = np.random.default_rng(seed); res = {s: [] for s in SC}
    for _ in range(nb):
        ss = rng.choice(ums, len(ums)); ii = np.concatenate([grp[u] for u in ss])
        yy = y[ii]; a0 = auc(yy, p0b[ii])
        for s, d in F.items():
            res[s].append((auc(yy, d.p.values[ii]) - a0) / (1 - a0))
    R = {s: np.array(v) for s, v in res.items()}
    return {s: dict(ci=[float(np.quantile(R[s] - R['S0'], 0.025)), float(np.quantile(R[s] - R['S0'], 0.975))],
                    frac_le0=float(np.mean(R[s] - R['S0'] <= 0)), skill_ci=[float(np.quantile(R[s], 0.025)), float(np.quantile(R[s], 0.975))]) for s in SC if s != 'S0'} | {'S0_skill_ci': [float(np.quantile(R['S0'], 0.025)), float(np.quantile(R['S0'], 0.975))]}
out['boot_seed0_1000'] = boot(1000, 0)
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
out['boot_seed1_2000'] = boot(2000, 1)
out['boot_seed2_2000'] = boot(2000, 2)
# dev: 원점별 skill 평균
dv = {s: load(s, 'dev') for s in SC}
dsk = {}
for s, d in dv.items():
    o = {}
    for v, g in d.groupby('t0'):
        a = auc(g.yb.values, g.p.values); a0 = auc(g.yb.values, g.p_base.values); o[int(v)] = (a - a0) / (1 - a0)
    dsk[s] = o
out['dev_mean_skill'] = {s: float(np.mean(list(o.values()))) for s, o in dsk.items()}
out['dev_delta'] = {s: out['dev_mean_skill'][s] - out['dev_mean_skill']['S0'] for s in SC}
out['dev_n_origins'] = {s: len(o) for s, o in dsk.items()}
json.dump(out, open(os.path.join(MY, 'r0_boot_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ['boot_seed1_2000', 'boot_seed2_2000', 'dev_mean_skill', 'dev_delta', 'dev_n_origins']}, ensure_ascii=False, indent=1))
