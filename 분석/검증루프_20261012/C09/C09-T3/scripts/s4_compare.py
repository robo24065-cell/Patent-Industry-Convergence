"""s4: 시나리오 결과 취합 + IPC 메인그룹 군집 부트스트랩(공식 cluster_boot 와 같은 seed=0 재표본, 시나리오 간 짝) → scenario_result.csv, cache/s4_results.json
사용: python s4_compare.py final|dev|both"""
import sys, os, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
SC = ['S0', 'S1', 'S2', 'S3', 'S4', 'S2b', 'S4b']
OFF_CI = (0.36580365189908404, 0.47307183171311484)
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def load(scen, mode):
    fs = sorted(glob.glob(os.path.join(CACHE, f'preds_{scen}_{mode}*.parquet')))
    if not fs: return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.drop_duplicates(['cell', 't0'], keep='last').sort_values(['t0', 'cell']).reset_index(drop=True)
def boot(blocks_by_scen, n_boot=1000, seed=0):
    """공식 final_run_r3.cluster_boot 와 같은 재표본 순서. blocks_by_scen: {scen: [(mg, yb, p, p0) 원점별]}. 반환 {scen: 배열(n_boot, [auc, auc0, skill_pooled, skill_mean_origins])}"""
    sc0 = list(blocks_by_scen)[0]; blocks0 = blocks_by_scen[sc0]
    rng = np.random.default_rng(seed); ums = np.unique(np.concatenate([b[0] for b in blocks0]))
    gi = [pd.Series(np.arange(len(b[0]))).groupby(b[0]).apply(lambda s: s.values).to_dict() for b in blocks0]
    out = {s: [] for s in blocks_by_scen}
    for _ in range(n_boot):
        s_ = rng.choice(ums, len(ums))
        idx = [np.concatenate([g[u] for u in s_ if u in g]) for g in gi]
        for sc, blocks in blocks_by_scen.items():
            am, ab = [], []
            for b, ii in zip(blocks, idx):
                y = b[1][ii]
                if y.min() == y.max(): continue
                am.append(auc(y, b[2][ii])); ab.append(auc(y, b[3][ii]))
            a, a0 = np.mean(am), np.mean(ab)
            out[sc].append((a, a0, (a - a0) / (1 - a0), np.mean([(x - y0) / (1 - y0) for x, y0 in zip(am, ab)])))
    return {k: np.array(v) for k, v in out.items()}
R = {}; rows = []
modes = ['final', 'dev'] if which == 'both' else [which]
for mode in modes:
    P = {s: load(s, mode) for s in SC}; P = {k: v for k, v in P.items() if v is not None}
    base = P['S0']
    for k, v in P.items():
        assert len(v) == len(base) and (v.cell.values == base.cell.values).all() and (v.t0.values == base.t0.values).all(), k
        assert np.allclose(v.p_base.values, base.p_base.values) and (v.yb.values == base.yb.values).all(), k
    origins = [int(x) for x in sorted(base.t0.unique())]
    # S0 재현: 공식 예측 파일 대조
    off = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet' if mode == 'final' else 'dev_preds_r3.parquet'))
    m = base.merge(off, on=['cell', 't0'], suffixes=('', '_off'))
    R[f'{mode}_S0_vs_official'] = dict(n_rows_mine=int(len(base)), n_rows_official=int(len(off)), n_merged=int(len(m)),
        p_maxdiff=float(np.abs(m.p - m.p_off).max()), p_spearman=float(spearmanr(m.p, m.p_off).statistic),
        pbase_maxdiff=float(np.abs(m.p_base - m.p_base_off).max()), yb_mismatch=int((m.yb != m.yb_off).sum()))
    blocks = {s: [(d[d.t0 == v].mg.values, d[d.t0 == v].yb.values, d[d.t0 == v].p.values, d[d.t0 == v].p_base.values) for v in origins] for s, d in P.items()}
    B = boot(blocks)
    for s, d in P.items():
        sk = {}
        for v in origins:
            g = d[d.t0 == v]; a = auc(g.yb.values, g.p.values); a0 = auc(g.yb.values, g.p_base.values); sk[int(v)] = (a - a0) / (1 - a0)
        msk = float(np.mean(list(sk.values()))); s0 = R.get(f'{mode}_S0_skill_by_origin')
        if s == 'S0': R[f'{mode}_S0_skill_by_origin'] = sk; s0 = sk
        dlt = {v: sk[v] - s0[v] for v in origins}
        col = 3 if mode == 'dev' else 2
        ci = np.percentile(B[s][:, col], [2.5, 97.5]); dci = np.percentile(B[s][:, col] - B['S0'][:, col], [2.5, 97.5])
        row = dict(scenario=s, mode=mode, origins=f'{origins[0]}-{origins[-1]}' if len(origins) > 1 else str(origins[0]),
                   skill=msk, skill_ci_lo=ci[0], skill_ci_hi=ci[1], delta_vs_S0=msk - float(np.mean(list(s0.values()))),
                   delta_ci_lo=dci[0], delta_ci_hi=dci[1],
                   auc=float(np.mean([auc(d[d.t0 == v].yb.values, d[d.t0 == v].p.values) for v in origins])),
                   p_at10=float(np.mean([d[d.t0 == v].yb.values[np.argsort(-d[d.t0 == v].p.values)[:int(round(0.1 * (d.t0 == v).sum()))]].mean() for v in origins])),
                   rank_corr_vs_S0=float(np.mean([spearmanr(d[d.t0 == v].p, base[base.t0 == v].p).statistic for v in origins])))
        if mode == 'dev':
            row['delta_early_2010_2013'] = float(np.mean([dlt[v] for v in origins if v <= 2013]))
            row['delta_late_2014_2018'] = float(np.mean([dlt[v] for v in origins if v >= 2014]))
            row['delta_by_origin'] = ' '.join(f'{v}:{dlt[v]:+.4f}' for v in origins)
        rows.append(row)
SR = pd.DataFrame(rows)
# 판정(PREREG §4): S1–S4
if set(['final', 'dev']).issubset(set(SR['mode'])):
    ver = []
    for s in SR.scenario.unique():
        f = SR[(SR.scenario == s) & (SR['mode'] == 'final')].iloc[0]; d = SR[(SR.scenario == s) & (SR['mode'] == 'dev')].iloc[0]
        A = (f.skill < OFF_CI[0]) or (f.skill > OFF_CI[1]) or (f.skill_ci_hi < OFF_CI[0]) or (f.skill_ci_lo > OFF_CI[1])
        Bc = abs(d.delta_vs_S0) >= 0.02
        sep = not (f.delta_ci_lo <= 0 <= f.delta_ci_hi)
        spec = (abs(f.delta_vs_S0 - d.delta_vs_S0) >= 0.02) and sep
        ver.append(dict(scenario=s, crit_A=bool(A), crit_B=bool(Bc), verdict=('기준 시나리오' if s == 'S0' else ('단서 필요' if (A or Bc) else '영향 없음')) + (' (보조, 판정 외)' if s in ('S2b', 'S4b') else ''),
                        final_delta_separable=bool(sep), final_specific=bool(spec)))
    V = pd.DataFrame(ver); SR = SR.merge(V, on='scenario', how='left')
    R['verdicts'] = V.to_dict('records')
out = os.path.join(WD, 'scenario_result.csv') if which == 'both' else os.path.join(CACHE, f'scenario_result_{which}.csv')
SR.round(5).to_csv(out, index=False, encoding='utf-8-sig')
json.dump(R, open(os.path.join(CACHE, f's4_results_{which}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(SR.drop(columns=[c for c in ['delta_by_origin'] if c in SR]).round(4).to_string()); print(json.dumps(R, ensure_ascii=False, indent=1, default=float)[:3000])
