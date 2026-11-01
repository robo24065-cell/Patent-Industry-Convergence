"""s10: 시나리오 결과 취합 + IPC 메인그룹 군집 부트스트랩(C09-T3 s4_compare 방식: seed=0, 시나리오 간 짝) → t3_sens.csv, cache/s10_results.json
판정(PREREG §3): 개발 평균 |Δskill| < 0.005 그리고 최종 |Δskill| < 0.005 → '영향 없음', 아니면 '영향 있음'."""
import sys, os, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
SC = ['S0', 'S1', 'S2a', 'S2b']
THR = 0.005


def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(scen, mode):
    fs = sorted(glob.glob(os.path.join(CACHE, f'preds_{scen}_{mode}*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.drop_duplicates(['cell', 't0'], keep='last').sort_values(['t0', 'cell']).reset_index(drop=True)


def boot(blocks_by_scen, n_boot=1000, seed=0):
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
for mode in ['final', 'dev']:
    P = {s: load(s, mode) for s in SC}
    base = P['S0']
    for k, v in P.items():
        assert len(v) == len(base) and (v.cell.values == base.cell.values).all() and (v.t0.values == base.t0.values).all(), k
        assert np.allclose(v.p_base.values, base.p_base.values) and (v.yb.values == base.yb.values).all(), k
    origins = [int(x) for x in sorted(base.t0.unique())]
    off = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet' if mode == 'final' else 'dev_preds_r3.parquet'))
    m = base.merge(off, on=['cell', 't0'], suffixes=('', '_off'))
    R[f'{mode}_S0_vs_official'] = dict(n_rows_mine=int(len(base)), n_rows_official=int(len(off)), n_merged=int(len(m)),
                                       p_maxdiff=float(np.abs(m.p - m.p_off).max()), yb_mismatch=int((m.yb != m.yb_off).sum()))
    # C09-T3 S0 예측과도 대조
    c09 = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(C09, 'cache', f'preds_S0_{mode}*.parquet'))]).drop_duplicates(['cell', 't0'])
    m2 = base.merge(c09[['cell', 't0', 'p']], on=['cell', 't0'], suffixes=('', '_c09'))
    R[f'{mode}_S0_vs_C09'] = dict(n_merged=int(len(m2)), p_maxdiff=float(np.abs(m2.p - m2.p_c09).max()))
    blocks = {s: [(d[d.t0 == v].mg.values, d[d.t0 == v].yb.values, d[d.t0 == v].p.values, d[d.t0 == v].p_base.values) for v in origins] for s, d in P.items()}
    B = boot(blocks)
    s0 = None
    for s, d in P.items():
        sk = {}
        for v in origins:
            g = d[d.t0 == v]; a = auc(g.yb.values, g.p.values); a0 = auc(g.yb.values, g.p_base.values); sk[v] = (a - a0) / (1 - a0)
        if s == 'S0': s0 = sk
        msk = float(np.mean(list(sk.values())))
        col = 3 if mode == 'dev' else 2
        ci = np.percentile(B[s][:, col], [2.5, 97.5]); dci = np.percentile(B[s][:, col] - B['S0'][:, col], [2.5, 97.5])
        row = dict(scenario=s, mode=mode, origins=f'{origins[0]}-{origins[-1]}' if len(origins) > 1 else str(origins[0]),
                   skill=msk, skill_ci_lo=ci[0], skill_ci_hi=ci[1], delta_vs_S0=msk - float(np.mean(list(s0.values()))), delta_ci_lo=dci[0], delta_ci_hi=dci[1],
                   auc=float(np.mean([auc(d[d.t0 == v].yb.values, d[d.t0 == v].p.values) for v in origins])),
                   rank_corr_vs_S0=float(np.mean([spearmanr(d[d.t0 == v].p, base[base.t0 == v].p).statistic for v in origins])),
                   top10_overlap=float(np.mean([len(set(d[d.t0 == v].nlargest(int(round(0.1 * (d.t0 == v).sum())), 'p').cell) &
                                                     set(base[base.t0 == v].nlargest(int(round(0.1 * (base.t0 == v).sum())), 'p').cell)) / int(round(0.1 * (d.t0 == v).sum())) for v in origins])))
        if mode == 'dev':
            row['delta_by_origin'] = ' '.join(f'{v}:{sk[v] - s0[v]:+.4f}' for v in origins)
            row['n_origins_up'] = int(sum(sk[v] > s0[v] for v in origins))
        rows.append(row)
SR = pd.DataFrame(rows)
ver = []
for s in SC:
    f = SR[(SR.scenario == s) & (SR['mode'] == 'final')].iloc[0]; d = SR[(SR.scenario == s) & (SR['mode'] == 'dev')].iloc[0]
    if s == 'S0': v = '기준'
    else: v = '영향 없음' if (abs(f.delta_vs_S0) < THR and abs(d.delta_vs_S0) < THR) else '영향 있음'
    ver.append(dict(scenario=s, verdict=v))
SR = SR.merge(pd.DataFrame(ver), on='scenario')
SR.round(5).to_csv(os.path.join(WD, 't3_sens.csv'), index=False, encoding='utf-8-sig')
R['verdicts'] = ver
json.dump(R, open(os.path.join(CACHE, 's10_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(SR.round(4).to_string()); print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
