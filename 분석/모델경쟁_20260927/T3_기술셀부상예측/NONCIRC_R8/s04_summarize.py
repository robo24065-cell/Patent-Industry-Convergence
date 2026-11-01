# -*- coding: utf-8 -*-
"""R8 s04: (1) 라벨 무관 mg 결과 비교, (2) 대칭 2×2(재학습) + 재학습 없는 이식, 를 요약하고 부트스트랩 CI를 붙인다.
CI = 원점마다 단위(셀/mg)를 복원추출 1,000회 → 9원점 평균 ΔAUC 의 2.5–97.5%. 개발 원점 2010–2018만."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE); E7 = os.path.join(T3, 'ESSENTIAL_R7')
RD = os.path.join(HERE, 'results'); B = 1000; rng = np.random.default_rng(20261005)
ORIG = list(range(2010, 2019))


def auc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(tag):
    return pd.read_parquet(os.path.join(RD, f'{tag}_preds.parquet'))


def boot_idx(n):
    return rng.integers(0, n, (B, n))


def paired(dfA, dfB, colA='p', colB='p'):
    """같은 평가 행(같은 과제)의 두 점수. 반환: 평균 AUC A/B, Δ, CI, 우세 원점 수"""
    out = dict(aA=[], aB=[], bs=np.zeros((len(ORIG), B)))
    for i, v in enumerate(ORIG):
        a = dfA[dfA.t0 == v].sort_values('cell'); b = dfB[dfB.t0 == v].sort_values('cell')
        assert (a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all()
        y = a.yb.values; pa = a[colA].values; pb = b[colB].values
        out['aA'].append(auc(y, pa)); out['aB'].append(auc(y, pb))
        I = boot_idx(len(y))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            if yy.sum() in (0, len(yy)): out['bs'][i, j] = np.nan; continue
            out['bs'][i, j] = auc(yy, pa[ii]) - auc(yy, pb[ii])
    d = np.array(out['aA']) - np.array(out['aB']); bm = np.nanmean(out['bs'], 0)
    return dict(aucA=float(np.mean(out['aA'])), aucB=float(np.mean(out['aB'])), dAUC=float(d.mean()),
                ci=[float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))], wins=int((d > 0).sum()), n_orig=len(ORIG),
                by_origin=[round(float(x), 4) for x in d], boot=bm)


def skill(a, b): return (a - b) / (1 - b)


R = {}
J = lambda tag: json.load(open(os.path.join(RD, f'{tag}.json'), encoding='utf-8'))
# ---------------- (1) 라벨 무관 mg ----------------
mg = {k: load(f'mg_ctx-{k}') for k in ['none', 'prov', 'conc', 'concm'] if os.path.exists(os.path.join(RD, f'mg_ctx-{k}_preds.parquet'))}
for k in list(mg):
    for extra in ['_lite', '_s3']:
        if os.path.exists(os.path.join(RD, f'mg_ctx-{k}{extra}_preds.parquet')): mg[k + extra] = load(f'mg_ctx-{k}{extra}')
base = mg['none'].copy(); base['p'] = base.p_base
tab1 = []
for k, d in mg.items():
    r = paired(d, base); j = J(f'mg_ctx-{k}')
    tab1.append(dict(model=k, auc=round(r['aucA'], 4), base_auc=round(r['aucB'], 4), skill=round(j['summary']['mean_skill'], 4),
                     p10=round(j['summary']['mean_p10'], 4), n=round(j['summary']['mean_n'], 1), evt=round(j['summary']['mean_evt'], 1)))
tab1 = pd.DataFrame(tab1); print(tab1.to_string()); R['mg_models'] = tab1.to_dict('records')
pairs = [('prov', 'conc'), ('prov', 'none'), ('conc', 'none'), ('concm', 'none'), ('prov', 'concm'),
         ('prov_lite', 'conc_lite'), ('prov_lite', 'none'), ('conc_lite', 'none'), ('prov_s3', 'conc_s3')]
R['mg_pairs'] = {}
for A, Bn in pairs:
    if A in mg and Bn in mg:
        r = paired(mg[A], mg[Bn]); r.pop('boot')
        R['mg_pairs'][f'{A}_vs_{Bn}'] = r
        print(f"mg {A} vs {Bn}: {r['aucA']:.4f} vs {r['aucB']:.4f} Δ{r['dAUC']:+.4f} CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] 우세 {r['wins']}/9")

# ---------------- (2) 대칭 2×2 (27코드 해상도, 재학습) ----------------
T = {}
for tag in ['pcode_feat-pcode', 'pcode_feat-conc', 'conc_feat-conc', 'conc_feat-pcode']:
    if os.path.exists(os.path.join(RD, f'{tag}_preds.parquet')): T[tag] = load(tag)
if len(T) == 4:
    rP = paired(T['pcode_feat-pcode'], T['pcode_feat-conc']); rC = paired(T['conc_feat-pcode'], T['conc_feat-conc'])
    both = (rP['boot'] + rC['boot']) / 2
    R['sym2x2'] = dict(
        cells={f'{t}': dict(auc=round(J(t)['summary']['mean_auc'], 4), base_auc=round(J(t)['summary']['mean_base_auc'], 4),
                            skill=round(J(t)['summary']['mean_skill'], 4), p10=round(J(t)['summary']['mean_p10'], 4), n=J(t)['summary']['mean_n']) for t in T},
        P27_outcome_prov_minus_conc=dict(dAUC=rP['dAUC'], ci=rP['ci'], wins=rP['wins'], by_origin=rP['by_origin']),
        C27_outcome_prov_minus_conc=dict(dAUC=rC['dAUC'], ci=rC['ci'], wins=rC['wins'], by_origin=rC['by_origin']),
        both_outcomes_avg_prov_minus_conc=dict(dAUC=float((rP['dAUC'] + rC['dAUC']) / 2), ci=[float(np.nanpercentile(both, 2.5)), float(np.nanpercentile(both, 97.5))],
                                              wins=int((np.array(rP['by_origin']) + np.array(rC['by_origin']) > 0).sum())),
        home_advantage=dict(P27=rP['dAUC'], C27=-rC['dAUC']))
    print(json.dumps({k: v for k, v in R['sym2x2'].items() if k != 'cells'}, ensure_ascii=False, indent=1))
    for t in T: print(t, R['sym2x2']['cells'][t])

# ---------------- (2b) 재학습 없는 이식 ----------------
def norm_rank(df):
    df = df.copy(); df['r'] = df.groupby('t0').p.rank(pct=True); return df


def transfer_cell(home, other_lab):
    """home 모형 점수를 other 정의 레이블로: 양쪽 적격 교집합에서 home 점수 vs other 자기 모형 점수"""
    out = []
    for v in ORIG:
        h = home[home.t0 == v][['cell', 'p']].rename(columns={'p': 'ph'}); o = other_lab[other_lab.t0 == v][['cell', 'yb', 'p']]
        m = o.merge(h, on='cell')
        out.append(dict(t0=v, n=len(m), n_other=len(o), auc_home_model=auc(m.yb, m.ph), auc_own_model=auc(m.yb, m.p)))
    return pd.DataFrame(out)


R['transfer'] = {}
if len(T) == 4:
    a = transfer_cell(T['pcode_feat-pcode'], T['conc_feat-conc']); b = transfer_cell(T['conc_feat-conc'], T['pcode_feat-pcode'])
    R['transfer']['P27_model_on_C27_labels'] = dict(auc_P27model=float(a.auc_home_model.mean()), auc_C27model=float(a.auc_own_model.mean()), n_mean=float(a.n.mean()), cover=float((a.n / a.n_other).mean()))
    R['transfer']['C27_model_on_P27_labels'] = dict(auc_C27model=float(b.auc_home_model.mean()), auc_P27model=float(b.auc_own_model.mean()), n_mean=float(b.n.mean()), cover=float((b.n / b.n_other).mean()))
    print(json.dumps(R['transfer'], ensure_ascii=False, indent=1))


def to_mg(cellpred, mgref, agg='max'):
    """셀 점수(원점별 백분위) → mg 점수(최댓값). 적격 셀이 없는 mg 는 0(최하)."""
    c = norm_rank(cellpred); g = c.groupby(['t0', 'mg']).r.max().rename('s').reset_index()
    m = mgref[['t0', 'mg', 'cell', 'yb']].merge(g, on=['t0', 'mg'], how='left')
    m['covered'] = m.s.notna(); m['s'] = m.s.fillna(0.0); return m


if 'none' in mg:
    ref = mg['none']
    src = {'P50_full(R7 동결 재현)': pd.read_parquet(os.path.join(E7, 'results', 'pub_full_preds.parquet')),
           'C27_task(R7 conc_task)': pd.read_parquet(os.path.join(E7, 'results', 'pub_conc_task_preds.parquet'))}
    for t in ['pcode_feat-pcode', 'conc_feat-conc']:
        if t in T: src[t] = T[t]
    M = {k: to_mg(v, ref) for k, v in src.items()}
    rows = []
    for k, m in M.items():
        a_all = np.mean([auc(m[m.t0 == v].yb, m[m.t0 == v].s) for v in ORIG]); cov = m.groupby('t0').covered.mean().mean()
        rows.append(dict(source=k, mg_auc_all=round(float(a_all), 4), mg_coverage=round(float(cov), 4)))
    # 양쪽 다 덮는 mg 에서 비교 (P50 vs C27, P27 vs C27)
    for A, Bn in [('P50_full(R7 동결 재현)', 'C27_task(R7 conc_task)'), ('pcode_feat-pcode', 'conc_feat-conc')]:
        if A in M and Bn in M:
            mA, mB = M[A], M[Bn]; aa, bb = [], []
            for v in ORIG:
                x = mA[mA.t0 == v].merge(mB[mB.t0 == v][['mg', 's', 'covered']], on='mg', suffixes=('', '_b'))
                x = x[x.covered & x.covered_b]; aa.append(auc(x.yb, x.s)); bb.append(auc(x.yb, x.s_b))
            rows.append(dict(source=f'{A} vs {Bn} (공통 덮인 mg)', mg_auc_all=f'{np.mean(aa):.4f} vs {np.mean(bb):.4f}', mg_coverage=f'우세 {int((np.array(aa) > np.array(bb)).sum())}/9'))
    tt = pd.DataFrame(rows); print(tt.to_string()); R['transfer_to_mg'] = tt.to_dict('records')
    R['mg_baseline_auc'] = float(np.mean([auc(ref[ref.t0 == v].yb, ref[ref.t0 == v].p_base) for v in ORIG]))
json.dump(R, open(os.path.join(HERE, 'summary_r8.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('saved summary_r8.json')
