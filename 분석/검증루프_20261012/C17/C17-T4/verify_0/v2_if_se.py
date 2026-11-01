# C17-T4 verify v2: ΔAUC 군집 SE를 영향함수(해석적)로 재계산(s6 부트스트랩 코드 미사용).
import json, numpy as np, pandas as pd
from scipy.stats import norm, rankdata
W = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C17/C17-T4'
P = pd.read_parquet(W + '/cache/pred_2023_xgb.parquet')
best = json.load(open(W + '/cache/s6_p0b_xgb.json'))['best_public']
NL = {2: 4171, 3: 2756}


def auc_if(p, y):
    # 순위 기반 AUC와 관측별 영향함수 (DeLong 성분)
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    r_all = rankdata(p); r_pos = rankdata(p[pos]); r_neg = rankdata(p[~pos])
    auc = (r_all[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    v10 = (r_all[pos] - r_pos) / n0          # 양성 i: 음성보다 높은 비율
    v01 = 1 - (r_all[~pos] - r_neg) / n1     # 음성 j: 양성보다 낮은 비율
    IF = np.empty(len(p))
    IF[pos] = (v10 - auc) / n1
    IF[~pos] = (v01 - auc) / n0
    return auc, IF


def cl_var(IF, g):
    s = pd.Series(IF).groupby(np.asarray(g)).sum().values
    G = len(s)
    return G / (G - 1) * np.sum(s ** 2)


def run(d):
    y = d.y.values.astype(int)
    a1, i1 = auc_if(d['prov'].values, y); a2, i2 = auc_if(d[best].values, y)
    IF = i1 - i2
    cell = d.firm + '|' + d.k_prov
    V = {c: cl_var(IF, g) for c, g in [('firm', d.firm), ('ksic', d.k_prov), ('cell', cell)]}
    se_cgm = np.sqrt(max(V['firm'] + V['ksic'] - V['cell'], max(V['firm'], V['ksic'])))
    return {'dAUC': a1 - a2, 'se_firm': np.sqrt(V['firm']), 'se_ksic': np.sqrt(V['ksic']), 'se_cell': np.sqrt(V['cell']), 'se_cgm': se_cgm,
            'se_gate': max(se_cgm, np.sqrt(V['ksic']))}


out = {'full2023': run(P)}
rows = []
for i in range(20):
    d = pd.concat([P[P.s == s].sample(NL[s], random_state=20261017 + i) for s in (2, 3)], ignore_index=True)
    rows.append(run(d))
R = pd.DataFrame(rows)
out['sub_mean'] = R.mean().to_dict()
out['power_0.002'] = float(norm.cdf(0.002 / R.se_gate.mean() - 1.96))
out['power_full_size_0.002'] = float(norm.cdf(0.002 / out['full2023']['se_gate'] - 1.96))
json.dump(out, open(W + '/verify_0/v2_out.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
