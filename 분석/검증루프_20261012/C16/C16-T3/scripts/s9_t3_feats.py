"""s9: 시나리오 피처 표(셀×t0) + S0 재계산 대조 (C09-T3 s2_feats 방식).
사용: python -I s9_t3_feats.py dev|fin → cache/t3feats_{which}.parquet, cache/s9_check_{which}.json
r3lib.feats 정의: all_newapp_sh = W(Σcc n_newapp_apps)/max(W(Σcc n),1), all_newapp_n = log1p(W(Σcc n_newapp)), W = t0-2..t0 합.
S1  CN 성분 재가중: 조직층 w_o(y)=s_ref/s_y, 개인층 w_i(y)=(1-s_ref)/(1-s_y), 결측 문서 가중 1. s_ref = CN 2010–2018 풀링 조직 비중.
    (구현 규칙) 연도 비결측 CN 문서 <1000 또는 s_y∈{0,1}이면 그 연도 가중 1(2006–09: 출원인 결측 ≈100%).
S2a CN 제외 4개국. S2b CN 성분은 출원인1 조직 문서만(분자·분모).
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
which = sys.argv[1]
src = CP_DEV if which == 'dev' else CP_FIN
T0S = list(range(2005, 2019)) + ([2021] if which == 'fin' else [])
Y0 = 2000; Y1 = 2021 if which == 'dev' else 2024
CCS = ['KR', 'US', 'CN', 'JP', 'EP']; CN = 2
off = pd.read_parquet(os.path.join(src, 'panel_ccy.parquet'))
mine = pd.read_parquet(os.path.join(CACHE, f't3panel_{which}.parquet'))
for d in (off, mine): d['cell'] = d.ksic + '|' + d.mg
cells = np.array(sorted(off.cell.unique()))
chk = dict(n_cells_official=int(len(cells)), cells_equal=bool(set(cells) == set(mine.cell.unique())))
m = off.merge(mine, on=['ksic', 'mg', 'cc', 'year'], how='outer', suffixes=('_o', '_m'), indicator=True)
chk['row_merge'] = {str(k): int(v) for k, v in m._merge.value_counts().items()}
for c in ['n', 'n_app', 'n_newapp_apps', 'n_newapp']:
    chk[f'ndiff_{c}'] = int((m[c + '_o'].fillna(0) != m[c + '_m'].fillna(0)).sum())
cn = mine[mine.cc == 'CN']
chk['cn_identity_newapp_apps'] = int((cn.nna_org + cn.nna_ind != cn.n_newapp_apps).sum())
chk['cn_identity_newapp'] = int((cn.nn_org + cn.nn_ind != cn.n_newapp).sum())
chk['cn_identity_n'] = int((cn.n_org + cn.n_ind + cn.n_miss != cn.n).sum())
# 연도 가중
sh = pd.read_csv(os.path.join(CACHE, f't3_cn_orgshare_{which}.csv')).set_index('year')
ref = sh.loc[2010:2018]; s_ref = float(ref.n_org.sum() / (ref.n_org.sum() + ref.n_ind.sum()))
wo = {}; wi = {}
for y in range(Y0, Y1 + 1):
    if y in sh.index and (sh.n_org[y] + sh.n_ind[y]) >= 1000 and 0 < sh.org_share[y] < 1:
        wo[y] = s_ref / sh.org_share[y]; wi[y] = (1 - s_ref) / (1 - sh.org_share[y])
    else:
        wo[y] = 1.0; wi[y] = 1.0
chk['s_ref'] = s_ref; chk['w_org'] = {int(k): round(v, 4) for k, v in wo.items() if v != 1}; chk['w_ind'] = {int(k): round(v, 4) for k, v in wi.items() if v != 1}
cid = {c: i for i, c in enumerate(cells)}
mine = mine[mine.year.between(Y0, Y1)]
T = Y1 - Y0 + 1


def arr(col):
    a = np.zeros((len(cells), len(CCS), T), dtype=np.float64)
    s = mine[['cell', 'cc', 'year', col]].dropna()
    a[s.cell.map(cid).values, s.cc.map({c: i for i, c in enumerate(CCS)}).values, s.year.values - Y0] = s[col].values
    return a


A = {c: arr(c) for c in ['n', 'n_newapp_apps', 'n_newapp', 'n_org', 'n_ind', 'n_miss', 'nna_org', 'nna_ind', 'nn_org', 'nn_ind']}
WO = np.array([wo[y] for y in range(Y0, Y1 + 1)]); WI = np.array([wi[y] for y in range(Y0, Y1 + 1)])


def W(x, t0): return x[..., t0 - 2 - Y0:t0 - Y0 + 1].sum(-1)


non = [0, 1, 3, 4]
rows = []
for t0 in T0S:
    d = dict(cell=cells, t0=t0)
    tot = W(A['n'].sum(1), t0)
    d['all_newapp_sh_S0'] = W(A['n_newapp_apps'].sum(1), t0) / np.maximum(tot, 1)
    d['all_newapp_n_S0'] = np.log1p(W(A['n_newapp'].sum(1), t0))
    # S1
    cn_nna = A['nna_org'][:, CN] * WO + A['nna_ind'][:, CN] * WI
    cn_n = A['n_org'][:, CN] * WO + A['n_ind'][:, CN] * WI + A['n_miss'][:, CN]
    cn_nn = A['nn_org'][:, CN] * WO + A['nn_ind'][:, CN] * WI
    nonsum = lambda col: A[col][:, non].sum(1)
    d['all_newapp_sh_S1'] = W(nonsum('n_newapp_apps') + cn_nna, t0) / np.maximum(W(nonsum('n') + cn_n, t0), 1)
    d['all_newapp_n_S1'] = np.log1p(W(nonsum('n_newapp') + cn_nn, t0))
    # S2a
    d['all_newapp_sh_S2a'] = W(nonsum('n_newapp_apps'), t0) / np.maximum(W(nonsum('n'), t0), 1)
    d['all_newapp_n_S2a'] = np.log1p(W(nonsum('n_newapp'), t0))
    # S2b
    d['all_newapp_sh_S2b'] = W(nonsum('n_newapp_apps') + A['nna_org'][:, CN], t0) / np.maximum(W(nonsum('n') + A['n_org'][:, CN], t0), 1)
    d['all_newapp_n_S2b'] = np.log1p(W(nonsum('n_newapp') + A['nn_org'][:, CN], t0))
    d['cn_sh'] = W(A['n'][:, CN], t0) / np.maximum(tot, 1)
    d['cn_ind_sh'] = W(A['n_ind'][:, CN], t0) / np.maximum(W(A['n_org'][:, CN] + A['n_ind'][:, CN], t0), 1)
    rows.append(pd.DataFrame(d))
F = pd.concat(rows, ignore_index=True)
mx = {}
for t0 in T0S:
    f = pd.read_parquet(os.path.join(src, f'feat_{t0}.parquet'), columns=['cell', 't0', 'all_newapp_sh', 'all_newapp_n'])
    g = f.merge(F[F.t0 == t0], on=['cell', 't0'], how='left')
    for a, b in [('all_newapp_sh', 'all_newapp_sh_S0'), ('all_newapp_n', 'all_newapp_n_S0')]:
        mx[a] = max(mx.get(a, 0.0), float(np.nanmax(np.abs(g[a].values - g[b].values))))
chk['maxdiff_vs_official_feat'] = mx
# 시나리오 변화 크기(적격 셀 근사: 공식 피처 캐시 셀)
desc = {}
for s in ['S1', 'S2a', 'S2b']:
    for f_ in ['all_newapp_sh', 'all_newapp_n']:
        dd = (F[f'{f_}_{s}'] - F[f'{f_}_S0'])
        desc[f'{f_}_{s}'] = dict(mean_abs=float(dd.abs().mean()), by_t0_mean={int(t): float(v) for t, v in dd.groupby(F.t0).mean().items()},
                                 spearman_vs_S0_by_t0={int(t): float(g[f'{f_}_{s}'].rank().corr(g[f'{f_}_S0'].rank())) for t, g in F.groupby('t0')})
chk['scenario_shift'] = desc
F.to_parquet(os.path.join(CACHE, f't3feats_{which}.parquet'))
json.dump(chk, open(os.path.join(CACHE, f's9_check_{which}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in chk.items() if k != 'scenario_shift'}, ensure_ascii=False, indent=1, default=str))
for k, v in desc.items(): print(k, round(v['mean_abs'], 4), {t: round(x, 3) for t, x in v['spearman_vs_S0_by_t0'].items() if t in (2005, 2010, 2015, 2018, 2021)})
