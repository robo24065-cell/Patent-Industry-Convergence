"""s2: 시나리오 피처 표(셀×t0) + S0 재계산 일치 점검.
사용: python s2_feats.py dev|fin  → cache/scen_feats_{dev|fin}.parquet, cache/s2_check_{which}.json
r3lib.py 125–131행 정의를 그대로 따른다: share(col, allcc) = W(Σcc col)/max(W(tot),1), all_newapp_n = log1p(W(Σcc n_newapp)).
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
which = sys.argv[1]
src = CP_DEV if which == 'dev' else CP_FIN
T0S = list(range(2005, 2019)) + ([2021] if which == 'fin' else [])
Y0 = 2000; Y1 = 2021 if which == 'dev' else 2024
CCS = ['KR', 'US', 'CN', 'JP', 'EP']
off = pd.read_parquet(os.path.join(src, 'panel_ccy.parquet'))
mine = pd.read_parquet(os.path.join(CACHE, f'panel_scen_{which}.parquet'))
for d in (off, mine): d['cell'] = d.ksic + '|' + d.mg
cells = np.array(sorted(off.cell.unique()))
chk = dict(n_cells_official=int(len(cells)), n_cells_mine=int(mine.cell.nunique()), cells_equal=bool(set(cells) == set(mine.cell.unique())))
# 공식 열과 행 단위 대조
m = off.merge(mine, on=['ksic', 'mg', 'cc', 'year'], how='outer', suffixes=('_o', '_m'), indicator=True)
chk['row_merge'] = m._merge.value_counts().to_dict()
for c in ['n', 'n_app', 'n_newapp_apps', 'n_newapp']:
    chk[f'maxdiff_{c}'] = float(np.nanmax(np.abs(m[c + '_o'].fillna(0) - m[c + '_m'].fillna(0))))
    chk[f'ndiff_{c}'] = int((m[c + '_o'].fillna(0) != m[c + '_m'].fillna(0)).sum())
cid = {c: i for i, c in enumerate(cells)}
mine = mine[mine.year.between(Y0, Y1)]
T = Y1 - Y0 + 1
def arr(col):
    a = np.zeros((len(cells), len(CCS), T), dtype=np.float32)
    s = mine[['cell', 'cc', 'year', col]].dropna()
    a[s.cell.map(cid).values, s.cc.map({c: i for i, c in enumerate(CCS)}).values, s.year.values - Y0] = s[col].values
    return a
A = {c: arr(c) for c in ['n', 'n_null', 'n_usold', 'n_cnmiss59', 'n_corp', 'n_app', 'n_newapp_apps', 'n_newapp', 'n_newapp_apps_s1', 'n_newapp_s1', 'n_newapp_apps_corp']}
def W(x, t0): return x[..., t0 - 2 - Y0:t0 - Y0 + 1].sum(-1)
rows = []
for t0 in T0S:
    tot = W(A['n'].sum(1), t0)
    d = dict(cell=cells, t0=t0, tot_W_m=tot)
    d['all_newapp_sh_S0'] = W(A['n_newapp_apps'].sum(1), t0) / np.maximum(tot, 1)
    d['all_newapp_n_S0'] = np.log1p(W(A['n_newapp'].sum(1), t0))
    d['kr_newapp_sh_S0m'] = W(A['n_newapp_apps'][:, 0], t0) / np.maximum(W(A['n'][:, 0], t0), 1)
    d['kr_newapp_n_S0m'] = np.log1p(W(A['n_newapp'][:, 0], t0))
    usold = W(A['n_usold'].sum(1), t0)
    d['all_newapp_sh_S1'] = W(A['n_newapp_apps_s1'].sum(1), t0) / np.maximum(tot - usold, 1)
    d['all_newapp_n_S1'] = np.log1p(W(A['n_newapp_s1'].sum(1), t0))
    d['all_newapp_sh_S2'] = W(A['n_newapp_apps'].sum(1), t0) / np.maximum(tot - W(A['n_cnmiss59'].sum(1), t0), 1)
    d['all_newapp_sh_S2b'] = W(A['n_newapp_apps'].sum(1), t0) / np.maximum(tot - W(A['n_null'].sum(1), t0), 1)
    # 진단용
    d['us_sh'] = W(A['n'][:, 1], t0) / np.maximum(tot, 1)
    d['usold_sh'] = usold / np.maximum(tot, 1)
    d['cnmiss_sh'] = W(A['n_cnmiss59'].sum(1), t0) / np.maximum(tot, 1)
    d['null_sh'] = W(A['n_null'].sum(1), t0) / np.maximum(tot, 1)
    d['jp_null_sh'] = W(A['n_null'][:, 3], t0) / np.maximum(tot, 1)
    for j, c in enumerate(CCS):
        nn = W(A['n'][:, j], t0)
        d[f'{c}_newapp_sh'] = np.where(nn > 0, W(A['n_newapp_apps'][:, j], t0) / np.maximum(nn, 1), np.nan)
    nc = W(A['n_corp'][:, 1], t0)
    d['US_newapp_sh_corp'] = np.where(nc > 0, W(A['n_newapp_apps_corp'][:, 1], t0) / np.maximum(nc, 1), np.nan)
    d['US_corp_sh'] = np.where(W(A['n'][:, 1], t0) > 0, nc / np.maximum(W(A['n'][:, 1], t0), 1), np.nan)
    rows.append(pd.DataFrame(d))
F = pd.concat(rows, ignore_index=True)
# 공식 피처 캐시와 대조(적격 ≥10 행만 캐시에 있음)
mx = {}
for t0 in T0S:
    f = pd.read_parquet(os.path.join(src, f'feat_{t0}.parquet'), columns=['cell', 't0', 'tot_W', 'all_newapp_sh', 'all_newapp_n', 'kr_newapp_sh', 'kr_newapp_n'])
    g = f.merge(F[F.t0 == t0], on=['cell', 't0'], how='left')
    for a, b in [('tot_W', 'tot_W_m'), ('all_newapp_sh', 'all_newapp_sh_S0'), ('all_newapp_n', 'all_newapp_n_S0'), ('kr_newapp_sh', 'kr_newapp_sh_S0m'), ('kr_newapp_n', 'kr_newapp_n_S0m')]:
        mx[a] = max(mx.get(a, 0.0), float(np.nanmax(np.abs(g[a].values - g[b].values))))
    mx.setdefault('n_rows', 0); mx['n_rows'] += len(g)
chk['maxdiff_vs_official_feat'] = mx
F.to_parquet(os.path.join(CACHE, f'scen_feats_{which}.parquet'))
json.dump(chk, open(os.path.join(CACHE, f's2_check_{which}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
