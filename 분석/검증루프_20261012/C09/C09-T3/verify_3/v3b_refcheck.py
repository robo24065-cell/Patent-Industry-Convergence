"""verify_3 보조(판정 외, 민감도): 판정 3의 기준 국가를 EP 대신 KR·JP·비US 4개국 합으로 바꿔 excess와 평균 dsh_ss를 다시 계산."""
import os, json, numpy as np, pandas as pd
MY = os.path.dirname(os.path.abspath(__file__)); WD = os.path.dirname(MY)
ROOT = os.path.abspath(os.path.join(WD, '..', '..', '..'))
SRC = os.path.join(ROOT, '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3', 'cache_pub_final')
PC = pd.read_parquet(os.path.join(SRC, 'panel_ccy.parquet'), columns=['ksic', 'mg', 'cc', 'year', 'n', 'n_newapp_apps', 'n_app'])
PC['cell'] = PC.ksic + '|' + PC.mg
D = pd.read_parquet(os.path.join(WD, 'verify_0', 'v0_D_final.parquet'), columns=['cell', 't0'])
R = pd.read_csv(os.path.join(MY, 'v3_by_origin.csv'))
rows = []
for t0, g in D.groupby('t0'):
    w = PC[(PC.year >= t0 - 2) & (PC.year <= t0) & PC.cell.isin(set(g.cell))]
    d = dict(t0=int(t0))
    for c in ['US', 'EP', 'KR', 'JP', 'CN']:
        s = w[w.cc == c]; d['r_' + c] = s.n_newapp_apps.sum() / max(s.n.sum(), 1)
    s = w[(w.cc != 'US') & (w.cc != 'CN')]; d['r_EPKRJP'] = s.n_newapp_apps.sum() / max(s.n.sum(), 1)
    rows.append(d)
X = pd.DataFrame(rows).merge(R[['t0', 'mean_us_sh']], on='t0')
b = X[X.t0 == 2021].iloc[0]
for ref in ['EP', 'KR', 'JP', 'EPKRJP']:
    X['excess_' + ref] = (X.r_US - b.r_US) - (X['r_' + ref] - b['r_' + ref])
    X['mean_dsh_ss_' + ref] = X['excess_' + ref] * X.mean_us_sh
X['excess_noref'] = X.r_US - b.r_US
X.round(4).to_csv(os.path.join(MY, 'v3b_refcheck.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
print(X.round(3).to_string())
