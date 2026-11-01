"""verify_2: C09-T3 발견 'all_newapp_sh 학습→최종 PSI 0.148, 2005–11 원점 집중, 주원인 CN 결측, US 발명자 이름이 일부 상쇄' 반박 시도.
입력(읽기 전용): ..\cache\panel_scen_fin.parquet (s1_panel 산출, 공식 panel과 최대차 0 확인됨)
독립 경로: s2_feats/s3b 코드를 쓰지 않고 pandas groupby 로 창 합(t0-2..t0)을 다시 만든다.
출력: verify_2\v2_psi_by_origin.csv, v2_results.json
"""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_parquet(os.path.join(HERE, '..', 'cache', 'panel_scen_fin.parquet'),
                    columns=['ksic', 'mg', 'cc', 'year', 'n', 'n_null', 'n_usold', 'n_cnmiss59', 'n_newapp_apps', 'n_newapp_apps_s1'])
P['cell'] = P.ksic + '|' + P.mg
P = P[P.year <= 2021]
num = ['n', 'n_null', 'n_usold', 'n_cnmiss59', 'n_newapp_apps', 'n_newapp_apps_s1']
# 국가별 열로 펼침
W = P.pivot_table(index=['cell', 'year'], columns='cc', values=num, aggfunc='sum', fill_value=0)
W.columns = [f'{a}_{b}' for a, b in W.columns]
W = W.reset_index()
for c in [x for x in W.columns if x not in ('cell', 'year')]:
    W[c] = W[c].astype('float64')
CCS = ['KR', 'US', 'CN', 'JP', 'EP']
for a in num:
    for cc in CCS:
        if f'{a}_{cc}' not in W: W[f'{a}_{cc}'] = 0.0
T0S = list(range(2005, 2019)) + [2021]
rows = []
for t0 in T0S:
    g = W[W.year.between(t0 - 2, t0)].drop(columns='year').groupby('cell').sum()
    g['t0'] = t0
    rows.append(g)
F = pd.concat(rows).reset_index()
tot = sum(F[f'n_{c}'] for c in CCS)
F = F[tot >= 50].copy(); tot = tot[tot >= 50]
num0 = sum(F[f'n_newapp_apps_{c}'] for c in CCS)
num1 = sum(F[f'n_newapp_apps_s1_{c}'] for c in CCS)
usold = sum(F[f'n_usold_{c}'] for c in CCS)
nnull = {c: F[f'n_null_{c}'] for c in CCS}
cn59 = sum(F[f'n_cnmiss59_{c}'] for c in CCS)
nullall = sum(nnull.values())
d = lambda x: np.maximum(x, 1)
S = {}
S['S0'] = num0 / d(tot)
S['S1'] = num1 / d(tot - usold)
S['S2'] = num0 / d(tot - cn59)
S['S2b'] = num0 / d(tot - nullall)
# 결측 국가별 분해
S['S2b_CNonly'] = num0 / d(tot - nnull['CN'])
S['S2b_USonly'] = num0 / d(tot - nnull['US'])
S['S2b_JPonly'] = num0 / d(tot - nnull['JP'])
S['S2b_noCN'] = num0 / d(tot - nullall + nnull['CN'])  # CN 결측만 남기고 나머지 결측 제외
# US 상쇄 검증: S1 을 결측 보정(S2b) 위에 얹기 (US 결측은 사실상 출원 ≤2011 → usold 에 포함되므로 비US 결측만 추가로 뺌)
S['S1_on_S2b'] = num1 / d(tot - usold - (nullall - nnull['US']))
# US 출원 ≤2011 문서를 분모에 두되 '신규' 비율을 법인 시기 수준 r 로 바꾼 반사실 (분자: S1 분자 + r*usold_nonnull)
us_old_nonnull = usold - nnull['US']
for r in (0.40, 0.55, 0.72, 0.85):
    S[f'S1r{int(r*100)}'] = (num1 + r * us_old_nonnull) / d(tot)
F = F.assign(**{k: v.values for k, v in S.items()})
F['us_old_sh'] = (usold / d(tot)).values
F['cn_null_sh'] = (nnull['CN'] / d(tot)).values
F['us_null_sh'] = (nnull['US'] / d(tot)).values
F['null_sh'] = (nullall / d(tot)).values

def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); q[0] = -np.inf; q[-1] = np.inf
    pa = np.clip(np.histogram(a, q)[0] / len(a), 1e-4, None); pb = np.clip(np.histogram(b, q)[0] / len(b), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))
def psi_rev(a, b, bins=10):  # 최종 분위 기준(기준 분포 뒤집기) 민감도
    return psi(b, a, bins)

fin = F[F.t0 == 2021]
keys = list(S.keys())
out = []
groups = [(str(t), F[F.t0 == t]) for t in range(2005, 2019)] + [
    ('train_2005_2018', F[F.t0.between(2005, 2018)]), ('train_2012_2018', F[F.t0.between(2012, 2018)]),
    ('train_2005_2011', F[F.t0.between(2005, 2011)])]
for name, g in groups:
    r = dict(t0=name, n=len(g))
    for k in keys:
        r[f'psi_{k}'] = psi(g[k].values, fin[k].values)
        r[f'mean_{k}'] = float(g[k].mean())
    r['psi_S0_rev'] = psi_rev(g['S0'].values, fin['S0'].values)
    r['psi_S0_bins20'] = psi(g['S0'].values, fin['S0'].values, 20)
    for c in ['us_old_sh', 'cn_null_sh', 'us_null_sh', 'null_sh']:
        r[f'mean_{c}'] = float(g[c].mean())
    out.append(r)
r = dict(t0='final_2021', n=len(fin)); r.update({f'mean_{k}': float(fin[k].mean()) for k in keys})
for c in ['us_old_sh', 'cn_null_sh', 'us_null_sh', 'null_sh']: r[f'mean_{c}'] = float(fin[c].mean())
out.append(r)
R = pd.DataFrame(out)
R.round(4).to_csv(os.path.join(HERE, 'v2_psi_by_origin.csv'), index=False, encoding='utf-8-sig')
# 원래 s3b 값과 대조
orig = pd.read_csv(os.path.join(HERE, '..', 'cache', 's3b_psi_by_origin_scen.csv'))
cmp = {}
for k in ['S0', 'S1', 'S2', 'S2b']:
    a = orig.set_index('t0')[f'psi_{k}'].dropna(); b = R.set_index('t0')[f'psi_{k}']
    cmp[k] = float(max(abs(a[i] - b[str(i)]) for i in a.index))
# 학습 PSI 기여(원점별 분해: 학습 풀에서 원점 하나씩 뺐을 때)
tr = F[F.t0.between(2005, 2018)]
loo = {}
for t in range(2005, 2019):
    loo[t] = psi(tr[tr.t0 != t]['S0'].values, fin['S0'].values)
res = dict(maxdiff_vs_s3b=cmp, n_train=int(len(tr)), n_final=int(len(fin)), loo_psi_S0=loo)
json.dump(res, open(os.path.join(HERE, 'v2_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.set_option('display.width', 250)
show = ['t0', 'psi_S0', 'psi_S1', 'psi_S2', 'psi_S2b', 'psi_S2b_CNonly', 'psi_S2b_USonly', 'psi_S2b_noCN', 'psi_S1_on_S2b', 'psi_S1r40', 'psi_S1r55', 'psi_S1r72', 'psi_S1r85']
print(R[show].round(3).to_string())
print(R[['t0', 'mean_S0', 'mean_S1', 'mean_S2b', 'mean_us_old_sh', 'mean_cn_null_sh', 'mean_us_null_sh', 'mean_null_sh', 'psi_S0_rev', 'psi_S0_bins20']].round(3).to_string())
print(json.dumps(res, ensure_ascii=False, indent=1))
