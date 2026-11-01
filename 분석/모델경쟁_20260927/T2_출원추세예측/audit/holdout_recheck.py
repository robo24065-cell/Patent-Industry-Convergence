# 3단계 독립 재계산: final_run.py 코드를 import 하지 않고 원천 meta에서 직접 집계해 홀드아웃 지표를 대조한다.
# - 예측값은 holdout_forecast.csv(동결 예측)만 읽고, 1단계 동결 예측(outputs/frozen_forecast_origin2018.csv)과 일치하는지 확인
# - 실제값·naive·MASE 척도·skill·WAPE·KSIC 군집 부트스트랩 CI 를 별도 구현
# - 진단(모형 변경 아님): 국가·h별 편향, JP 2021 제외, CN 제외, 오차 기여 상위 산업
import duckdb, os, json
import numpy as np, pandas as pd
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(T, '..', '_shared', 'meta', '*.parquet').replace(os.sep, '/')
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
cnt = con.sql(f"""select ksic, cc, appl_year::int y, count(distinct appl_no) n from read_parquet('{P}')
                  where appl_year between 2000 and 2021 group by 1,2,3""").df()
con.close()
W = cnt.pivot_table(index=['ksic', 'cc'], columns='y', values='n', aggfunc='sum').reindex(columns=range(2000, 2022)).fillna(0.0)
fc = pd.read_csv(os.path.join(T, 'holdout_forecast.csv'), dtype={'ksic': str})
fz = pd.read_csv(os.path.join(T, 'outputs', 'frozen_forecast_origin2018.csv'), dtype={'ksic': str})
m = fc.merge(fz[['ksic', 'cc', 'target', 'f']], on=['ksic', 'cc', 'target'], suffixes=('', '_stage1'))
out = {'n_series': int(W.shape[0]), 'n_fc_rows': len(fc),
       'max_absdiff_vs_stage1_frozen': float((m.f - m.f_stage1).abs().max()), 'n_matched_stage1': len(m)}
W.index = W.index.set_levels([W.index.levels[0].astype(str), W.index.levels[1]])
hist = W.loc[:, 2000:2018].values
scale = np.mean(np.abs(np.diff(hist, axis=1)), axis=1); scale = np.where(scale <= 0, 1.0, scale)
S = pd.DataFrame({'scale': scale, 'y2018': hist[:, -1]}, index=W.index)
r = fc.join(S, on=['ksic', 'cc'])
r['y'] = [W.loc[(k, c), t] for k, c, t in zip(r.ksic, r.cc, r.target)]
r['h'] = r.target - 2018
out['y_origin_matches'] = bool(np.allclose(r.y_origin, r.y2018))
out['naive_matches'] = bool(np.allclose(r.f_naive, r.y2018))
for nm in ['f', 'f_naive', 'f_drift', 'f_damped_ets', 'f_ridge_lead', 'f_flow_um', 'f_topdown']:
    r['ase_' + nm] = (r.y - r[nm]).abs() / r.scale
mn = r.ase_f_naive.mean()
out['MASE'] = r.ase_f.mean(); out['MASE_naive'] = mn; out['skill'] = 1 - r.ase_f.mean() / mn
out['WAPE'] = (r.y - r.f).abs().sum() / r.y.sum(); out['WAPE_naive'] = (r.y - r.f_naive).abs().sum() / r.y.sum()
for nm in ['f_drift', 'f_damped_ets', 'f_ridge_lead', 'f_flow_um', 'f_topdown']:
    out['skill_' + nm] = 1 - r['ase_' + nm].mean() / mn
# 독립 부트스트랩 (KSIC 50 군집, seed 12345, 5000회)
rng = np.random.default_rng(12345)
g = r.groupby('ksic')[['ase_f', 'ase_f_naive']].agg(['sum', 'count'])
ks = g.index.values; A = g[('ase_f', 'sum')].values; Bn = g[('ase_f_naive', 'sum')].values; C = g[('ase_f', 'count')].values
sk = []
for _ in range(5000):
    ii = rng.integers(0, len(ks), len(ks))
    sk.append(1 - (A[ii].sum() / C[ii].sum()) / (Bn[ii].sum() / C[ii].sum()))
out['skill_CI95_indep'] = np.percentile(sk, [2.5, 97.5]).tolist()
out['P_skill_gt0'] = float(np.mean(np.array(sk) > 0))
# 진단
def sk_of(mask):
    return float(1 - r[mask].ase_f.mean() / r[mask].ase_f_naive.mean())
out['skill_excl_JP2021'] = sk_of(~((r.cc == 'JP') & (r.target == 2021)))
out['skill_excl_CN'] = sk_of(r.cc != 'CN')
out['skill_h12'] = sk_of(r.h <= 2)
out['bias_ratio_sumF_sumY'] = {f'{c}_h{h}': float(d.f.sum() / d.y.sum()) for (c, h), d in r.groupby(['cc', 'h'])}
out['bias_ratio_naive'] = {f'{c}_h{h}': float(d.f_naive.sum() / d.y.sum()) for (c, h), d in r.groupby(['cc', 'h'])}
out['skill_by_cc_h'] = {f'{c}_h{h}': float(1 - d.ase_f.mean() / d.ase_f_naive.mean()) for (c, h), d in r.groupby(['cc', 'h'])}
# 방향 적중: 원점 대비 증가/감소 방향 (h=3)
d3 = r[r.h == 3]
out['direction_hit_h3'] = float(np.mean(np.sign(d3.f - d3.y_origin) == np.sign(d3.y - d3.y_origin)))
# 산업별 오차 기여 상위
kk = r.groupby('ksic').apply(lambda d: pd.Series({'ase_model': d.ase_f.sum(), 'ase_naive': d.ase_f_naive.sum()}))
kk['diff'] = kk.ase_model - kk.ase_naive
out['worst_ksic_vs_naive'] = kk.sort_values('diff', ascending=False).head(6).round(2).reset_index().to_dict('records')
out['best_ksic_vs_naive'] = kk.sort_values('diff').head(6).round(2).reset_index().to_dict('records')
# 비전문가용: 계열별 h별 절대백분율오차 중앙값, 10% 이내 비율
ape = (r.f - r.y).abs() / r.y.clip(lower=1); apen = (r.f_naive - r.y).abs() / r.y.clip(lower=1)
out['median_APE'] = {'model': float(ape.median()), 'naive': float(apen.median())}
out['median_APE_by_h'] = {int(h): [float(ape[r.h == h].median()), float(apen[r.h == h].median())] for h in (1, 2, 3)}
out['share_within10pct'] = {'model': float((ape <= 0.10).mean()), 'naive': float((apen <= 0.10).mean())}
out['model_better_share'] = float((r.ase_f < r.ase_f_naive).mean())
json.dump(out, open(os.path.join(T, 'audit', 'holdout_recheck.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
