"""s6: 최종 원점 적중 셀 중 출원인 피처(AF)가 순위를 크게 올린 셀 — S0 대 S4(AF 제거), S0 대 S1(US ≤2011 제외) 비교 → cache/s6_*.csv, cache/s6_results.json
최종 원점 예측·레이블은 이미 채점된 공식 결과의 진단적 분해일 뿐 모형 선택에 쓰지 않는다."""
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
P = {s: pd.read_parquet(os.path.join(CACHE, f'preds_{s}_final.parquet')).set_index('cell') for s in ['S0', 'S1', 'S3', 'S4']}
SF = pd.read_parquet(os.path.join(CACHE, 'scen_feats_fin.parquet')); SF = SF[SF.t0 == 2021].set_index('cell')
F = pd.read_parquet(os.path.join(CP_FIN, 'feat_2021.parquet')).set_index('cell')
f2 = pd.read_parquet(os.path.join(CP_FIN, 'feat2.parquet')); f2 = f2[f2.t0 == 2021].assign(cell=lambda d: d.ksic + '|' + d.mg).set_index('cell')
d = P['S0'][['ksic', 'mg', 'kr_W', 'tot_W', 'y', 'yb', 'p']].rename(columns={'p': 'p_S0'})
for s in ['S1', 'S3', 'S4']: d[f'p_{s}'] = P[s].loc[d.index, 'p']
n = len(d); k = int(round(0.1 * n))
for s in ['S0', 'S1', 'S3', 'S4']:
    d[f'rk_{s}'] = d[f'p_{s}'].rank(ascending=False, method='first').astype(int); d[f'top_{s}'] = d[f'rk_{s}'] <= k
d['lift_AF'] = d.rk_S4 - d.rk_S0         # +: AF 가 있어서 순위가 올라감
d['lift_S1'] = d.rk_S0 - d.rk_S1         # +: US ≤2011 제외 시 순위가 올라감
d = d.join(SF[['us_sh', 'usold_sh', 'null_sh', 'jp_null_sh', 'US_newapp_sh', 'CN_newapp_sh', 'KR_newapp_sh']]).join(
    F[['all_newapp_sh', 'all_newapp_n', 'kr_napp', 'kr_hhi', 'kr_newapp_sh', 'kr_newapp_n', 'kr_newk_sh']]).join(f2[['app_mom', 'app_size', 'big_sh', 'top1_sh']])
R = dict(n=n, k=k)
for s in ['S1', 'S3', 'S4']:
    R[f'hits_S0'] = int((d.top_S0 & (d.yb == 1)).sum()); R[f'hits_{s}'] = int((d[f'top_{s}'] & (d.yb == 1)).sum())
    R[f'top_overlap_S0_{s}'] = int((d.top_S0 & d[f'top_{s}']).sum())
    R[f'hits_lost_{s}'] = int((d.top_S0 & ~d[f'top_{s}'] & (d.yb == 1)).sum()); R[f'hits_gained_{s}'] = int((~d.top_S0 & d[f'top_{s}'] & (d.yb == 1)).sum())
hits = d[d.top_S0 & (d.yb == 1)].copy()
R['hits_lift_AF_desc'] = hits.lift_AF.describe().round(1).to_dict()
R['hits_drop_out_without_AF'] = int((hits.rk_S4 > k).sum())
R['spearman_liftAF_ussh_all'] = float(spearmanr(d.lift_AF, d.us_sh).statistic)
R['spearman_liftAF_ussh_hits'] = float(spearmanr(hits.lift_AF, hits.us_sh).statistic)
R['spearman_liftS1_ussh_all'] = float(spearmanr(d.lift_S1, d.us_sh).statistic)
# 큰 상승 셀(적중 중 AF 로 순위가 k/2 이상 오른 셀)의 피처 프로필 대 나머지 적중
big = hits[hits.lift_AF >= k / 2]; rest = hits[hits.lift_AF < k / 2]
cols = ['kr_W', 'tot_W', 'us_sh', 'all_newapp_sh', 'all_newapp_n', 'kr_napp', 'kr_hhi', 'kr_newapp_sh', 'top1_sh', 'app_size', 'US_newapp_sh']
R['big_lift_hits_n'] = int(len(big)); R['profile_big_vs_rest_median'] = {c: [float(big[c].median()), float(rest[c].median())] for c in cols}
top = hits.sort_values('lift_AF', ascending=False).head(15)
top[['ksic', 'mg', 'kr_W', 'tot_W', 'rk_S0', 'rk_S4', 'lift_AF', 'rk_S1', 'us_sh', 'all_newapp_sh', 'all_newapp_n', 'kr_napp', 'kr_hhi', 'top1_sh', 'app_size']].round(3).to_csv(os.path.join(CACHE, 's6_top_lift_hits.csv'), encoding='utf-8-sig')
# AF 가 끌어올렸지만 부상 아닌 셀(오경보) 상위
fa = d[d.top_S0 & (d.yb == 0)].sort_values('lift_AF', ascending=False).head(15)
fa[['ksic', 'mg', 'kr_W', 'tot_W', 'rk_S0', 'rk_S4', 'lift_AF', 'us_sh', 'all_newapp_sh', 'kr_napp', 'kr_hhi', 'top1_sh']].round(3).to_csv(os.path.join(CACHE, 's6_top_lift_falsealarm.csv'), encoding='utf-8-sig')
# KR 이력 구간별 AF 효과(상위 10% 적중 수)
d['kr_bin'] = pd.cut(d.kr_W, [-1, 0, 4, 1e9], labels=['KR0', 'KR1-4', 'KR5+'])
R['hits_by_krbin'] = {str(b): dict(S0=int((g.top_S0 & (g.yb == 1)).sum()), S4=int((g.top_S4 & (g.yb == 1)).sum()), S1=int((g.top_S1 & (g.yb == 1)).sum())) for b, g in d.groupby('kr_bin', observed=True)}
d.to_parquet(os.path.join(CACHE, 's6_cells_final.parquet'))
json.dump(R, open(os.path.join(CACHE, 's6_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float)); print(top[['ksic', 'mg', 'kr_W', 'rk_S0', 'rk_S4', 'lift_AF', 'us_sh', 'kr_napp', 'kr_hhi', 'top1_sh']].round(3).to_string())
