"""verify_1 반박 시도: '발명자 이름은 셀 간 상대 순위에 균일하게 작용해 영향이 상쇄된다'.
원점별(적격 셀) US 문서 비중과 피처 왜곡량(all_newapp_sh S0−S1, all_newapp_n S0−S1, 원점 내 백분위 순위 차)의 스피어만 상관.
PREREG §7: 원점 내 |ρ|≥0.3 이 학습 원점(2005–2018) 절반 이상(≥7/14)에서 나오면 '균일 작용 가설 기각'.
보조: 왜곡량이 부상 레이블과 관련되는지(왜곡량의 부상 판별 AUC), S0·S1 원점 내 순위상관."""
import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
MY = os.path.dirname(os.path.abspath(__file__)); WD = os.path.dirname(MY)
D = pd.read_parquet(os.path.join(WD, 'verify_0', 'v0_D_final.parquet'), columns=['cell', 't0', 'y', 'yb', 'tot_W', 'kr_W', 'all_newapp_sh', 'all_newapp_n'])
SF = pd.read_parquet(os.path.join(WD, 'cache', 'scen_feats_fin.parquet'), columns=['cell', 't0', 'all_newapp_sh_S1', 'all_newapp_n_S1', 'us_sh', 'usold_sh', 'US_newapp_sh'])
X = D.merge(SF, on=['cell', 't0'], how='left')
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rows = []
for t0, g in X.groupby('t0'):
    dsh = g.all_newapp_sh - g.all_newapp_sh_S1; dn = g.all_newapp_n - g.all_newapp_n_S1
    drk = g.all_newapp_sh.rank(pct=True) - g.all_newapp_sh_S1.rank(pct=True)
    rows.append(dict(t0=int(t0), n=len(g), mean_us_sh=g.us_sh.mean(), mean_usold_sh=g.usold_sh.mean(),
                     mean_abs_dsh=dsh.abs().mean(), rho_ussh_dsh=spearmanr(g.us_sh, dsh).statistic, rho_ussh_dn=spearmanr(g.us_sh, dn).statistic,
                     rho_ussh_drank=spearmanr(g.us_sh, drk).statistic, rho_ussh_absdrank=spearmanr(g.us_sh, drk.abs()).statistic,
                     mean_abs_drank=drk.abs().mean(), rank_corr_S0_S1=spearmanr(g.all_newapp_sh, g.all_newapp_sh_S1).statistic,
                     auc_dsh_for_yb=auc(g.yb.values, dsh.values) if g.yb.sum() > 0 else np.nan,
                     auc_S0sh_for_yb=auc(g.yb.values, g.all_newapp_sh.values), auc_S1sh_for_yb=auc(g.yb.values, g.all_newapp_sh_S1.values),
                     rho_ussh_y=spearmanr(g.us_sh, g.y).statistic))
R = pd.DataFrame(rows)
R.round(4).to_csv(os.path.join(MY, 'v1_by_origin.csv'), index=False, encoding='utf-8-sig')
tr = R[R.t0 <= 2018]
n_strong = int((tr[['rho_ussh_dsh', 'rho_ussh_drank']].abs().max(axis=1) >= 0.3).sum())
n_strong_dsh = int((tr.rho_ussh_dsh.abs() >= 0.3).sum())
out = dict(n_train_origins=int(len(tr)), n_origins_absrho_ge_0_3_either=n_strong, n_origins_absrho_ge_0_3_dsh=n_strong_dsh,
           verdict=('균일 작용 가설 기각(왜곡이 셀의 US 비중에 따라 체계적)' if n_strong_dsh >= 7 else '기각 못함'),
           by_origin=R.round(3).to_dict('records'))
json.dump(out, open(os.path.join(MY, 'v1_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250); print(R.round(3).to_string()); print({k: v for k, v in out.items() if k != 'by_origin'})
