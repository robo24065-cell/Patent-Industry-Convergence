import pandas as pd, numpy as np, json, os
from scipy.stats import spearmanr, mannwhitneyu
MY = os.path.dirname(os.path.abspath(__file__)); CA = os.path.join(os.path.dirname(MY), 'cache')
P = {s: pd.read_parquet(os.path.join(MY, f'v4_preds_{s}.parquet')).set_index('cell') for s in ['S0', 'S4a5', 'S4ak', 'S4']}
off0 = pd.read_parquet(os.path.join(CA, 'preds_S0_final.parquet')).set_index('cell'); off4 = pd.read_parquet(os.path.join(CA, 'preds_S4_final.parquet')).set_index('cell')
idx = off0.index; out = {}
out['S0_vs_official_maxdiff'] = float(np.abs(P['S0'].p.loc[idx].values - off0.p.values).max())
out['S4_vs_C09cache_maxdiff'] = float(np.abs(P['S4'].p.loc[idx].values - off4.p.values).max())
y = off0.yb == 1; n = len(idx); k = int(round(.1 * n))
rk = {s: P[s].p.loc[idx].rank(ascending=False, method='first') for s in P}; top = {s: rk[s] <= k for s in P}
for s in ['S4a5', 'S4ak', 'S4']:
    out[s] = dict(hits=int((top[s] & y).sum()), top_swapped=int((top['S0'] & ~top[s]).sum()), hits_lost=int((top['S0'] & ~top[s] & y).sum()),
                  hits_gained=int((~top['S0'] & top[s] & y).sum()), skill=json.load(open(os.path.join(MY, f'v4_res_{s}.json')))['skill'])
out['S0_skill'] = json.load(open(os.path.join(MY, 'v4_res_S0.json')))['skill']
for s in ['S4a5', 'S4ak', 'S4']: out[s]['dskill'] = out[s]['skill'] - out['S0_skill']
SF = pd.read_parquet(os.path.join(CA, 'scen_feats_fin.parquet')); SF = SF[SF.t0 == 2021].set_index('cell')
d = pd.DataFrame(dict(krW=off0.kr_W, us=SF.us_sh.loc[idx], y=y, top0=top['S0'], liftAF=rk['S4'] - rk['S0'], liftAF5=rk['S4a5'] - rk['S0'], liftAFK=rk['S4ak'] - rk['S0']))
h = d[d.top0 & d.y].sort_values('liftAF', ascending=False); t15 = h.head(15)
out['top15'] = t15.round(3).reset_index().values.tolist()
out['top15_AF5_gt_AFK'] = int((t15.liftAF5 > t15.liftAFK).sum())
out['hits_corr_liftAF5_us'] = float(spearmanr(h.liftAF5, h.us).statistic); out['hits_corr_liftAFK_us'] = float(spearmanr(h.liftAFK, h.us).statistic)
out['all_corr_liftAF5_us'] = float(spearmanr(d.liftAF5, d.us).statistic)
out['hits_abs_lift_mean'] = {c: float(h[c].abs().mean()) for c in ['liftAF', 'liftAF5', 'liftAFK']}
out['all_abs_lift_mean'] = {c: float(d[c].abs().mean()) for c in ['liftAF', 'liftAF5', 'liftAFK']}
r = h.iloc[15:]; out['C_us_median_top15_rest'] = [float(t15.us.median()), float(r.us.median()), float(mannwhitneyu(t15.us, r.us).pvalue)]
k13 = h[h.krW.between(1, 3)]; t = k13.head(15) if len(k13) > 15 else k13
kt = k13.sort_values('liftAF', ascending=False); a, bb = kt.head(10), kt.iloc[10:]
out['C_within_KR1_3_us_median_top10_rest_p'] = [float(a.us.median()), float(bb.us.median()), float(mannwhitneyu(a.us, bb.us).pvalue), int(len(kt))]
out['C_hits_us_by_krbin_median'] = {str(b): float(g.us.median()) for b, g in h.groupby(pd.cut(h.krW, [-1, 0, 3, 1e9]), observed=True)}
out['C_spearman_liftAF_us_within_KR1_3_hits'] = float(spearmanr(kt.liftAF, kt.us).statistic)
json.dump(out, open(os.path.join(MY, 'v4_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'top15'}, ensure_ascii=False, indent=1, default=float)); print(t15.round(3).to_string())
