"""s3: 데이터 확인 표 + 피처 분포 이동(PSI·KS) → cache/s3_*.csv, feature_shift.csv, cache/s3_results.json"""
import sys, os, json, numpy as np, pandas as pd, duckdb
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
R = {}
# 1) 국가×연도 문서 결측·법인형 비율
for nm in ['appl', 'pub']:
    d = pd.read_csv(os.path.join(CACHE, f'docrates_fin_{nm}.csv'))
    d = d[d.year.between(2000, 2024)]
    piv = d.pivot(index='year', columns='cc', values=['miss_rate', 'corp_rate']).round(4)
    piv.to_csv(os.path.join(CACHE, f's3_docrates_{nm}_pivot.csv'))
# 2) 셀 단위 n_app·n_newapp 분포 (국가×공개연도, 문서 있는 셀×연도 행)
P = pd.read_parquet(os.path.join(CACHE, 'panel_scen_fin.parquet'))
P = P[P.year.between(2000, 2024)]
g = P.groupby(['cc', 'year'])
tab = pd.DataFrame(dict(
    n_cellyears=g.size(), docs=g.n.sum(), miss_rate=g.n_null.sum() / g.n.sum(), corp_rate=g.n_corp.sum() / (g.n.sum() - g.n_null.sum()),
    n_app_med=g.n_app.median(), n_app_p90=g.n_app.quantile(0.9), n_app_per_doc=g.n_app.sum() / g.n.sum(),
    newapp_doc_sh=g.n_newapp_apps.sum() / g.n.sum(), newapp_doc_sh_nonmiss=g.n_newapp_apps.sum() / (g.n.sum() - g.n_null.sum()),
    n_newapp_med=g.n_newapp.median(), n_newapp_p90=g.n_newapp.quantile(0.9),
    newapp_doc_sh_S1=g.n_newapp_apps_s1.sum() / np.maximum(g.n.sum() - g.n_usold.sum(), 1),
    newapp_doc_sh_corponly=g.n_newapp_apps_corp.sum() / np.maximum(g.n_corp.sum(), 1),
)).reset_index()
tab.round(4).to_csv(os.path.join(CACHE, 's3_cell_cc_year.csv'), index=False)
R['us_by_pub'] = tab[tab.cc == 'US'][['year', 'docs', 'miss_rate', 'corp_rate', 'newapp_doc_sh_nonmiss', 'newapp_doc_sh_corponly', 'n_app_per_doc']].round(3).to_dict('records')
# 3) 결측 행의 산업 치우침(선택 편향): CN 출원 2005–09, JP 출원 2019–20 결측 대 비결측 KSIC 분포 TVD
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
B = ddb(os.path.join(CP_FIN, 'base_rows.parquet'))
for cc, a, b in [('CN', 2005, 2009), ('JP', 2019, 2020), ('US', 2000, 2004)]:
    d = con.sql(f"""select ksic, (app1 is null or app1='') miss, count(distinct appl_no) n from read_parquet('{B}')
                    where cc='{cc}' and appl_year between {a} and {b} group by 1,2""").df()
    pv = d.pivot(index='ksic', columns='miss', values='n').fillna(0)
    if True in pv and False in pv:
        s1 = pv[True] / pv[True].sum(); s0 = pv[False] / pv[False].sum()
        top = ((s1 - s0).abs().sort_values(ascending=False).head(5)).round(4).to_dict()
        R[f'tvd_ksic_{cc}_{a}_{b}'] = dict(tvd=float(0.5 * (s1 - s0).abs().sum()), miss_docs=int(pv[True].sum()), nonmiss_docs=int(pv[False].sum()), top_gap_ksic=top)
# 4) 피처 분포 이동: 학습 원점(2005–2018) 대 최종 원점(2021), 적격(≥50) 셀
SF = pd.read_parquet(os.path.join(CACHE, 'scen_feats_fin.parquet'))
fe = []
for t0 in list(range(2005, 2019)) + [2021]:
    f = pd.read_parquet(os.path.join(CP_FIN, f'feat_{t0}.parquet'), columns=['cell', 't0', 'tot_W', 'kr_W', 'all_newapp_sh', 'all_newpair_sh', 'kr_hhi', 'all_newapp_n', 'kr_newapp_sh', 'kr_newapp_n', 'kr_prio_for_sh'])
    fe.append(f[f.tot_W >= 50])
F = pd.concat(fe, ignore_index=True).merge(SF[['cell', 't0', 'all_newapp_sh_S1', 'all_newapp_n_S1', 'us_sh', 'usold_sh', 'US_newapp_sh', 'US_newapp_sh_corp', 'CN_newapp_sh', 'JP_newapp_sh', 'EP_newapp_sh', 'KR_newapp_sh', 'jp_null_sh', 'null_sh']], on=['cell', 't0'], how='left')
F.to_parquet(os.path.join(CACHE, 's3_elig_feats_fin.parquet'))
def psi(a, b, bins=10):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return np.nan
    q[0] = -np.inf; q[-1] = np.inf
    pa = np.histogram(a, q)[0] / len(a); pb = np.histogram(b, q)[0] / len(b)
    pa = np.clip(pa, 1e-4, None); pb = np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))
rows = []
feats = ['all_newapp_sh', 'all_newapp_sh_S1', 'all_newpair_sh', 'kr_hhi', 'all_newapp_n', 'all_newapp_n_S1', 'kr_newapp_sh', 'kr_newapp_n', 'kr_prio_for_sh', 'US_newapp_sh', 'US_newapp_sh_corp', 'CN_newapp_sh', 'JP_newapp_sh', 'EP_newapp_sh', 'KR_newapp_sh']
for ref_name, ref in [('train_2005_2018', F.t0.between(2005, 2018)), ('train_2015_2018', F.t0.between(2015, 2018)), ('dev_2010_2018', F.t0.between(2010, 2018))]:
    for f in feats:
        a = F.loc[ref, f].values.astype(float); b = F.loc[F.t0 == 2021, f].values.astype(float)
        ks = ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)])
        rows.append(dict(ref=ref_name, target='final_2021', feature=f, psi=psi(a, b), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                         ref_median=float(np.nanmedian(a)), final_median=float(np.nanmedian(b)), ref_mean=float(np.nanmean(a)), final_mean=float(np.nanmean(b)),
                         ref_nan=float(np.isnan(a).mean()), final_nan=float(np.isnan(b).mean()), n_ref=int(len(a)), n_final=int(len(b))))
# 원점별 추이(평균)
traj = F.groupby('t0')[feats + ['us_sh', 'usold_sh', 'jp_null_sh', 'null_sh']].mean().round(4)
traj.to_csv(os.path.join(CACHE, 's3_feature_traj_by_t0.csv'))
# 원점 쌍 PSI: 각 원점 대 2021
for t0 in list(range(2005, 2019)):
    for f in ['all_newapp_sh', 'all_newapp_sh_S1', 'all_newpair_sh', 'kr_hhi']:
        a = F.loc[F.t0 == t0, f].values.astype(float); b = F.loc[F.t0 == 2021, f].values.astype(float)
        ks = ks_2samp(a[~np.isnan(a)], b[~np.isnan(b)])
        rows.append(dict(ref=f'origin_{t0}', target='final_2021', feature=f, psi=psi(a, b), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                         ref_median=float(np.nanmedian(a)), final_median=float(np.nanmedian(b)), ref_mean=float(np.nanmean(a)), final_mean=float(np.nanmean(b)),
                         ref_nan=float(np.isnan(a).mean()), final_nan=float(np.isnan(b).mean()), n_ref=int(len(a)), n_final=int(len(b))))
FS = pd.DataFrame(rows)
FS.round(5).to_csv(os.path.join(WD, 'feature_shift.csv'), index=False, encoding='utf-8-sig')
R['shift_main'] = FS[FS.ref == 'train_2005_2018'][['feature', 'psi', 'ks_stat', 'ref_median', 'final_median']].round(4).to_dict('records')
R['traj'] = traj[['all_newapp_sh', 'all_newapp_sh_S1', 'US_newapp_sh', 'US_newapp_sh_corp', 'us_sh', 'usold_sh', 'jp_null_sh', 'all_newpair_sh', 'kr_hhi']].reset_index().to_dict('records')
json.dump(R, open(os.path.join(CACHE, 's3_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float)[:12000])
