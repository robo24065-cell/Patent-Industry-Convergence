# -*- coding: utf-8 -*-
"""보고서 '데이터 연계 결과'용 연계표·연계율 → R2/ext_link_*.csv, R2/ext_link_summary.json
기준 모집단: 개발 원점 마지막 기준일 L=2018-01-01 의 위험집합(살아 있는 KR 등록특허, 특허만).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import r2lib as R, r2ext as X, r2io

OUT = R.R2
L = '2018-01-01'
f = R.T.features(L)
nk = f.groupby('ksic').size().rename('n_patents_alive_2018')
names = pd.read_csv(R.SPEC0['paths']['ksic_names'], dtype={'세세분류코드': str}).drop_duplicates('세세분류코드').set_index('세세분류코드')['산업명']
summ = {}

# ---- 1) 무역: KSIC 5자리 ↔ HS6 ↔ UN Comtrade 한국 수입
mp = pd.read_csv(os.path.join(X.BASE, '분석', 'fusion', 'ksic_hs6_map.csv'), dtype=str)
imp = pd.read_csv(X.TRADE_CSV, dtype={'ksic': str})
g = mp.groupby('k5').agg(n_hs6=('hs6', 'nunique'), io_code=('io', 'first'))
hsset = mp.groupby('k5').hs6.apply(lambda s: tuple(sorted(set(s))))
share = {k: ','.join(sorted(k2 for k2, v2 in hsset.items() if v2 == v and k2 != k)) for k, v in hsset.items()}
yr = imp.groupby('ksic').agg(year_min=('year', 'min'), year_max=('year', 'max'), wld_imp_2016_usd=('wld_imp_usd', lambda s: np.nan))
yr['wld_imp_2016_usd'] = imp[imp.year == 2016].set_index('ksic').wld_imp_usd
yr['cn_share_2016'] = (imp[imp.year == 2016].set_index('ksic').cn_share_imp / 100).round(3)
t = pd.DataFrame(index=names.index.intersection(nk.index))
t['industry'] = names
t = t.join(nk).join(g).join(yr)
t['same_hs_set_as'] = pd.Series(share)
t['linked'] = t.n_hs6.notna() & t.year_min.notna()
t.index.name = 'ksic'
t.reset_index().to_csv(os.path.join(OUT, 'ext_link_trade.csv'), index=False, encoding='utf-8-sig')
summ['trade'] = dict(ksic_linked=int(t.linked.sum()), ksic_total=len(t), patent_link_rate=round(float(t.loc[t.linked, 'n_patents_alive_2018'].sum() / t.n_patents_alive_2018.sum()), 4),
                     hs6_total=int(mp.hs6.nunique()), ksic_sharing_hs_set=int((t.same_hs_set_as.fillna('') != '').sum()))

# ---- 2) KOSIS 광업제조업조사 5자리
k = pd.read_csv(X.KOSIS_CSV, dtype={'세세분류코드': str})
kk = k.groupby(['세세분류코드', '기준']).연도.agg(['min', 'max', 'nunique']).unstack()
kk.columns = [f'{b}_{a}' for a, b in kk.columns]
msk = k.groupby('세세분류코드').마스킹.sum().rename('n_masked_cells')
t2 = pd.DataFrame(index=t.index).join(names.rename('industry')).join(nk).join(kk).join(msk)
sh = k[(k['항목코드'] == 'T04') & (k['연도'] == 2016)].set_index('세세분류코드').값.rename('shipment_2016_mil_krw')
t2 = t2.join(sh)
t2['linked'] = t2['10차_nunique'].notna()
t2.reset_index().to_csv(os.path.join(OUT, 'ext_link_kosis.csv'), index=False, encoding='utf-8-sig')
summ['kosis_survey'] = dict(ksic_linked=int(t2.linked.sum()), ksic_total=len(t2),
                            patent_link_rate=round(float(t2.loc[t2.linked, 'n_patents_alive_2018'].sum() / t2.n_patents_alive_2018.sum()), 4),
                            years='10차 2008–2020(2010 없음), 11차 2020–2024; 피처는 L연도-2 까지만', usable_in_training='학습 기준일 L>=2011 부터(성장률 3년 창)')

# ---- 3) 산업연관표 수요구조
t3 = r2io.link_table().set_index('ksic')
t3 = t3.join(nk)
t3 = t3[t3.index.isin(t.index)]
t3['linked'] = t3.n_prodidx_codes > 0
t3.reset_index().to_csv(os.path.join(OUT, 'ext_link_io.csv'), index=False, encoding='utf-8-sig')
summ['io_table'] = dict(ksic_linked=int(t3.linked.sum()), ksic_total=int(len(t.index)), patent_link_rate=round(float(t3.loc[t3.linked, 'n_patents_alive_2018'].sum() / nk.sum()), 4),
                        n_io_sectors_50ksic=int(t3.io_sec.nunique()))

# ---- 4) 제공 5개국 메타 기술셀(내부 융합: 등록원부 IPC 메인그룹 ↔ 5개국 KSIC별 특허 리스트)
mg = X.pid_mg()[f.pid.values]
M = X._meta_tab('mg')['w']
hit = pd.Series(mg).isin(M.index).values
t4 = pd.DataFrame({'ksic': f.ksic.values, 'hit': hit}).groupby('ksic').hit.mean().rename('mg_match_rate').to_frame().join(nk)
t4.reset_index().to_csv(os.path.join(OUT, 'ext_link_meta_cells.csv'), index=False, encoding='utf-8-sig')
summ['meta_cells'] = dict(patent_match_rate=round(float(hit.mean()), 4), n_mg_cells=int(len(M)))
json.dump(summ, open(os.path.join(OUT, 'ext_link_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
