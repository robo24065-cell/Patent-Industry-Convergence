# (b) NTIS 과제 수집본 자체 점검 — 원천 parquet 읽기 전용
import pandas as pd, numpy as np, sys, json, os
EXT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
cols = ['pjt_no','year','ministry','corp_rgst_no','perform_agent_cd','perform_agent','perform_grp','secret_flag',
        'dev_phase','region_cd','gov_funds','total_funds','research_agency','org_pno','title_ko','budget_project']
d = pd.read_parquet(os.path.join(EXT,'ntis_project_raw.parquet'), columns=cols)
print(d.shape, d.memory_usage(deep=True).sum()/1e9)
print(d.perform_grp.value_counts(dropna=False).to_string())
print(d.perform_agent.value_counts(dropna=False).head(20).to_string())
print(d.secret_flag.value_counts(dropna=False).to_string())
res = {}
res['dup_pjt_year'] = int(d.duplicated(['pjt_no','year']).sum())
res['dup_pjt_any'] = int(d.duplicated(['pjt_no']).sum())
res['n'] = len(d)
yr = d.groupby('year').agg(n=('pjt_no','size'), n_uniq=('pjt_no','nunique'), gov=('gov_funds','sum'),
    gov_na=('gov_funds', lambda s: s.isna().mean()), gov_zero=('gov_funds', lambda s: (s==0).mean()),
    gov_neg=('gov_funds', lambda s: (s<0).mean()), secret=('secret_flag','mean'),
    n_min=('ministry','nunique'), title_na=('title_ko', lambda s: s.isna().mean()))
firm = d.perform_grp.isin(['대기업','중견기업','중소기업'])
yr['firm_bizno_missing'] = d[firm].groupby('year').corp_rgst_no.apply(lambda s: (s.fillna('').str.replace(r'\D','',regex=True).str.len()!=10).mean())
yr['all_bizno_missing'] = d.groupby('year').corp_rgst_no.apply(lambda s: (s.fillna('').str.replace(r'\D','',regex=True).str.len()!=10).mean())
yr['region_na'] = d.groupby('year').region_cd.apply(lambda s: s.isna().mean() + (s=='').mean())
print(yr.to_string())
yr.to_csv(os.path.join(OUT,'_tmp_year.csv'), encoding='utf-8-sig')
# 단위 확인: 2015 정부연구비 합
print('gov 2015 sum', d.loc[d.year==2015,'gov_funds'].sum())
# 부처 x 연도
mt = d.pivot_table(index='ministry', columns='year', values='gov_funds', aggfunc='sum', fill_value=0)
mt.to_csv(os.path.join(OUT,'_tmp_ministry_year_gov.csv'), encoding='utf-8-sig')
mn = d.pivot_table(index='ministry', columns='year', values='pjt_no', aggfunc='size', fill_value=0)
mn.to_csv(os.path.join(OUT,'_tmp_ministry_year_n.csv'), encoding='utf-8-sig')
pg = d.pivot_table(index='perform_grp', columns='year', values='gov_funds', aggfunc='sum', fill_value=0)
pg.to_csv(os.path.join(OUT,'_tmp_pgrp_year_gov.csv'), encoding='utf-8-sig')
pa = d.pivot_table(index='perform_agent', columns='year', values='gov_funds', aggfunc='sum', fill_value=0)
pa.to_csv(os.path.join(OUT,'_tmp_pagent_year_gov.csv'), encoding='utf-8-sig')
pan = d.pivot_table(index='perform_agent', columns='year', values='pjt_no', aggfunc='size', fill_value=0)
pan.to_csv(os.path.join(OUT,'_tmp_pagent_year_n.csv'), encoding='utf-8-sig')
rg = d.pivot_table(index='region_cd', columns='year', values='gov_funds', aggfunc='sum', fill_value=0)
rg.to_csv(os.path.join(OUT,'_tmp_region_year_gov.csv'), encoding='utf-8-sig')
# 보안과제
s = d[d.secret_flag==True]
print('secret by grp', s.perform_grp.value_counts().to_string())
print('secret by ministry', s.ministry.value_counts().head(10).to_string())
print('secret title sample na', s.title_ko.isna().mean(), s.gov_funds.isna().mean(), s.corp_rgst_no.isna().mean())
json.dump(res, open(os.path.join(OUT,'_tmp_raw_res.json'),'w'), ensure_ascii=False)
print(res)
