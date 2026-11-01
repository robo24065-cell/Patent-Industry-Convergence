# x1: data_check용 NTIS corp_rgst_no 보유율·연도 분포(차단 (ii) 근거 자료 점검). 개발·잠금 차단 기간(2010–2021) 위주.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
nt = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'corp_rgst_no', 'perform_agent', 'research_agency'])
nt['year'] = pd.to_numeric(nt.year, errors='coerce')
nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True)
nt['has10'] = nt.bz.str.len() == 10
nt['tc'] = nt.has10 & nt.bz.str[3:5].isin(CORP_TC)
t = nt.groupby('year').agg(n=('has10', 'size'), has10=('has10', 'mean'), corp_tc=('tc', 'mean'), uniq_corp=('bz', lambda s: s[s.str.len() == 10].nunique()))
print(t.round(4).to_string())
print('perform_agent top', nt.perform_agent.value_counts().head(8).to_dict())
print('dup pjt rows by year?', nt.duplicated(['year', 'corp_rgst_no', 'research_agency']).mean().round(4))
