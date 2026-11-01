# 해외 선행(EP) 매칭: EP 출원인명(영문) ↔ 법인 영문명(registry applicantEngName) 정규화 일치
import duckdb, re, pandas as pd, numpy as np, time
from common import *
t0 = time.time()
M = (B/'분석/모델경쟁_20260927/_shared/meta').as_posix() + '/*.parquet'
ep = duckdb.sql(f"select appl_no, applicant, appl_year, pub_date, ksic, prio_cc from read_parquet('{M}') where cc='EP' and appl_year<=2012").df()
print('EP rows', len(ep), time.time() - t0, flush=True)
SUF = re.compile(r"\b(CO|COMPANY|CORP|CORPORATION|INC|INCORPORATED|LTD|LIMITED|LLC|PLC|GMBH|AG|SA|SAS|SPA|BV|NV|SE|AB|OY|KK|KABUSHIKI KAISHA|KABUSHIKIKAISHA|CO LTD|THE|& CO|KG|S A|S P A|B V|N V)\b")
def norm(s):
    s = str(s).upper(); s = re.sub(r"[\.\,\(\)\-&'/]", ' ', s); s = SUF.sub(' ', s); return re.sub(r'\s+', '', s)
ep = ep.dropna(subset=['applicant'])
ep['nm'] = ep.applicant.str.split('|'); ep = ep.explode('nm'); ep['n'] = ep.nm.map(norm)
app = pd.read_sql('select applicantCd, applicantEngName from applicant', RG())
firms = set(pd.read_parquet(CACHE/'firms.parquet').query("status=='OK'").applicant_cd)
app = app[app.applicantCd.isin(firms)].dropna(subset=['applicantEngName'])
app['n'] = app.applicantEngName.map(norm); app = app[app.n.str.len() >= 4].drop_duplicates(['applicantCd', 'n'])
# 한 정규화 이름이 여러 코드에 대응하면 모호 → 제외
amb = app.groupby('n').applicantCd.nunique(); app = app[app.n.map(amb) == 1]
m = ep.merge(app[['n', 'applicantCd']], on='n').drop_duplicates(['appl_no', 'ksic', 'applicantCd'])
m = m.rename(columns={'applicantCd': 'applicant_cd'})[['applicant_cd', 'appl_no', 'appl_year', 'pub_date', 'ksic', 'prio_cc']]
m.to_parquet(CACHE/'ep_firm.parquet', index=False)
print('matched EP apps', m.appl_no.nunique(), 'firms', m.applicant_cd.nunique(), 'share of EP rows matched', round(m.appl_no.nunique() / ep.appl_no.nunique(), 3))
print(m.prio_cc.fillna('none').str[:2].value_counts().head(8).to_dict()); print('time', time.time() - t0)
