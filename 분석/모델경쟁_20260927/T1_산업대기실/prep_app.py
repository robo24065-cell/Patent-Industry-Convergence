# 공개 출원(범위안, 미등록 포함) — kr_patents.parquet 출원인명을 법인 별칭으로 코드화. 기준시점에 공개돼 있던 '시도'(등록 안 된 것 포함) 정보.
import pandas as pd, numpy as np, re, time
from common import *
t0 = time.time()
k = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원일', '출원연도', '등록번호', '등록일자', '공개일자', '출원인명', '세세분류코드'])
k.columns = ['appl_no', 'appl_date', 'ay', 'rgst_no', 'rgst_date', 'pub_date', 'names', 'ksic']
k['ksic'] = k.ksic.astype(str); k['reg'] = k.rgst_no.fillna('').astype(str).str.len() > 3
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
firms = pd.read_parquet(CACHE/'firms.parquet'); firms = firms[firms.status == 'OK']
app = pd.read_parquet(CACHE/'ins_applicant.parquet')
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]
n2c = dict(zip(al.n, al.applicant_cd))
k['nm'] = k.names.fillna('').str.split('|'); k = k.explode('nm'); k['applicant_cd'] = k.nm.map(norm).map(n2c)
m = k.dropna(subset=['applicant_cd']).drop_duplicates(['appl_no', 'applicant_cd'])
m = m[['applicant_cd', 'appl_no', 'ay', 'appl_date', 'pub_date', 'rgst_date', 'reg', 'ksic']]
m.to_parquet(CACHE/'app_pub.parquet', index=False)
print('apps', k.appl_no.nunique(), 'matched apps', m.appl_no.nunique(), 'firms', m.applicant_cd.nunique(), 'unregistered matched', int((~m.reg).sum()), round(time.time() - t0))
