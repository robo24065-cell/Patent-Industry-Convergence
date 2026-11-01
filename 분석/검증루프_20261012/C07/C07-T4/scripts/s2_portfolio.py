# 포트폴리오 인스턴스 표 PI(name, appl_no, date, path) — KSIC는 나중에 문서 배정표로 붙인다(P5 재구성용).
# A: 제공 kr_patents(출원 당시 명칭) / B: registry applicant 출원인코드 경로(2026 명칭끼리 이름→코드) / C: registry owner 보유 행(행 날짜)
import sys, time, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
t = time.time(); R = {}
k = pd.read_parquet(KRP, columns=['출원번호', '출원일', '공개일자', '등록일자', '등록번호', '출원인명', '세세분류코드', '메인IPC'])
k.columns = ['appl_no', 'adate', 'pub', 'rg', 'rgno', 'aname', 'ksic', 'mipc']
for c in ['pub', 'rg']:
    k[c] = k[c].fillna('').astype(str).str.strip(); k.loc[k[c].str.len() != 8, c] = None
k['vis'] = smin(k, ['pub', 'rg'])
k['adate'] = k.adate.astype(str).str.replace('-', '').str[:8]
k['ksic'] = k.ksic.astype(str)
k['nn'] = k.aname.map(norm)
R['kr_docs'] = len(k); R['kr_vis_null'] = int(k.vis.isna().sum())
docs = k[['appl_no', 'ksic', 'vis', 'adate', 'mipc', 'nn', 'rgno', 'rg']].copy()
docs.to_parquet(C + r'\docs.parquet', index=False)
pat = pd.read_parquet(C + r'\pat.parquet', columns=['rgst_no', 'applNo', 'ksic', 'opnDate', 'pubDate', 'rgstDate'])
pat = pat.rename(columns={'applNo': 'appl_no', 'ksic': 'ksic_r'})
m = pat.merge(docs[['appl_no', 'ksic', 'vis']], on='appl_no', how='left')
R['reg_in_kr'] = float(m.ksic.notna().mean()); R['reg_ksic_eq_kr'] = float((m.ksic == m.ksic_r.astype(str)).mean())
m['vis_r'] = smin(m, ['opnDate', 'pubDate', 'rgstDate'])
m['vis'] = m.vis.fillna(m.vis_r)
r2a = m[['rgst_no', 'appl_no', 'vis']]
# A
A = docs.loc[docs.vis.notna() & (docs.nn.str.len() >= 1), ['nn', 'appl_no', 'vis']].rename(columns={'nn': 'name', 'vis': 'date'}); A['path'] = 'A'
# B
app = pd.read_parquet(C + r'\app.parquet'); app['nn'] = app.applicantName.map(norm)
app = app.merge(r2a, on='rgst_no', how='left')
nc = app.dropna(subset=['applicantCd'])[['nn', 'applicantCd']].drop_duplicates()
nc = nc[~nc.applicantCd.str[:1].isin(['4', '6'])]          # 개인 코드 제외(동명이인)
cd = app.dropna(subset=['applicantCd', 'vis'])[['applicantCd', 'appl_no', 'vis']].drop_duplicates()
Bt = nc.merge(cd, on='applicantCd')[['nn', 'appl_no', 'vis']].rename(columns={'nn': 'name', 'vis': 'date'}); Bt['path'] = 'B'
R['names_per_code_gt1_share'] = float((nc.groupby('applicantCd').nn.nunique() > 1).mean())
R['codes_per_name_gt1_share'] = float((nc.groupby('nn').applicantCd.nunique() > 1).mean())
# C
own = pd.read_parquet(C + r'\own.parquet', columns=['rgst_no', 'ownerName', 'd'])
own['name'] = own.ownerName.map(norm)
Ct = own.merge(r2a[['rgst_no', 'appl_no']], on='rgst_no')[['name', 'appl_no', 'd']].rename(columns={'d': 'date'}); Ct['path'] = 'C'
PI = pd.concat([A, Bt, Ct], ignore_index=True)
PI = PI[PI.name.str.len() >= 1]
R['PI_rows_raw'] = {p: int((PI.path == p).sum()) for p in 'ABC'}
# 경로별 최소 날짜(민감도용) — 이름×문서×경로
PI = PI.groupby(['name', 'appl_no', 'path'], as_index=False).date.min()
PI.to_parquet(C + r'\PI.parquet', index=False)
R['PI_rows'] = len(PI); R['PI_names'] = int(PI.name.nunique()); R['sec'] = round(time.time() - t, 1)
json.dump(R, open(C + r'\s2_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(R)
