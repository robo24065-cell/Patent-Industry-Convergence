# s1: C-D 기초표(읽기 전용 원천 → 이번 바퀴 cache). 잠금창(2025-01-01 이후) owner 행은 SQL에서 잘라 내고 건수만 기록.
import os, re, sys, json, time, sqlite3
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
T = time.time()
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = B + r'\분석\모델경쟁_20260927'
W = B + r'\분석\검증루프_20261012\C02\C02-T4'
C = W + r'\cache'
con = sqlite3.connect('file:' + B + r'\수집파이프라인\registry_full.sqlite?mode=ro', uri=True)
log = {}
# 1) owner: 2025 이전만 적재, 잠금 행은 건수만
log['owner_rows_ge_20250101_count_only'] = con.execute("select count(*) from owner where ownerRgstCsDate >= '20250101'").fetchone()[0]
log['owner_rows_null_date'] = con.execute("select count(*) from owner where ownerRgstCsDate is null").fetchone()[0]
o = pd.read_sql("select rgst_no, seq, ownerName nm, ownerNatl natl, ownerRgstCsDate d, ownerRgstCsName cs from owner where ownerRgstCsDate < '20250101'", con)
log['owner_rows_loaded_lt_2025'] = len(o)
assert o.d.max() < '20250101'
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
CORP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|코포레이션|인코포레이티드|리미티드|컴퍼니|co\.|ltd|inc\.|corp|gmbh|llc", re.I)
INST = re.compile(r"산학협력단|산학협력|대학교|대학|연구원|연구소|재단법인|사단법인|학교법인|공사|진흥원|기술원")
o['nm'] = o.nm.fillna('')
o['nn'] = o.nm.map(norm)
o['corp'] = o.nm.map(lambda s: bool(CORP.search(s)))
o['inst'] = o.nm.map(lambda s: bool(INST.search(s)))
ACC = ['양도', '일부양도', '지분양도', '신탁계약에 의한 양도']
o['acc'] = o.cs.isin(ACC)
o.to_parquet(C + r'\own_lt2025.parquet', index=False)
log['acc_rows_lt2025'] = int(o.acc.sum())
# 2) patent / applicant
pat = pd.read_sql("select rgst_no, ksic, applNo appl_no, opnDate, pubDate, rgstDate from patent", con)
for c_ in ['opnDate', 'pubDate', 'rgstDate']:
    pat[c_] = pat[c_].where(pat[c_].fillna('').str.len() == 8)
pat['vis'] = pat[['opnDate', 'pubDate', 'rgstDate']].fillna('99999999').min(axis=1).replace('99999999', None)
log['patent_vis_null'] = int(pat.vis.isna().sum())
app = pd.read_sql("select rgst_no, applicantCd cd, applicantName an from applicant", con)
con.close()
app = app.dropna(subset=['cd'])
app['nn'] = app.an.fillna('').map(norm)
app = app.merge(pat[['rgst_no', 'appl_no', 'ksic', 'vis', 'rgstDate']], on='rgst_no', how='left')
app.to_parquet(C + r'\app_reg.parquet', index=False)
pat.to_parquet(C + r'\pat.parquet', index=False)
# 3) 문서 표: 등록(참 코드) + 미등록 공개 출원(T1 app_pub 법인 코드)
ap = pd.read_parquet(MC + r'\T1_산업대기실\cache\app_pub.parquet')
ap = ap[~ap.reg].copy()
ap['vis'] = ap.pub_date.where(ap.pub_date.fillna('').astype(str).str.len() == 8)
log['app_pub_unreg_rows'] = len(ap); log['app_pub_unreg_vis_null'] = int(ap.vis.isna().sum())
d1 = app[['cd', 'appl_no', 'ksic', 'vis']].assign(reg=True)
d2 = ap.rename(columns={'applicant_cd': 'cd'})[['cd', 'appl_no', 'ksic', 'vis']].assign(reg=False)
docs = pd.concat([d1, d2], ignore_index=True).dropna(subset=['vis']).drop_duplicates(['cd', 'appl_no'])
docs['ksic'] = docs.ksic.astype(str)
docs.to_parquet(C + r'\docs.parquet', index=False)
log['docs_rows'] = len(docs); log['docs_firms_prefix1'] = int(docs[docs.cd.str[0] == '1'].cd.nunique())
# 4) 공개 배정 원천: TXT-FREE, KR 메인 IPC sg (rec_conc)
tf = pd.read_parquet(MC + r'\T1_산업대기실\TXT_R9\cache\kr_txtfree.parquet')
dk = duckdb.connect(); dk.execute("set memory_limit='3GB'; set threads=2")
rc = dk.sql(f"select appl_no, any_value(sg) sg from read_parquet('{(MC + r'/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet').replace(chr(92), '/')}') where cc='KR' group by appl_no").df()
kk = pd.read_parquet(MC + r'\T1_산업대기실\TXT_R9\cache\kr_txt.parquet', columns=['appl_no', 'appl_year'])
lab = kk.merge(tf, on='appl_no', how='left').merge(rc, on='appl_no', how='left')
lab['sg'] = lab.sg.where(lab.sg.notna() & (lab.sg != ''))
lab.to_parquet(C + r'\kr_publab_src.parquet', index=False)
log['kr_lab_rows'] = len(lab); log['kr_lab_sg_null'] = int(lab.sg.isna().sum()); log['kr_lab_kfree_null'] = int(lab.k_free.isna().sum())
log['docs_appl_in_lab'] = float(docs.appl_no.isin(set(lab.appl_no)).mean())
log['elapsed'] = round(time.time() - T, 1)
json.dump(log, open(C + r'\s1_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(log)
