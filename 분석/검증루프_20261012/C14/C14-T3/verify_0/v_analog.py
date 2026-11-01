# -*- coding: utf-8 -*-
"""verify_0 (i): 아날로그 누락률 독립 재계산 — 캐시 parquet 대신 원천 sqlite(registry_full.applicant, business_rights.right_list/applicant_query, mode=ro)를
 SQL로 직접 읽고, 자사 판정을 다른 코드(str.translate·토큰 치환 반복, 정규식 미사용)로 다시 구현한다. 제공 KR 등록 목록만 kr_registered.parquet에서 읽는다.
 → verify_0/v_analog_out.json (analog_dropout.csv 대비 차)"""
import sqlite3, pathlib, json, time, sys
import pandas as pd, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
t0 = time.time()
TOK = ['주식회사', '(주)', '㈜', '유한회사', '유한책임회사', '재단법인', '사단법인', '학교법인', '가부시키가이샤', '가부시끼가이샤', '가부시기가이샤', '가부시키카이샤',
       '코포레이션', '인코포레이티드', '리미티드', '컴퍼니']
TR = str.maketrans('', '', ' \t\n.,-()\'"·&')
def nz(s):
    s = '' if s is None else str(s)
    for t in TOK: s = s.replace(t, '')
    return s.translate(TR).lower()
br = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri() + '?mode=ro', uri=True)
rg = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
aq = pd.read_sql("select applicant_cd, applicant_name from applicant_query where status='OK'", br)
OK = set(aq.applicant_cd)
ap = pd.read_sql("select rgst_no, applicantCd, applicantName from applicant where substr(rgst_no,1,2)='10'", rg)
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['등록번호', '등록일자', '출원연도'])
kr.columns = ['rgst_no', 'rgst_date', 'ay']; kr['rgst_no'] = kr.rgst_no.astype(str); kr['rgst_date'] = kr.rgst_date.astype(str)
kr = kr[kr.rgst_no.str[:2] == '10'].drop_duplicates('rgst_no')
names = {}
for cd, nm in zip(aq.applicant_cd, aq.applicant_name): names.setdefault(cd, set()).add(nz(nm))
for cd, nm in zip(ap.applicantCd, ap.applicantName):
    if cd in OK: names.setdefault(cd, set()).add(nz(nm))
names = {k: [n for n in v if len(n) >= 2] for k, v in names.items()}
def mine(cd, info):
    a = nz(info); parts = None
    for n in names.get(cd, []):
        if len(n) >= 3:
            if n in a: return True
        else:
            if parts is None: parts = [nz(p) for p in str(info).split(',')]
            if n in parts: return True
    return False
rl = pd.read_sql("select applicant_cd, rgst_no, json_extract(raw,'$.pertDiv') pert, json_extract(raw,'$.applicantInfo') ainfo from right_list where substr(rgst_no,1,2)='10'", br)
rl = rl[rl.pert == '권리자란'].drop_duplicates(['applicant_cd', 'rgst_no'])
rl = rl[rl.rgst_no.isin(set(kr.rgst_no)) & rl.applicant_cd.isin(OK)]
rl['own'] = [mine(c, a) for c, a in zip(rl.applicant_cd.values, rl.ainfo.values)]
surv = set(zip(rl[rl.own].applicant_cd, rl[rl.own].rgst_no))
A = ap[ap.applicantCd.isin(OK)][['rgst_no', 'applicantCd']].drop_duplicates().merge(kr, on='rgst_no')
A = A[A.ay.astype(int) >= 2000]
A['ry'] = A.rgst_date.str[:4].astype(int)
A['s'] = [(c, r) in surv for c, r in zip(A.applicantCd.values, A.rgst_no.values)]
a19 = A[A.ry <= 2019]
OUT = dict(n_A_le2019=int(len(a19)), dropout_le2019=float(1 - a19.s.mean()))
grp = pd.cut(a19.ry, [1999, 2004, 2009, 2014, 2019], labels=['2000–04', '2005–09', '2010–14', '2015–19'])
OUT['by_group'] = {str(k): float(1 - v) for k, v in a19.groupby(grp, observed=False).s.mean().items()}
ref = pd.read_csv(HERE.parent/'analog_dropout.csv')
r_all = float(ref[(ref.dim == '전체(등록 2000–2019)')].dropout.iloc[0])
OUT['ref_dropout_le2019'] = r_all; OUT['diff_all'] = OUT['dropout_le2019'] - r_all
rg_ = ref[ref.dim == '등록연도군'].set_index('key').dropout
OUT['diff_by_group'] = {k: OUT['by_group'][k] - float(rg_[k]) for k in OUT['by_group']}
OUT['n_ref'] = int(ref[(ref.dim == '전체(등록 2000–2019)')].n_A.iloc[0])
OUT['verdict'] = '일치' if abs(OUT['diff_all']) <= 0.005 and all(abs(v) <= 0.005 for v in OUT['diff_by_group'].values()) else '불일치'
OUT['sec'] = round(time.time() - t0)
json.dump(OUT, open(HERE/'v_analog_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
