# -*- coding: utf-8 -*-
"""주장 정의(최대 seq 최종권리자 명칭에 F 별칭 포함 = 보유 유지) 재구현으로 55.5%/1.2%/26.0% 재계산. 생존은 별칭 없는 (F,등록번호) 권리자란 존재."""
import pathlib, json, sqlite3
import pandas as pd, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
TOK = ['주식회사','(주)','㈜','유한회사','유한책임회사','재단법인','사단법인','학교법인','가부시키가이샤','가부시끼가이샤','가부시기가이샤','가부시키카이샤','코포레이션','인코포레이티드','리미티드','컴퍼니']
TR = str.maketrans('', '', ' \t\n.,-()\'"·&')
def nz(s):
    s = '' if s is None else str(s)
    for t in TOK: s = s.replace(t, '')
    return s.translate(TR).lower()
br = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri()+'?mode=ro', uri=True)
rg = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri()+'?mode=ro', uri=True)
aq = pd.read_sql("select applicant_cd, applicant_name from applicant_query where status='OK'", br); OK = set(aq.applicant_cd)
ap = pd.read_sql("select rgst_no, applicantCd, applicantName from applicant where rgst_no like '10%'", rg)
names = {}
for c, n in zip(aq.applicant_cd, aq.applicant_name): names.setdefault(c, set()).add(nz(n))
for c, n in zip(ap.applicantCd, ap.applicantName):
    if c in OK: names.setdefault(c, set()).add(nz(n))
names = {k: [x for x in v if len(x) >= 2] for k, v in names.items()}
def mine(c, info):
    a = nz(info); parts = None
    for n in names.get(c, []):
        if len(n) >= 3:
            if n in a: return True
        else:
            if parts is None: parts = [nz(p) for p in str(info).split(',')]
            if n in parts: return True
    return False
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet'); c = list(kr.columns)
kr = kr[[c[3], c[4], c[2]]]; kr.columns = ['rn', 'rd', 'ay']; kr['rn'] = kr.rn.astype(str); kr = kr[kr.rn.str[:2] == '10'].drop_duplicates('rn')
A = ap[ap.applicantCd.isin(OK)][['rgst_no', 'applicantCd']].drop_duplicates().merge(kr, left_on='rgst_no', right_on='rn')
A = A[(pd.to_numeric(A.ay, errors='coerce') >= 2000) & (A.rd.astype(str).str[:4].astype(int) <= 2019)].copy()
rl = pd.read_sql("select distinct applicant_cd, rgst_no from right_list where rgst_no like '10%' and json_extract(raw,'$.pertDiv')='권리자란'", br)
s = set(zip(rl.applicant_cd, rl.rgst_no)); A['s'] = [(a, b) in s for a, b in zip(A.applicantCd, A.rgst_no)]
ow = pd.read_sql("select rgst_no, seq, ownerName from owner where rgst_no like '10%'", rg)
mx = ow.groupby('rgst_no').seq.max()
fin = ow[ow.seq == ow.rgst_no.map(mx)].groupby('rgst_no').ownerName.apply(lambda x: ','.join(map(str, x))).to_dict()
A['ret'] = [mine(a, fin.get(b, '')) for a, b in zip(A.applicantCd, A.rgst_no)]
O = dict(n=int(len(A)), transferred_share=float((~A.ret).mean()), dropout_if_retained=float(1 - A[A.ret].s.mean()), dropout_if_transferred=float(1 - A[~A.ret].s.mean()), dropout=float(1 - A.s.mean()))
json.dump(O, open(HERE/'r2_transfer_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
