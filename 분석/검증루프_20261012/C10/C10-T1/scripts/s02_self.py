# -*- coding: utf-8 -*-
"""s02: 디자인·실용신안 행의 자사(원출원인=조회 법인) 판정 — T1 prep2.py와 같은 별칭·정규화·규칙(독립 재구현).
 + 동결 원천표 src_design/src_um(R3/cache, 읽기만)과 행 정렬 → cache/dsum_flags.parquet"""
import pathlib, re, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CACHE = T1/'cache'; R3 = T1/'R3'
t0 = time.time()
d = pd.read_parquet(HERE/'cache/rl_dsum.parquet')
firms = pd.read_parquet(CACHE/'firms.parquet'); firms = firms[firms.status == 'OK']
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd', 'applicantName'])
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
def norm(s): return STRIP.sub('', str(s)).lower()
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
def is_self(cd, ainfo):
    a = norm(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(ainfo).split(',')): return True
    return False
d['self'] = [is_self(c, a) for c, a in zip(d.applicant_cd.values, d.ainfo.values)]
d['in_holder'] = [is_self(c, h) for c, h in zip(d.applicant_cd.values, d.hinfo.values)]
d['a_eq_h'] = [norm(a) == norm(h) for a, h in zip(d.ainfo.values, d.hinfo.values)]
d['ainfo_na'] = d.ainfo.isna() | (d.ainfo.fillna('').str.strip() == '')
print('self done', round(time.time() - t0), flush=True)
# 동결 원천표와 정렬
for kind in ('design', 'um'):
    s = pd.read_parquet(R3/'cache'/f'src_{kind}.parquet', columns=['applicant_cd', 'rgst_no', 'rgst_date'])
    x = d[d.kind == kind].drop_duplicates(['applicant_cd', 'rgst_no'])
    s = s.merge(x[['applicant_cd', 'rgst_no', 'pert', 'tp', 'self', 'in_holder', 'a_eq_h', 'ainfo_na', 'st', 'expire_date']], on=['applicant_cd', 'rgst_no'], how='left', validate='1:1')
    print(kind, 'src rows', len(s), 'unmatched', int(s.self.isna().sum()), 'pert', s.pert.value_counts(dropna=False).to_dict(), flush=True)
    print(kind, 'self rate', round(float(s.self.mean()), 4), 'holder≠applicant(정규화 문자열)', round(1 - float(s.a_eq_h.mean()), 4), 'firm in holder', round(float(s.in_holder.mean()), 4), flush=True)
d.drop(columns=['title']).to_parquet(HERE/'cache/dsum_flags.parquet', index=False)
print('saved', round(time.time() - t0))
