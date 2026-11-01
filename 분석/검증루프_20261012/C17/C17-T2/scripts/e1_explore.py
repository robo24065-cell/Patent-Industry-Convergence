# -*- coding: utf-8 -*-
"""C17-T2 e1(사전 탐색, 판정 미사용): 권리목록 적중 문서의 원출원인명(applicantInfo)·보유자 코드와 주소 규칙의 관계를 본다.
 결과: cache/e1_out.json, cache/e1_examples.csv"""
import sys, pathlib, json, re, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
import numpy as np, pandas as pd
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
HERE = pathlib.Path(__file__).resolve().parent.parent; CC = HERE/'cache'
C16 = B/'분석/검증루프_20261012/C16/C16-T1'; TC = B/'분석/모델경쟁_20260927/T1_산업대기실/cache'
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
DOC = pd.read_parquet(C16/'cache/alias_docs.parquet'); DOC['appl_no'] = DOC.appl_no.astype(str); DOC['firm'] = DOC.firm.astype(str)
RL = pd.read_parquet(CC/'rl_hits.parquet'); RL['holder_cd'] = RL.holder_cd.astype(str)
firms = pd.read_parquet(TC/'firms.parquet'); firms['applicant_cd'] = firms.applicant_cd.astype(str)
app = pd.read_parquet(TC/'ins_applicant.parquet'); app['applicantCd'] = app.applicantCd.astype(str)
names = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                   app[['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
names['n'] = names.nm.map(norm); fn = names.groupby('applicant_cd').n.apply(set).to_dict()
out = {}
out['firm_code_prefix_P'] = DOC[DOC.grp == 'P'].firm.str[0].value_counts().to_dict()
out['firm_code_prefix_R'] = DOC[DOC.grp == 'R'].firm.str[0].value_counts().to_dict()
qs = pd.read_parquet(TC/'firms.parquet')
out['firms_status_of_doc_firms'] = qs[qs.applicant_cd.astype(str).isin(set(DOC.firm))].status.value_counts().to_dict()
D = DOC.merge(RL, on='appl_no', how='left')
D['in_rl'] = D.holder_cd.notna()
D['holder_firm'] = D.holder_cd == D.firm


def ainfo_match(firm, ainfo):
    if not isinstance(ainfo, str): return None
    ns = fn.get(firm, set())
    parts = [norm(p) for p in re.split(r'[,|]', ainfo)]
    return any(p in ns for p in parts if p)


D['a_match'] = [ainfo_match(f, a) for f, a in zip(D.firm, D.ainfo)]
D['h_match'] = [ainfo_match(f, a) for f, a in zip(D.firm, D.hinfo)]
x = D[D.in_rl]
out['tab_inrl'] = x.groupby(['grp', 'is_reg', 'holder_firm', 'a_match', 'h_match'], dropna=False).size().reset_index(name='n').astype(str).to_dict('records')
pu = D[(D.grp == 'P') & ~D.is_reg].drop_duplicates('appl_no')
out['Punreg_by_e2_inrl'] = pu.groupby(['e2']).agg(n=('appl_no', 'size'), in_rl=('in_rl', 'sum'), holder_firm=('holder_firm', 'sum'), a_match=('a_match', lambda s: int((s == True).sum()))).reset_index().astype(str).to_dict('records')
out['Punreg_by_ay_inrl'] = pu.groupby(['ay']).agg(n=('appl_no', 'size'), in_rl=('in_rl', 'sum')).reset_index().astype(str).to_dict('records')
pu = pu.assign(dom=pu.firm.str[0].isin(['1', '2', '3']))
out['Punreg_by_dom_inrl'] = pu.groupby(['dom', 'e2']).agg(n=('appl_no', 'size'), in_rl=('in_rl', 'sum'), holder_firm=('holder_firm', 'sum')).reset_index().astype(str).to_dict('records')
ex = pd.concat([x[(x.grp == 'P') & ~x.is_reg & ~(x.holder_firm & (x.a_match == True))].head(60), x[(x.grp == 'P') & ~x.is_reg & (x.e2 == 'mismatch')].head(40)])
ex[['grp', 'firm', 'appl_no', 'ay', 'name_raw', 'e2', 'app_regions', 'firm_regions', 'holder_cd', 'ainfo', 'hinfo', 'pert', 'st', 'a_match', 'h_match', 'corp_rgstno', 'firm_jurir']].to_csv(CC/'e1_examples.csv', index=False, encoding='utf-8-sig')
out['secs'] = round(time.time() - t0)
json.dump(out, open(CC/'e1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
