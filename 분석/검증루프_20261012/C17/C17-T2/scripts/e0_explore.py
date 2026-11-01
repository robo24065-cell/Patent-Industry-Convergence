# -*- coding: utf-8 -*-
"""C17-T2 e0(사전 탐색, 판정 미사용): 이름경로 근거 문서 중 '코드 판정 가능' 문서가 얼마나 되는지 센다.
 - 미등록(kr_patents 등록번호 공란) 문서가 등록원부(registry_full.patent.applNo)·권리목록(business_rights.right_list.appl_no)에 있는지
 결과: cache/e0_out.json, cache/rl_hits.parquet (right_list에서 근거 문서 appl_no에 해당하는 행)"""
import sys, pathlib, json, sqlite3, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
import numpy as np, pandas as pd
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
HERE = pathlib.Path(__file__).resolve().parent.parent; CC = HERE/'cache'
C16 = B/'분석/검증루프_20261012/C16/C16-T1'
DOC = pd.read_parquet(C16/'cache/alias_docs.parquet'); DOC['appl_no'] = DOC.appl_no.astype(str)
need = set(DOC.appl_no)
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
con = ro(B/'수집파이프라인/registry_full.sqlite')
pat = pd.read_sql('select rgst_no, applNo, applDate, rgstDate from patent', con); con.close()
pat['applNo'] = pat.applNo.astype(str)
ph = pat[pat.applNo.isin(need)]
out = dict(docs=len(DOC), reg_by_applNo=int(ph.applNo.nunique()))
print('registry', len(pat), len(ph), round(time.time() - t0), flush=True)
con = ro(B/'수집파이프라인/business_rights.sqlite')
rows = []; n = 0
cur = con.execute("select applicant_cd, rgst_no, rgst_date, appl_no, corp_rgstno, business_no, json_extract(raw,'$.applicantInfo'), json_extract(raw,'$.rightHolderInfo'), json_extract(raw,'$.pertDiv'), json_extract(raw,'$.rgstStatus') from right_list")
while True:
    ch = cur.fetchmany(200000)
    if not ch: break
    n += len(ch)
    for r in ch:
        if r[3] in need: rows.append(r)
    if n % 2000000 == 0: print(n, len(rows), round(time.time() - t0), G.rss_now(), flush=True)
con.close()
RL = pd.DataFrame(rows, columns=['holder_cd', 'rgst_no', 'rgst_date', 'appl_no', 'corp_rgstno', 'business_no', 'ainfo', 'hinfo', 'pert', 'st'])
RL.to_parquet(CC/'rl_hits.parquet', index=False)
out['right_list_rows'] = n; out['rl_hit_rows'] = len(RL); out['rl_hit_appl'] = int(RL.appl_no.nunique())
D = DOC.assign(in_reg=DOC.appl_no.isin(set(ph.applNo)), in_rl=DOC.appl_no.isin(set(RL.appl_no)))
tab = D.groupby(['grp', 'is_reg']).agg(n=('appl_no', 'size'), in_reg=('in_reg', 'sum'), in_rl=('in_rl', 'sum')).reset_index()
out['tab'] = tab.astype(str).to_dict('records')
fm = D.merge(RL[['appl_no', 'holder_cd']].drop_duplicates(), on='appl_no', how='left')
fm['holder_is_firm'] = fm.holder_cd.astype(str) == fm.firm.astype(str)
g = fm[fm.holder_cd.notna()].groupby(['grp', 'is_reg', 'appl_no', 'firm']).holder_is_firm.any().reset_index()
out['holder_is_firm'] = g.groupby(['grp', 'is_reg']).holder_is_firm.agg(['size', 'sum']).reset_index().astype(str).to_dict('records')
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
json.dump(out, open(CC/'e0_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
