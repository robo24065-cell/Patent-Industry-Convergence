# -*- coding: utf-8 -*-
"""C17-T2 s1b(진단): P 오매칭·이전형 문서의 상대 코드, 이름, 법인번호(CA·권리목록), 주소 시도를 나열해 '같은 실체의 중복 코드' 가능성을 본다.
 산출: cache/s1b_wrong_docs.csv, cache/s1b_out.json"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
D = pd.read_parquet(CC/'doc_judg.parquet')
W = D[(D.st == 'wrong')].copy()
con = ro(REG); ra = pd.read_sql('select rgst_no, applicantCd, applicantName, applicantAddr from applicant', con); pat = pd.read_sql('select rgst_no, applNo from patent', con); con.close()
ra['applicantCd'] = ra.applicantCd.astype(str); a2r = dict(zip(pat.applNo.astype(str), pat.rgst_no))
RL = pd.read_parquet(CC/'rl_hits.parquet'); RL['holder_cd'] = RL.holder_cd.astype(str); RL['corp'] = RL.corp_rgstno.map(digits)
ca = pd.read_parquet(CA, columns=['출원인코드', '출원인명', 'jurir', 'bizr']); ca['출원인코드'] = ca['출원인코드'].astype(str)
cj = {c: digits(j) for c, j in zip(ca['출원인코드'], ca.jurir)}
for c, j in zip(RL.holder_cd, RL.corp):
    if j and not cj.get(c): cj[c] = j
rname = ra.drop_duplicates('applicantCd').set_index('applicantCd').applicantName.to_dict(); raddr = ra.drop_duplicates('applicantCd').set_index('applicantCd').applicantAddr.to_dict()
con = ro(BIZ); aq = pd.read_sql('select applicant_cd, applicant_name from applicant_query', con); con.close(); aqn = dict(zip(aq.applicant_cd.astype(str), aq.applicant_name))
rows = []
for r in W.itertuples(index=False):
    rk = r.rgst_no if len(str(r.rgst_no)) > 5 else a2r.get(r.appl_no, '')
    others = set(ra[ra.rgst_no == rk].applicantCd) if rk else set()
    if not others:
        g = RL[RL.appl_no == r.appl_no]; others = set(g.holder_cd)
    for o in sorted(others):
        if o == r.firm: continue
        jf, jo = cj.get(r.firm, ''), cj.get(o, '')
        rows.append(dict(grp=r.grp, firm=r.firm, appl_no=r.appl_no, v=r.v, name_raw=r.name_raw, firm_name=rname.get(r.firm, aqn.get(r.firm, '')), other=o,
                         other_name=rname.get(o, aqn.get(o, '')), other_same_norm=norm(rname.get(o, aqn.get(o, ''))) == r.nn,
                         firm_corp=jf, other_corp=jo, corp_both=bool(jf and jo), corp_differ=bool(jf and jo and jf != jo),
                         firm_addr=str(raddr.get(r.firm, ''))[:30], other_addr=str(raddr.get(o, ''))[:30]))
X = pd.DataFrame(rows); X.to_csv(CC/'s1b_wrong_docs.csv', index=False, encoding='utf-8-sig')
o = {}
for g in ('P', 'R'):
    x = X[X.grp == g]
    o[g] = dict(rows=len(x), by_v=x.groupby('v').agg(n=('appl_no', 'size'), same_name=('other_same_norm', 'sum'), corp_both=('corp_both', 'sum'), corp_differ=('corp_differ', 'sum')).reset_index().astype(str).to_dict('records'))
jdump(o, CC/'s1b_out.json'); print(json.dumps(o, ensure_ascii=False, indent=1))
