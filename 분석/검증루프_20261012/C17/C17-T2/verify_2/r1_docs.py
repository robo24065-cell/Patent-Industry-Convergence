# -*- coding: utf-8 -*-
"""verify_2 r1: R 코드 확정 오매칭 121쌍의 근거 문서를 등록원부에서 직접 다시 읽어 판정 재검토(읽기 전용).
 - 문서별 등록원부 원출원인 코드·이름·주소, 법인 코드의 등록원부 이름·주소와 비교
 - '다른 코드·같은 이름·같은 상세 주소'면 같은 실체의 중복 코드 의심
 - Ry 양성(18)의 구성(y_reg/y_pub)
 산출: verify_2/r1_out.json, verify_2/r1_docs.csv"""
import pathlib, sqlite3, json, re
import numpy as np, pandas as pd
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
V = B/'분석/검증루프_20261012'; C = V/'C17/C17-T2'; OUT = C/'verify_2'
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
out = {}
PJ = pd.read_parquet(C/'cache/pair_judg.parquet'); PJ['firm'] = PJ.firm.astype(str)
W = PJ[(PJ.grp == 'R') & (PJ.pv == 'wrong')][['firm', 'k', 'ptype']]
D = pd.read_parquet(C/'cache/doc_judg.parquet'); D['firm'] = D.firm.astype(str)
DW = D[D.grp == 'R'].merge(W, on=['firm', 'k'])
out['n_pairs'] = int(len(W)); out['n_docs'] = int(len(DW))
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd, applicantName, applicantAddr from applicant', con)
pat = pd.read_sql('select rgst_no, applNo, applDate, rgstDate from patent', con)
ow = pd.read_sql('select rgst_no, seq, finalOwnerYn, ownerName, ownerRgstCsReason from owner', con)
con.close()
ra['applicantCd'] = ra.applicantCd.astype(str); pat['applNo'] = pat.applNo.astype(str)
a2r = dict(zip(pat.applNo, pat.rgst_no))
DW['rg'] = np.where(DW.rgst_no.fillna('').astype(str).str.len() > 5, DW.rgst_no.fillna('').astype(str), DW.appl_no.astype(str).map(a2r).fillna(''))
addr_key = lambda s: ' '.join(str(s or '').replace(',', ' ').split()[:4])
firm_rows = ra[ra.applicantCd.isin(set(DW.firm))]
firm_names = firm_rows.assign(n=firm_rows.applicantName.map(norm)).groupby('applicantCd').n.apply(set).to_dict()
firm_addr = firm_rows.assign(a=firm_rows.applicantAddr.map(addr_key)).groupby('applicantCd').a.apply(set).to_dict()
rows = []
rag = {k: g for k, g in ra[ra.rgst_no.isin(set(DW.rg))].groupby('rgst_no')}
for r in DW.itertuples(index=False):
    g = rag.get(r.rg)
    if g is None:
        rows.append(dict(firm=r.firm, k=r.k, appl_no=r.appl_no, ay=r.ay, ptype=r.ptype, rg=r.rg, in_reg=False)); continue
    codes = set(g.applicantCd); nms = set(g.applicantName.map(norm)); adrs = set(g.applicantAddr.map(addr_key))
    fn = firm_names.get(r.firm, set()) | {r.nn}
    same_name_other = bool(nms & fn)
    same_addr_other = bool(adrs & firm_addr.get(r.firm, set()))
    rows.append(dict(firm=r.firm, k=r.k, appl_no=r.appl_no, ay=r.ay, ptype=r.ptype, rg=r.rg, in_reg=True, firm_in_codes=r.firm in codes,
                     reg_codes='/'.join(sorted(codes)), reg_names='/'.join(sorted(g.applicantName.astype(str)))[:80], kr_name=r.name_raw,
                     same_name_other=same_name_other, same_addr4_other=same_addr_other, v=r.v))
X = pd.DataFrame(rows)
X.to_csv(OUT/'r1_docs.csv', index=False, encoding='utf-8-sig')
out['docs_in_reg'] = int(X.in_reg.sum()); out['firm_in_codes_any'] = int(X.firm_in_codes.fillna(False).sum())
out['by_type_docs'] = X.groupby('ptype').agg(n=('appl_no', 'size'), same_name=('same_name_other', 'sum'), same_addr4=('same_addr4_other', 'sum'),
                                              both=('same_name_other', lambda s: int((s & X.loc[s.index, 'same_addr4_other']).sum()))).astype(int).to_dict('index')
# 쌍 단위: 모든 문서가 '같은 이름+같은 주소 4어절'인 다른 코드(중복 코드 의심)
X['dup_susp'] = X.same_name_other.fillna(False) & X.same_addr4_other.fillna(False)
pr = X.groupby(['firm', 'k', 'ptype']).agg(any_dup=('dup_susp', 'any'), all_dup=('dup_susp', 'all')).reset_index()
out['pairs_dup_suspect_any'] = int(pr.any_dup.sum()); out['pairs_dup_suspect_all'] = int(pr.all_dup.sum())
out['pairs_dup_by_type'] = pr.groupby('ptype')[['any_dup', 'all_dup']].sum().astype(int).to_dict('index')
# Ry 구성
L = pd.read_parquet(V/'C16/C16-T1/cache/labels.parquet'); L = L[(L.src == 'te') & L.prior_any]; L['firm'] = L.firm.astype(str)
m = W.merge(L[['firm', 'k', 'y_reg', 'y_pub', 'y_u']], on=['firm', 'k'], how='left')
out['Ry_pos'] = int(m.y_u.sum()); out['Ry_pos_reg'] = int(m.y_reg.sum()); out['Ry_pos_pub_only'] = int(((m.y_pub == 1) & (m.y_reg == 0)).sum())
pr = pr.merge(m[['firm', 'k', 'y_reg', 'y_pub', 'y_u']], on=['firm', 'k'])
pr.to_csv(OUT/'r1_pairs.csv', index=False, encoding='utf-8-sig')
out['dup_suspect_all_pos'] = int(pr[pr.all_dup].y_u.sum())
json.dump(out, open(OUT/'r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=int)
print(json.dumps(out, ensure_ascii=False, indent=1, default=int))
