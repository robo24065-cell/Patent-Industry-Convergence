# -*- coding: utf-8 -*-
"""별칭 감사 진단(사후, 판정 규칙 변경 없음): 등록 근거 문서의 코드 불일치가 무엇인지 — 등록원부 코드의 이름·상태·CA 법인번호 비교"""
import sys, pathlib, sqlite3; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard16 as G
from c16common import *
DOC = pd.read_parquet(CC/'alias_docs.parquet')
w = DOC[DOC.is_reg & DOC.wrong].copy()
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', con); con.close()
fm = pd.read_parquet(CACHE/'firms.parquet'); st = dict(zip(fm.applicant_cd, fm.status)); fnm = dict(zip(fm.applicant_cd, fm.applicant_name))
ca = pd.read_parquet(B/'공모전데이터/외부데이터/DART/corp_applicant.parquet', columns=['출원인코드', 'jurir', 'bizr']); ca['출원인코드'] = ca['출원인코드'].astype(str)
cj = dict(zip(ca['출원인코드'], ca.jurir.fillna('').astype(str))); cb = dict(zip(ca['출원인코드'], ca.bizr.fillna('').astype(str)))
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd'])
rows = []
for r in w.itertuples():
    rr = ra[ra.rgst_no == r.rgst_no]
    for x in rr.itertuples():
        rows.append(dict(grp=r.grp, firm=r.firm, firm_name=fnm.get(r.firm), firm_status=st.get(r.firm), matched_name=r.name_raw, rgst_no=r.rgst_no, reg_code=x.applicantCd, reg_name=x.applicantName,
                         reg_code_status=st.get(x.applicantCd, '(목록 없음)'), reg_code_in_ins=bool(((ins.rgst_no == r.rgst_no) & (ins.applicant_cd == x.applicantCd)).any()),
                         firm_jurir=cj.get(r.firm, ''), reg_jurir=cj.get(x.applicantCd, ''), firm_bizr=cb.get(r.firm, ''), reg_bizr=cb.get(x.applicantCd, ''),
                         code_prefix_firm=str(r.firm)[:1], code_prefix_reg=str(x.applicantCd)[:1]))
    if len(rows) > 3000: break
X = pd.DataFrame(rows)
X['same_norm_name'] = [str(a).replace(' ', '') == str(b).replace(' ', '') for a, b in zip(X.matched_name, X.reg_name)]
out = dict(n_wrong_reg_docs=int(len(w)), rows=int(len(X)), reg_code_status=X.reg_code_status.value_counts().to_dict(), same_name_share=float(X.same_norm_name.mean()),
           reg_code_in_ins_share=float(X.reg_code_in_ins.mean()), firm_jurir_present=float((X.firm_jurir.str.len() > 5).mean()), reg_jurir_present=float((X.reg_jurir.str.len() > 5).mean()),
           both_jurir_and_differ=int(((X.firm_jurir.str.len() > 5) & (X.reg_jurir.str.len() > 5) & (X.firm_jurir != X.reg_jurir)).sum()),
           both_jurir_and_equal=int(((X.firm_jurir.str.len() > 5) & (X.reg_jurir.str.len() > 5) & (X.firm_jurir == X.reg_jurir)).sum()),
           prefix_pairs={f'{a}-{b}': int(n) for (a, b), n in X.groupby(['code_prefix_firm', 'code_prefix_reg']).size().items()})
X.to_csv(CC/'alias_reg_wrong_diag.csv', index=False, encoding='utf-8-sig')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)); print(X.head(25).to_string())
