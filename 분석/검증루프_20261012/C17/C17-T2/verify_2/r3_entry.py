# -*- coding: utf-8 -*-
"""verify_2 r3: R 코드 확정 오매칭 121쌍 중 공개 경로 전용 양성 16쌍의 진입 근거 문서(출원 2020–23, 공개·등록 ≤ 2025-08-22)를
 등록원부(원출원인 코드)·권리목록 적중(rl_hits)으로 직접 판정. 산출: verify_2/r3_out.json, r3_entry_docs.csv"""
import pathlib, sqlite3, json, re
import numpy as np, pandas as pd
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
V = B/'분석/검증루프_20261012'; C = V/'C17/C17-T2'; CC = C/'cache'; OUT = C/'verify_2'
TC = B/'분석/모델경쟁_20260927/T1_산업대기실/cache'
pr1 = pd.read_csv(OUT/'r1_pairs.csv', dtype={'firm': str})
T = pr1[(pr1.y_pub == 1) & (pr1.y_reg == 0)][['firm', 'k', 'ptype']]
a = pd.read_parquet(TC/'app_pub.parquet'); K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.unique()); ki = {k: i for i, k in enumerate(K)}
a['k'] = a.ksic.astype(str).map(ki); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int); a = a[a.ay >= 2000]
def s8(x):
    s = pd.Series(x).fillna('').astype(str).str.strip().str[:8]
    return pd.to_numeric(s.where(s.str.len() == 8, '99999999'), errors='coerce').fillna(99999999).astype(np.int64).values
a['v'] = np.minimum(s8(a.pub_date.values), s8(a.rgst_date.values)); a['applicant_cd'] = a.applicant_cd.astype(str)
E = a[(a.ay > 2019) & (a.ay <= 2023) & (a.v <= 20250822)].merge(T, left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])
E['appl_no'] = E.appl_no.astype(str)
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
pat = pd.read_sql('select rgst_no, applNo from patent', con); pat['applNo'] = pat.applNo.astype(str)
ra = pd.read_sql('select rgst_no, applicantCd, applicantName from applicant', con); ra['applicantCd'] = ra.applicantCd.astype(str); con.close()
a2r = dict(zip(pat.applNo, pat.rgst_no)); E['rg'] = E.appl_no.map(a2r)
cb = ra.groupby('rgst_no').applicantCd.apply(lambda s: '/'.join(sorted(set(s)))).to_dict(); nb = ra.groupby('rgst_no').applicantName.apply(lambda s: '/'.join(sorted(set(s.astype(str))))).to_dict()
E['reg_codes'] = E.rg.map(cb); E['reg_names'] = E.rg.map(nb)
RL = pd.read_parquet(CC/'rl_hits.parquet'); RL['appl_no'] = RL.appl_no.astype(str); RL['holder_cd'] = RL.holder_cd.astype(str)
hb = RL.groupby('appl_no').holder_cd.apply(lambda s: '/'.join(sorted(set(s)))).to_dict(); E['rl_holders'] = E.appl_no.map(hb)
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '출원인명']); kp['출원번호'] = kp['출원번호'].astype(str)
E['kr_name'] = E.appl_no.map(kp.drop_duplicates('출원번호').set_index('출원번호')['출원인명'])
def st(r):
    if isinstance(r.reg_codes, str): return 'reg_firm' if r.firm in r.reg_codes.split('/') else 'reg_other'
    if isinstance(r.rl_holders, str): return 'rl_firm' if r.firm in r.rl_holders.split('/') else 'rl_other'
    return 'undet'
E['st'] = E.apply(st, axis=1)
E[['firm', 'k', 'ptype', 'appl_no', 'ay', 'kr_name', 'reg_codes', 'reg_names', 'rl_holders', 'st']].to_csv(OUT/'r3_entry_docs.csv', index=False, encoding='utf-8-sig')
pair = E.groupby(['firm', 'k']).st.apply(lambda s: 'any_firm' if s.isin(['reg_firm', 'rl_firm']).any() else ('all_other' if s.isin(['reg_other', 'rl_other']).all() else ('some_other' if s.isin(['reg_other', 'rl_other']).any() else 'undet')))
out = dict(pairs=int(len(T)), pairs_with_docs=int(E.groupby(['firm', 'k']).ngroups), docs=int(len(E)), doc_status=E.st.value_counts().to_dict(), pair_status=pair.value_counts().to_dict())
json.dump(out, open(OUT/'r3_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
