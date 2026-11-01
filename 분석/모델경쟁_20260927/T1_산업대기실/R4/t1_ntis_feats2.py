# -*- coding: utf-8 -*-
"""T1 R4 — NTIS 과제 수행 이력을 '수행기관명 ↔ 출원인명' 정규화 일치로 연결(NTIS Open API 활용).
R4E04(사업자번호 연결)는 사업자번호가 2010년 과제부터만 있어 c ≤ 2010 원점이 결측이었다. 수행기관명은 2002년부터 있으므로 모든 원점에서 값이 생긴다.
- 연결표: T1 출원인 목록(cache/firm_meta, 정규화 이름이 유일한 출원인만) — 미래 NTIS 활동과 무관한 정적 목록(2010년 이후 과제로 만든 이름↔사업자번호 표는
  '2010년 이후에도 과제를 한 기업'만 남기는 선택 편향이 있어 쓰지 않았다).
- 2012년 과제 검증: 기업 수행 과제 중 이름 일치 44.7%, 사업자번호 경로와 둘 다 있는 과제에서 같은 출원인 92.5%.
- 시점: 과제연도 ≤ c−1. 배분표는 성과 Year ≤ c−2(t1_ntis_feats.py 와 같음).
출력: R4/cache/ntis_name_o{c}.parquet (firm, k, q_any, q_lnf, q_n5, q_tgt_cls)
"""
import sys, re, time, pathlib
import numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parent; T1 = R4.parent; R3 = T1 / 'R3'
EXTD = T1.parent / '_shared' / 'ext'
ORIG = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2008, 2009, 2010, 2011, 2012, 2015]
for c in ORIG:
    assert c + 4 <= 2019


def norm(s):
    s = str(s).upper()
    s = re.sub(r'\(주\)|㈜|주식회사|\(유\)|유한회사|\(재\)|재단법인|사단법인|\(사\)|CO\.,? ?LTD\.?|INC\.?|CORPORATION|CORP\.?', '', s)
    return re.sub(r'[\s\.\,\(\)\-]', '', s)


src = pd.read_parquet(R3 / 'cache' / 'src_kr_ipc.parquet', columns=['ksic'])
K = sorted(src.ksic.astype(str).unique()); del src
fm = pd.read_parquet(T1 / 'cache' / 'firm_meta.parquet', columns=['applicant_cd', 'applicant_name'])
fm['n'] = fm.applicant_name.map(norm); fm['applicant_cd'] = fm.applicant_cd.astype(str)
u = fm.groupby('n').applicant_cd.nunique(); nm = fm[fm.n.isin(u[u == 1].index) & (fm.n.str.len() >= 2)].set_index('n').applicant_cd
pj = pd.read_parquet(EXTD / 'ntis_project_raw.parquet', columns=['pjt_no', 'year', 'research_agency', 'gov_funds', 'cls_L_canon'])
pj['applicantCd'] = pj.research_agency.map(norm).map(nm)
pj = pj[pj.applicantCd.notna()].copy()
pj['year'] = pd.to_numeric(pj.year, errors='coerce'); pj['gov_funds'] = pd.to_numeric(pj.gov_funds, errors='coerce').fillna(0.0)
pj['cls'] = pj.cls_L_canon.astype(str); pj.loc[pj.cls.isin(['NA', '<NA>', 'nan', 'None', '']), 'cls'] = np.nan
print('이름 연결 과제', len(pj), '출원인', pj.applicantCd.nunique(), flush=True)
cls_of = pd.read_parquet(EXTD / 'ntis_project_raw.parquet', columns=['pjt_no', 'cls_L_canon'])
cls_of['pjt_no'] = cls_of.pjt_no.astype(str)
cls_of = cls_of.drop_duplicates('pjt_no').set_index('pjt_no').cls_L_canon.astype(str)
cls_of = cls_of[~cls_of.isin(['NA', '<NA>', 'nan', 'None', ''])]
lk = pd.read_parquet(EXTD / 'ntis_patent_link.parquet', columns=['ksic', 'Year', 'ProjectID'])
lk['yr'] = pd.to_numeric(lk.Year, errors='coerce'); lk['cls'] = lk.ProjectID.astype(str).map(cls_of); lk['ksic'] = lk.ksic.astype(str)
lk = lk[lk.cls.notna()]
rp = pd.read_parquet(EXTD / 'ntis_rpatent_raw.parquet', columns=['num_digits', 'Year', 'is_kr', 'IprType', 'ProjectID'])
rp = rp[(rp.is_kr == True) & (rp.IprType.astype(str) == '특허') & (rp.num_digits.astype(str).str.len() == 13)]
rp = pd.DataFrame({'yr': pd.to_numeric(rp.Year, errors='coerce').values, 'cls': rp.ProjectID.astype(str).map(cls_of).values}).dropna()


def alloc(c):
    den = rp[rp.yr <= c - 2].groupby('cls').size()
    if den.sum() < 1000:
        return None
    t = lk[lk.yr <= c - 2].assign(m=1.0)
    num = t.groupby(['cls', 'ksic']).m.sum().unstack(fill_value=0.0).reindex(index=den.index, fill_value=0.0).reindex(columns=K, fill_value=0.0)
    return den.index, num.values / den.values[:, None], num.values.sum(0) / max(den.values.sum(), 1)


for c in ORIG:
    outp = R4 / 'cache' / f'ntis_name_o{c}.parquet'
    if outp.exists():
        print('exists', outp); continue
    te = pd.read_parquet(R3 / 'cache' / f'o{c}' / 'prep_te.parquet', columns=['firm', 'k', 'prior_any'])
    te = te[~te.prior_any.values].reset_index(drop=True); te['firm'] = te.firm.astype(str)
    firms = pd.Index(te.firm.unique()); fi = pd.Series(np.arange(len(firms)), index=firms); nF = len(firms)
    q = pj[(pj.year <= c - 1) & pj.applicantCd.isin(firms)]; q5 = q[q.year >= c - 5]
    q_any = (q.groupby('applicantCd').size().reindex(firms).fillna(0).values > 0).astype(float)
    q_lnf = np.log1p(q5.groupby('applicantCd').gov_funds.sum().reindex(firms).fillna(0).values / 1e6)
    q_n5 = np.log1p(q5.groupby('applicantCd').pjt_no.nunique().reindex(firms).fillna(0).values)
    A = alloc(c)
    if A is None:
        L = np.full((nF, 50), np.nan)
    else:
        idx_, Sk, sbar = A; cix = pd.Series(np.arange(len(idx_)), index=idx_)
        qc = q[q.cls.isin(idx_)]
        M = np.zeros((nF, len(idx_))); np.add.at(M, (fi[qc.applicantCd].values, cix[qc.cls].values), 1.0)
        has = M.sum(1) > 0
        L = np.log(((M / np.maximum(M.sum(1, keepdims=True), 1e-9)) @ Sk + 1e-4) / (sbar + 1e-4)); L[~has] = 0.0
    f = fi[te.firm].values; k = te.k.values.astype(int)
    out = pd.DataFrame({'firm': te.firm.values, 'k': k, 'q_any': q_any[f].astype(np.float32), 'q_lnf': q_lnf[f].astype(np.float32),
                        'q_n5': q_n5[f].astype(np.float32), 'q_tgt_cls': L[f, k].astype(np.float32)})
    out.to_parquet(outp, index=False)
    print(c, 'q_any 기업', int(q_any.sum()), '/', nF, 'q_any 쌍', round(float(out.q_any.mean()), 4), 'q_tgt_cls 결측', round(float(out.q_tgt_cls.isna().mean()), 3), flush=True)
