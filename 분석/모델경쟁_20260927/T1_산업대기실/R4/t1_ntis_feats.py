# -*- coding: utf-8 -*-
"""T1 R4 — NTIS(국가R&D, NTIS Open API 활용) 기업×목표 KSIC 피처 계산. 원점 c마다 R4/cache/ntis_o{c}.parquet 저장.
시점 규칙(PROTOCOL_R4 §3, 원점 연말 = 정보 마감):
  - 정부 기원 특허 = 원점 연말까지 '보이는' 자사 특허 중 NTIS 성과 레코드 Year ≤ c 인 것.
      범위안: 제공 KR 출원(ay ≤ c, min(공개일, 등록일) ≤ c-12-31)  ← R3 기술근접 피처와 같은 가시 집합(출원 후 미래 등록 여부에 조건부 아님)
      범위밖: 자사 등록특허(ay ≤ c, 등록일 ≤ c-12-31) — business_rights 포트폴리오(T1 cache/oos, self)
  - 과제분류→KSIC 배분표: 성과 레코드 Year ≤ c−2 (2007 이전 성과 없음 → c=2008 은 NaN)
  - 과제 수행 이력(사업자번호 연결): 과제연도 ≤ c−1(이듬해 공개). 사업자번호가 2010년부터만 채워져 c−1 < 2010 이면 NaN.
🔒 c + 4 ≤ 2019 (R3 최종시험 첫 결과연도 2020 미만) assert. 레이블은 읽지 않는다(피처만).
"""
import sys, os, json, time, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp
R4 = pathlib.Path(__file__).resolve().parent; T1 = R4.parent; R3 = T1 / 'R3'
EXTD = T1.parent / '_shared' / 'ext'
T0 = time.time()
GRP = {'대학': 'univ', '출연연구소': 'gri', '국공립연구소': 'gri', '대기업': 'big', '중견기업': 'big', '중소기업': 'sme', '기타': 'oth', '정부부처': 'oth'}
ORIG = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2008, 2009, 2010, 2011, 2012, 2015]
for c in ORIG:
    assert c + 4 <= 2019, '🔒 R3 최종시험 구간과 겹치는 원점 금지'


def peak():
    try:
        import psutil; return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return -1


# ---------------- 공통 자료 ----------------
src = pd.read_parquet(R3 / 'cache' / 'src_kr_ipc.parquet', columns=['appl_no', 'ksic'])
K = sorted(src.ksic.astype(str).unique()); assert len(K) == 50
kidx = {k: i for i, k in enumerate(K)}
X = dict(np.load(R3 / 'cache' / 'src_kr_ipc_sc.npz', allow_pickle=True)); X['nsc'] = int(X['nsc'])
assert len(X['ay']) == len(src) and (src.ksic.astype(str).map(kidx).values == X['k']).all(), 'KSIC 색인 불일치'
nA = len(X['ay'])
E = sp.csr_matrix((X['ew'], (X['ea'], X['ej'])), shape=(nA, X['nsc']))
appl = src.appl_no.astype(str).values
del src

cls_of = pd.read_parquet(EXTD / 'ntis_project_raw.parquet', columns=['pjt_no', 'cls_L_canon'])
cls_of['pjt_no'] = cls_of.pjt_no.astype(str)
cls_of = cls_of.drop_duplicates('pjt_no').set_index('pjt_no').cls_L_canon.astype(str)
cls_of = cls_of[~cls_of.isin(['NA', '<NA>', 'nan', 'None', ''])]

# 범위안 정부 기원: 제공 KR 연결(출원번호) — 출원 행 a 별 (grp, yr, cls)
lk = pd.read_parquet(EXTD / 'ntis_patent_link.parquet', columns=['appl_no', 'ksic', 'Year', 'ProjectID', 'PerformAgent'])
lk['yr'] = pd.to_numeric(lk.Year, errors='coerce').astype(float)
lk['grp'] = lk.PerformAgent.astype(str).map(GRP).fillna('oth')
lk['cls'] = lk.ProjectID.astype(str).map(cls_of)
ai = pd.Series(np.arange(nA), index=appl); ai = ai[~ai.index.duplicated()]
gin = lk[lk.appl_no.astype(str).isin(ai.index)].copy()
# 같은 출원번호가 여러 KSIC 행이면 모든 행(a)에 붙인다
arows = pd.DataFrame({'appl_no': appl, 'a': np.arange(nA)})
gin = gin.merge(arows, on='appl_no')[['a', 'yr', 'grp', 'cls']]
print('범위안 정부 기원 연결 행', len(gin), '출원행', gin.a.nunique(), flush=True)

# 범위밖(자사 등록) 정부 기원: NTIS KR 특허 성과(13자리) ↔ 등록번호/출원번호
rp = pd.read_parquet(EXTD / 'ntis_rpatent_raw.parquet', columns=['num_digits', 'RegistType', 'Year', 'is_kr', 'IprType', 'ProjectID', 'PerformAgent', 'ipc_codes'])
rp = rp[(rp.is_kr == True) & (rp.IprType.astype(str) == '특허') & (rp.num_digits.astype(str).str.len() == 13)].copy()
rp['yr'] = pd.to_numeric(rp.Year, errors='coerce').astype(float)
rp['grp'] = rp.PerformAgent.astype(str).map(GRP).fillna('oth')
rp['cls'] = rp.ProjectID.astype(str).map(cls_of)
rp['num_digits'] = rp.num_digits.astype(str); rp['RegistType'] = rp.RegistType.astype(str)
oos = pd.read_parquet(T1 / 'cache' / 'oos.parquet', columns=['applicant_cd', 'rgst_no', 'ay', 'rgst_date', 'self'])
oos = oos[oos.self & (oos.rgst_no.astype(str).str[:2] == '10')].drop(columns='self')
oos['rgst_no'] = oos.rgst_no.astype(str); oos['applicant_cd'] = oos.applicant_cd.astype(str)
oos['rd'] = pd.to_numeric(oos.rgst_date.astype(str).str.replace('-', '').str[:8], errors='coerce')
a2r = pd.read_parquet(T1 / 'cache' / 'rl_pat.parquet', columns=['appl_no', 'rgst_no']).dropna().drop_duplicates('rgst_no')
a2r['appl_no'] = a2r.appl_no.astype(str).str.replace('-', '', regex=False); a2r['rgst_no'] = a2r.rgst_no.astype(str)
oos = oos.merge(a2r, on='rgst_no', how='left')
g_r = oos.merge(rp[rp.RegistType == '등록'], left_on='rgst_no', right_on='num_digits')
g_a = oos[oos.appl_no.notna()].merge(rp[rp.RegistType == '출원'], left_on='appl_no', right_on='num_digits')
gout = pd.concat([g_r, g_a], ignore_index=True)[['applicant_cd', 'rgst_no', 'ay', 'rd', 'yr', 'grp', 'cls', 'ipc_codes']]
print('범위밖 정부 기원 연결 행', len(gout), '특허', gout.rgst_no.nunique(), flush=True)
oos_tot = oos[['applicant_cd', 'rgst_no', 'ay', 'rd']]

# 배분표 재료(제공 KR 연결 성과 / 전체 KR 특허 성과)
lk_cls = lk[lk.cls.notna()][['ksic', 'yr', 'cls']].copy(); lk_cls['ksic'] = lk_cls.ksic.astype(str)
rp_cls = rp[rp.cls.notna()][['yr', 'cls']]

# 과제 수행(사업자번호) — 출원인코드 → 사업자번호(r3nts), 과제 raw corp_rgst_no(10자리 사업자번호)
ab = pd.read_parquet(EXTD / 'r3nts_applicant_status.parquet', columns=['applicantCd', 'bizno'])
ab = ab.dropna(); ab['applicantCd'] = ab.applicantCd.astype(str); ab['bizno'] = ab.bizno.astype(str).str.replace('-', '', regex=False)
pj = pd.read_parquet(EXTD / 'ntis_project_raw.parquet', columns=['pjt_no', 'year', 'corp_rgst_no', 'gov_funds', 'cls_L_canon'])
pj['bizno'] = pj.corp_rgst_no.astype(str).str.replace('-', '', regex=False)
pj = pj[pj.bizno.str.len() == 10].copy()
pj['year'] = pd.to_numeric(pj.year, errors='coerce'); pj['gov_funds'] = pd.to_numeric(pj.gov_funds, errors='coerce').fillna(0.0)
pj['cls'] = pj.cls_L_canon.astype(str); pj.loc[pj.cls.isin(['NA', '<NA>', 'nan', 'None', '']), 'cls'] = np.nan
pj = pj.merge(ab, on='bizno')[['applicantCd', 'pjt_no', 'year', 'gov_funds', 'cls']]
print('출원인 연결 과제 행', len(pj), '출원인', pj.applicantCd.nunique(), 'peak', peak(), flush=True)


def sc_list(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return sorted({x.strip().replace(' ', '')[:4] for x in str(s).replace('|', ',').split(',') if len(x.strip()) >= 4})


# 서브클래스 색인(범위안 E 행렬과 같은 순서)은 npz에 이름이 없으므로 src_kr_ipc 전개에서 다시 만든다(값 동일 순서: sorted unique)
_scs = None
def subclass_index():
    global _scs
    if _scs is None:
        x = pd.read_parquet(R3 / 'cache' / 'src_kr_ipc.parquet', columns=['ipc_all'])
        e = x.ipc_all.fillna('').str.split('|').explode().str.strip().str[:4]
        _scs = pd.Index(sorted(e[e.str.len() == 4].unique()))
        assert len(_scs) == X['nsc'], (len(_scs), X['nsc'])
    return _scs


def alloc(c):
    """과제분류 cls → KSIC k 배분 비중(성과 Year ≤ c−2). 재료 < 1000건이면 None"""
    den = rp_cls[rp_cls.yr <= c - 2].groupby('cls').size()
    if den.sum() < 1000:
        return None
    t = lk_cls[lk_cls.yr <= c - 2].assign(m=1.0)
    num = t.groupby(['cls', 'ksic']).m.sum().unstack(fill_value=0.0).reindex(index=den.index, fill_value=0.0)
    num = num.reindex(columns=K, fill_value=0.0)
    Sk = num.values / den.values[:, None]
    sbar = num.values.sum(0) / max(den.values.sum(), 1)
    return den.index, Sk, sbar


def lift_of(Mc, A):
    idx, Sk, sbar = A
    has = Mc.sum(1) > 0
    L = np.log(((Mc / np.maximum(Mc.sum(1, keepdims=True), 1e-9)) @ Sk + 1e-4) / (sbar + 1e-4))
    L[~has] = 0.0
    return L


for c in ORIG:
    outp = R4 / 'cache' / f'ntis_o{c}.parquet'
    if outp.exists():
        print('exists', outp); continue
    te = pd.read_parquet(R3 / 'cache' / f'o{c}' / 'prep_te.parquet', columns=['firm', 'k', 'prior_any'])
    te = te[~te.prior_any.values].reset_index(drop=True); te['firm'] = te.firm.astype(str)
    firms = pd.Index(te.firm.unique()); fi = pd.Series(np.arange(len(firms)), index=firms); nF = len(firms)
    ce = int(f'{c}1231')
    va = (X['ay'] <= c) & (X['vis'] <= ce)                                   # 원점 연말까지 보이는 제공 KR 출원
    # --- 기업 ↔ 보이는 범위안 출원
    fa_f = X['fa_firm'].astype(str); sel = pd.Index(fa_f).isin(firms) & va[X['fa_a']]
    FA = pd.DataFrame({'firm': fa_f[sel], 'a': X['fa_a'][sel]})
    # --- 범위안 정부 기원(레코드 Year ≤ c)
    gi = gin[(gin.yr <= c) & va[gin.a.values]]
    gi = FA.merge(gi, on='a')
    # --- 범위밖 정부 기원(등록 ≤ c 연말, ay ≤ c, 레코드 Year ≤ c)
    go = gout[(gout.yr <= c) & (gout.ay <= c) & (gout.rd <= ce) & gout.applicant_cd.isin(firms)]
    # 특허 키
    gi = gi.assign(pk='i' + gi.a.astype(str)); go = go.assign(pk='o' + go.rgst_no.astype(str), firm=go.applicant_cd)
    G = pd.concat([gi[['firm', 'pk', 'grp', 'cls']], go[['firm', 'pk', 'grp', 'cls']]], ignore_index=True)
    gp = G.drop_duplicates(['firm', 'pk'])
    n_g = gp.groupby('firm').size().reindex(firms).fillna(0).values
    n_self = G[G.grp.isin(['big', 'sme'])].groupby('firm').pk.nunique().reindex(firms).fillna(0).values
    n_coop = G[G.grp.isin(['univ', 'gri'])].groupby('firm').pk.nunique().reindex(firms).fillna(0).values
    # 전체 보이는 특허 수(비율 분모)
    n_in = FA.groupby('firm').size().reindex(firms).fillna(0).values
    ot = oos_tot[(oos_tot.ay <= c) & (oos_tot.rd <= ce) & oos_tot.applicant_cd.isin(firms)]
    n_out = ot.groupby('applicant_cd').size().reindex(firms).fillna(0).values
    # --- KSIC IPC 서브클래스 프로파일(보이는 출원)과 정부특허 프로파일 코사인
    m = va[X['ea']]
    KP = np.zeros((50, X['nsc'])); np.add.at(KP, (X['k'][X['ea'][m]], X['ej'][m]), X['ew'][m])
    KPn = KP / np.maximum(np.linalg.norm(KP, axis=1, keepdims=True), 1e-12)
    gia = gi.drop_duplicates(['firm', 'a'])
    A1 = sp.csr_matrix((np.ones(len(gia)), (fi[gia.firm].values, gia.a.values)), shape=(nF, nA))
    FPg = (A1 @ E).toarray() if nF * X['nsc'] < 4e8 else None
    assert FPg is not None, '메모리 보호: 기업×서브클래스 행렬이 너무 큼'
    scs = subclass_index(); scix = pd.Series(np.arange(len(scs)), index=scs)
    goo = go.drop_duplicates(['firm', 'pk'])
    rr, cc_, ww = [], [], []
    for f_, s_ in zip(goo.firm.values, goo.ipc_codes.values):
        L_ = [x for x in sc_list(s_) if x in scix.index]
        for x in L_:
            rr.append(fi[f_]); cc_.append(scix[x]); ww.append(1.0 / len(L_))
    if rr:
        np.add.at(FPg, (np.array(rr), np.array(cc_)), np.array(ww))
    ng = np.linalg.norm(FPg, axis=1)
    COSg = (FPg / np.maximum(ng[:, None], 1e-12)) @ KPn.T
    del FPg
    # --- 과제분류→KSIC 리프트(정부 기원 특허의 과제분류 구성)
    Aal = alloc(c)
    if Aal is None:
        LIFTg = np.full((nF, 50), np.nan)
    else:
        idx_, _, _ = Aal; cix = pd.Series(np.arange(len(idx_)), index=idx_)
        gcl = G[G.cls.isin(idx_)]
        Mc = np.zeros((nF, len(idx_))); np.add.at(Mc, (fi[gcl.firm].values, cix[gcl.cls].values), 1.0)
        LIFTg = lift_of(Mc, Aal)
    # --- 과제 수행 이력(사업자번호, 과제연도 ≤ c−1)
    if c - 1 < 2010:
        p_any = np.full(nF, np.nan); p_lnf = np.full(nF, np.nan); p_n5 = np.full(nF, np.nan); LIFTp = np.full((nF, 50), np.nan)
    else:
        q = pj[(pj.year <= c - 1) & pj.applicantCd.isin(firms)]
        q5 = q[q.year >= c - 5]
        p_any = (q.groupby('applicantCd').size().reindex(firms).fillna(0).values > 0).astype(float)
        p_lnf = np.log1p(q5.groupby('applicantCd').gov_funds.sum().reindex(firms).fillna(0).values / 1e6)   # 최근 5년 정부연구비(백만원)
        p_n5 = np.log1p(q5.groupby('applicantCd').pjt_no.nunique().reindex(firms).fillna(0).values)
        if Aal is None:
            LIFTp = np.full((nF, 50), np.nan)
        else:
            idx_, _, _ = Aal; cix = pd.Series(np.arange(len(idx_)), index=idx_)
            qc = q[q.cls.isin(idx_)]
            Mp = np.zeros((nF, len(idx_))); np.add.at(Mp, (fi[qc.applicantCd].values, cix[qc.cls].values), 1.0)
            LIFTp = lift_of(Mp, Aal)
    f = fi[te.firm].values; k = te.k.values.astype(int)
    out = pd.DataFrame({'firm': te.firm.values, 'k': k})
    out['g_any'] = (n_g[f] > 0).astype(np.float32)
    out['g_ln'] = np.log1p(n_g[f]).astype(np.float32)
    out['g_share'] = (n_g[f] / np.maximum(n_in[f] + n_out[f], 1)).astype(np.float32)
    out['g_self'] = (n_self[f] > 0).astype(np.float32)
    out['g_coop'] = (n_coop[f] > 0).astype(np.float32)
    out['g_tgt_ipc'] = np.where(ng[f] > 0, COSg[f, k], 0.0).astype(np.float32)
    out['g_tgt_cls'] = LIFTg[f, k].astype(np.float32)
    out['p_any'] = p_any[f].astype(np.float32); out['p_lnf'] = p_lnf[f].astype(np.float32); out['p_n5'] = p_n5[f].astype(np.float32)
    out['p_tgt_cls'] = LIFTp[f, k].astype(np.float32)
    out.to_parquet(outp, index=False)
    st = {x: round(float(np.nanmean(out[x].values)), 4) for x in out.columns[2:]}
    print(c, 'rows', len(out), 'firms', nF, 'g_any 기업', int((n_g > 0).sum()), 'p_any 기업', int(np.nansum(p_any > 0)), st,
          'peak', peak(), round(time.time() - T0), flush=True)
