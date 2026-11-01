# T1 R3 B단계 외부데이터 결합(3라운드 새 수집분) — r2ext.EXT·r3ext.EXT에 더해 쓴다. 모든 피처는 각 원점(cut) 연말 시점 정보만.
#  oa   : OpenAlex 기술셀(IPC 메인그룹 460개) 전 세계 논문 수(_shared/ext/r3oa_mg_year_ALL.parquet)
#         → 목표 KSIC의 메인그룹 구성(기준시점 가시 범위안 등록) 가중 과학 모멘텀, 그리고 '기업 IPC 메인그룹 × 목표 KSIC 메인그룹' 겹침 위의 과학 모멘텀.
#         가설: 기업이 이미 가진 기술(메인그룹)이 목표 산업과 겹치고 그 겹친 기술의 과학이 가속하면 그 산업에 첫 진입한다.
#         시점: publication_year ≤ cut−1 (원점 연도 논문은 색인 지연으로 쓰지 않음), 전 세계 총논문 성장 차감(점유 성장). OpenAlex 스냅샷이라 준-정합.
#  ros  : Reliance on Science(특허→논문 인용, USPTO) × 제공 데이터 family US 토큰(_shared/ext/r3ros_*)
#         → 목표 KSIC의 과학 의존도(최근 10년 발행 연결 US 특허 중 논문 인용 비율), 기업 자신의 과학 기반(자사 KR 출원의 US 패밀리 중 논문 인용),
#           기업 기술 × 목표 KSIC 셀 과학 의존도 겹침, 보유산업 대비 격차. 시점: US 발행일 ≤ cut 연말, KR 출원은 cut 연말까지 공개·등록된 것.
#  kitem: KOSIS 광업제조업조사 품목편(_shared/ext/r3kosis_ksic50_items_annual·chain) → 목표 KSIC 품목 수·품목 HHI·연쇄 생산 성장·재고/출하
#         시점: 자료연도 Y ≤ cut−2(최종변경 Y+2년 1–2월 → 보수 규칙).
#  nts  : 국세청 휴·폐업(r3nts_applicant_status) — 모형 피처가 아니라 채점 단계 정제(폐업일 ≤ 원점 연말인 법인의 쌍 점수를 최저로). r3nts.py에서 사용.
import sys, pathlib
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; sys.path.insert(0, str(R3)); sys.path.insert(0, str(T1/'R2'))
import numpy as np, pandas as pd, scipy.sparse as sp
from r2 import lib, B, CACHE
import r2ext
EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
EXT = {}
K50 = r2ext.K50


def _put(d, name, vals, idx):
    if name not in d.columns: d[name] = np.float32(np.nan)
    d.loc[d.index[idx], name] = np.asarray(vals, dtype=np.float32)


def _mg(s):
    return s.fillna('').astype(str).str.split('/').str[0].str.strip()


_PROF = {}
def mg_profiles(cut):
    """기준시점 cut 연말까지 가시(등록)된 범위안 특허의 메인그룹 프로파일: KSIC×MG 점유(행 정규화, 전체 MG 기준), 기업×MG 점유(희소)"""
    if cut in _PROF: return _PROF[cut]
    ins = lib._cache['ins']; K = K50(); kidx = {k: i for i, k in enumerate(K)}
    v = ins[(ins.ay <= cut) & (ins.rgst_date <= f'{cut}1231')].drop_duplicates(['applicant_cd', 'rgst_no'])
    v = v.assign(mg=_mg(v.mipc)); v = v[v.mg.str.len() >= 6]
    mgs = pd.Index(v.mg.unique()); mi = pd.Series(np.arange(len(mgs)), index=mgs)
    u = v.drop_duplicates('rgst_no')
    KP = np.zeros((50, len(mgs)), np.float64); np.add.at(KP, (u.ksic.map(kidx).values, mi[u.mg].values), 1)
    KP = KP / np.maximum(KP.sum(1, keepdims=True), 1e-9)
    firms = pd.Index(v.applicant_cd.unique()); fi = pd.Series(np.arange(len(firms)), index=firms)
    FP = sp.csr_matrix((np.ones(len(v)), (fi[v.applicant_cd].values, mi[v.mg].values)), shape=(len(firms), len(mgs)))
    FP.sum_duplicates(); tot = np.asarray(FP.sum(1)).ravel()
    FP = sp.diags(1 / np.maximum(tot, 1e-9)) @ FP
    # KSIC 50 내부 믹스(기업 보유산업 격차용)
    _PROF[cut] = dict(mgs=mgs, KP=KP, fi=fi, FP=FP.tocsr())
    if len(_PROF) > 3: _PROF.pop(next(iter(_PROF)))
    return _PROF[cut]


def _pair_idx(P, d, idx):
    """d의 행(idx)에 대응하는 기업 행 번호(없으면 -1)"""
    f = pd.Series(d.firm.values[idx]).map(P['fi']).fillna(-1).astype(int).values
    return f


# ---------------- OpenAlex 기술셀 과학 ----------------
_OA = {}
def oa_table():
    if _OA: return _OA
    w = pd.read_parquet(EXTD/'r3oa_mg_year_ALL.parquet', columns=['mg', 'year', 'n_works'])
    W = w.pivot_table(index='year', columns='mg', values='n_works', aggfunc='sum').fillna(0)
    t = pd.read_parquet(EXTD/'science_openalex_country_total_year.parquet'); T = t[t.country == 'ALL'].set_index('year').n_works_total.sort_index()
    _OA.update(W=W, T=T); return _OA


def oa_growth(cut):
    O = oa_table(); W, T = O['W'], O['T']
    r = W.loc[cut - 3:cut - 1].sum(0); b = W.loc[cut - 6:cut - 4].sum(0)       # 🔒 연도 ≤ cut−1
    rt = T.loc[cut - 3:cut - 1].sum(); bt = T.loc[cut - 6:cut - 4].sum()
    g = (np.log1p(r) - np.log1p(b)) - (np.log(rt) - np.log(bt))
    return g, np.log1p(r)


def ext_oa(tr, te, c, final=False, prof=None):
    prof = prof or mg_profiles
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]
            P = prof(cut); g, lv = oa_growth(cut)
            cov = P['mgs'].isin(g.index); cj = np.nonzero(cov)[0]
            gg = g.reindex(P['mgs'][cj]).values; ll = lv.reindex(P['mgs'][cj]).values
            KPc = P['KP'][:, cj]; kc = KPc.sum(1)
            k_g = np.where(kc > 0, (KPc @ gg) / np.maximum(kc, 1e-12), np.nan); k_l = np.where(kc > 0, (KPc @ ll) / np.maximum(kc, 1e-12), np.nan)
            FPc = P['FP'][:, cj]
            den = np.asarray((FPc @ KPc.T)); num = np.asarray(FPc @ (KPc * gg[None, :]).T)
            fc = np.asarray(FPc.sum(1)).ravel(); f_g = np.where(fc > 0, np.asarray(FPc @ gg).ravel() / np.maximum(fc, 1e-12), np.nan)
            f = _pair_idx(P, d, idx); ok = f >= 0; fz = np.where(ok, f, 0)
            dd = np.where(ok, den[fz, k], 0.0); nn = np.where(ok, num[fz, k], 0.0)
            _put(d, 'oa_k_g', k_g[k], idx); _put(d, 'oa_k_cov', kc[k], idx); _put(d, 'oa_k_lvl', k_l[k], idx)
            _put(d, 'oa_fit_g', np.where(dd > 0, nn / np.maximum(dd, 1e-12), np.nan), idx)
            _put(d, 'oa_fit_w', dd * 1e3, idx)
            _put(d, 'oa_gap', k_g[k] - np.where(ok, f_g[fz], np.nan), idx)
    return tr, te
OA_COLS = ['oa_k_g', 'oa_k_cov', 'oa_k_lvl', 'oa_fit_g', 'oa_fit_w', 'oa_gap']
EXT['oa'] = ext_oa


# ---------------- Reliance on Science ----------------
_ROS = {}
def ros_src():
    """KR 출원번호 ↔ US 패밀리(발행일·논문 인용 수) 표: R3/cache/src_ros_kr.parquet(1회 생성)"""
    if 'kr' in _ROS: return _ROS
    p = R3/'cache'/'src_ros_kr.parquet'
    if not p.exists():
        pr = pd.read_parquet(EXTD/'r3ros_family_us_pairs.parquet', filters=[('cc', '=', 'KR')], columns=['appl_no', 'us_pn'])
        ps = pd.read_parquet(EXTD/'r3ros_patent_sci.parquet', columns=['us_pn', 'issue_date', 'n_cit'])
        x = pr.drop_duplicates().merge(ps, on='us_pn', how='inner')
        x['issue'] = pd.to_datetime(x.issue_date).dt.strftime('%Y%m%d')
        x[['appl_no', 'us_pn', 'issue', 'n_cit']].to_parquet(p, index=False)
    _ROS['kr'] = pd.read_parquet(p)
    c = pd.read_parquet(EXTD/'r3ros_cell_issueyear.parquet', columns=['ksic', 'mg', 'issue_year', 'first_rec_year', 'n_us_pat', 'n_us_pat_sci', 'n_paper_cit'])
    c['ksic'] = c.ksic.astype(str); _ROS['cell'] = c
    return _ROS


def ext_ros(tr, te, c, final=False, a=5.0, prof=None):
    prof = prof or mg_profiles
    R = ros_src(); K = K50(); kidx = {k: i for i, k in enumerate(K)}
    from r2 import app_pub
    A = app_pub()[['applicant_cd', 'appl_no', 'ay', 'pub_date', 'rgst_date']]
    for d in (tr, te):
        mix = r2ext.firm_mix(d)
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = f'{cut}1231'
            # 목표 KSIC·셀 과학 의존도: 발행 cut−9..cut, first_rec_year ≤ cut
            cc = R['cell']; cc = cc[(cc.issue_year >= cut - 9) & (cc.issue_year <= cut) & (cc.first_rec_year <= cut)]
            ks = cc.groupby('ksic')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum().reindex(K).fillna(0)
            k_sh = np.where(ks.n_us_pat.values > 0, ks.n_us_pat_sci.values / np.maximum(ks.n_us_pat.values, 1), np.nan)
            k_pc = np.log1p(ks.n_paper_cit.values / np.maximum(ks.n_us_pat.values, 1))
            # 셀(ksic×mg) 수축 추정 점유 → 기업 메인그룹 × 목표 KSIC 메인그룹 겹침 가중
            P = prof(cut)
            cm = cc.groupby(['ksic', 'mg'])[['n_us_pat', 'n_us_pat_sci']].sum().reset_index()
            cm = cm[cm.mg.isin(P['mgs']) & cm.ksic.isin(K)]
            prior = np.nan_to_num(k_sh, nan=np.nanmean(k_sh))
            ki = cm.ksic.map(kidx).values; mj = P['mgs'].get_indexer(cm.mg)
            SH = np.tile(prior[:, None], (1, len(P['mgs'])))
            SH[ki, mj] = (cm.n_us_pat_sci.values + a * prior[ki]) / (cm.n_us_pat.values + a)
            den = np.asarray(P['FP'] @ P['KP'].T); num = np.asarray(P['FP'] @ (P['KP'] * SH).T)
            f = _pair_idx(P, d, idx); ok = f >= 0; fz = np.where(ok, f, 0)
            dd = np.where(ok, den[fz, k], 0.0); nn = np.where(ok, num[fz, k], 0.0)
            # 기업 자신: cut 연말까지 공개·등록된 자사 KR 출원(ay ≤ cut) 중 US 패밀리 발행 ≤ cut 연말, 그중 논문 인용
            firms = pd.Index(d.firm.values[idx]).unique()
            a_ = A[A.applicant_cd.isin(firms) & (A.ay <= cut) & (((A.pub_date.str.len() == 8) & (A.pub_date <= ce)) | ((A.rgst_date.str.len() == 8) & (A.rgst_date <= ce)))]
            u = R['kr'][R['kr'].issue <= ce]; u = u.assign(sci=(u.n_cit.values > 0).astype(np.int8))
            u = u.groupby('appl_no').agg(nus=('us_pn', 'size'), ncit=('n_cit', 'sum'), sci=('sci', 'max'))
            m = a_[['applicant_cd', 'appl_no']].drop_duplicates().merge(u, left_on='appl_no', right_index=True, how='inner')
            g = m.groupby('applicant_cd').agg(n_us=('appl_no', 'size'), n_sci=('sci', 'sum'), n_cit=('ncit', 'sum'))
            fm = pd.Series(d.firm.values[idx])
            n_us = fm.map(g.n_us).fillna(0).values; n_sci = fm.map(g.n_sci).fillna(0).values; n_cit = fm.map(g.n_cit).fillna(0).values
            fi, Sn, S0 = mix[cut]; ff = fi[d.firm.values[idx]].values
            home = (Sn * np.isfinite(k_sh)[None, :]) @ np.nan_to_num(k_sh) / np.maximum((Sn * np.isfinite(k_sh)[None, :]).sum(1), 1e-9)
            _put(d, 'ros_k_sh', k_sh[k], idx); _put(d, 'ros_k_pc', k_pc[k], idx)
            _put(d, 'ros_fit', np.where(dd > 0, nn / np.maximum(dd, 1e-12), np.nan), idx)
            _put(d, 'ros_f_us', np.log1p(n_us), idx); _put(d, 'ros_f_sci', np.log1p(n_sci), idx)
            _put(d, 'ros_f_sh', np.where(n_us > 0, n_sci / np.maximum(n_us, 1), np.nan), idx)
            _put(d, 'ros_f_pc', np.where(n_us > 0, np.log1p(n_cit / np.maximum(n_us, 1)), np.nan), idx)
            _put(d, 'ros_gap', k_sh[k] - home[ff], idx)
    return tr, te
ROS_COLS = ['ros_k_sh', 'ros_k_pc', 'ros_fit', 'ros_f_us', 'ros_f_sci', 'ros_f_sh', 'ros_f_pc', 'ros_gap']
ROS_K_COLS = ['ros_k_sh', 'ros_k_pc', 'ros_fit', 'ros_gap']
ROS_F_COLS = ['ros_f_us', 'ros_f_sci', 'ros_f_sh', 'ros_f_pc']
EXT['ros'] = ext_ros


# ---------------- KOSIS 품목편 ----------------
def ext_kitem(tr, te, c, final=False):
    K = K50()
    a = pd.read_parquet(EXTD/'r3kosis_ksic50_items_annual.parquet'); a['ksic'] = a.ksic.astype(str)
    ch = pd.read_parquet(EXTD/'r3kosis_ksic50_items_chain.parquet'); ch['ksic'] = ch.ksic.astype(str)
    ch['g_prod'] = ch.g_prod.clip(-1, 1)   # 연쇄 이상치(|g|>1, 짝 품목 1–2개) 윈저라이즈
    NI = a.pivot_table(index='year', columns='ksic', values='n_items_pos', aggfunc='first').reindex(columns=K)
    HH = a.pivot_table(index='year', columns='ksic', values='item_hhi', aggfunc='first').reindex(columns=K)
    GP = ch.pivot_table(index='year', columns='ksic', values='g_prod', aggfunc='first').reindex(columns=K)
    IS = a.pivot_table(index='year', columns='ksic', values='inv_ship', aggfunc='first').reindex(columns=K)
    def row(M, y): return M.loc[y].values.astype(float) if y in M.index else np.full(50, np.nan)
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; y = cut - 2     # 🔒 Y ≤ cut−2
            g3 = np.nansum(np.vstack([row(GP, yy) for yy in (y - 2, y - 1, y)]), 0)
            g3 = np.where(np.isfinite(row(GP, y)), g3, np.nan)
            _put(d, 'ki_n', np.log1p(row(NI, y))[k], idx); _put(d, 'ki_dn', (np.log1p(row(NI, y)) - np.log1p(row(NI, y - 3)))[k], idx)
            _put(d, 'ki_hhi', row(HH, y)[k], idx); _put(d, 'ki_g3', g3[k], idx); _put(d, 'ki_invs', row(IS, y)[k], idx)
    return tr, te
KI_COLS = ['ki_n', 'ki_dn', 'ki_hhi', 'ki_g3', 'ki_invs']
EXT['kitem'] = ext_kitem


# ---------------- (제공 KR 공개·등록 출원 × 전체 IPC) 기업–목표 KSIC 기술 적합도 확장 ----------------
#  R2 E37 ipc_fit은 '등록된 범위안 특허의 메인 IPC 서브클래스'만 썼다(위험집합 기업의 상당수는 등록 특허가 적어 결측).
#  여기서는 기준시점 연말까지 '공개 또는 등록'된 자사 KR 출원(ay ≤ cut)의 전체 IPC(ipc_all, 출원당 1로 분할 가중)로 기업·KSIC 프로파일을 만든다.
_IPX = {}
def ipcx_src():
    if _IPX: return _IPX
    npz = R3/'cache'/'src_kr_ipc_sc.npz'          # 전개 결과 캐시(값 동일, 프로세스별 문자열 전개 메모리 절약)
    if npz.exists():
        z = np.load(npz, allow_pickle=True); _IPX.update({k: z[k] for k in z.files}); _IPX['nsc'] = int(_IPX['nsc']); return _IPX
    x = pd.read_parquet(R3/'cache'/'src_kr_ipc.parquet', columns=['appl_no', 'ksic', 'ay', 'pub_date', 'rgst_date', 'ipc_all'])
    K = K50(); kidx = {k: i for i, k in enumerate(K)}
    x = x[x.ksic.astype(str).isin(K)].reset_index(drop=True)
    pv = x.pub_date.where(x.pub_date.str.len() == 8, '99999999'); rv = x.rgst_date.where(x.rgst_date.str.len() == 8, '99999999')
    x['vis'] = np.minimum(pv.values.astype(np.int64), rv.values.astype(np.int64))
    e = x[['appl_no', 'ipc_all']].assign(sc=x.ipc_all.fillna('').str.split('|')).explode('sc')
    e['sc'] = e.sc.str.strip().str[:4]; e = e[e.sc.str.len() == 4].drop_duplicates(['appl_no', 'sc'])
    scs = pd.Index(sorted(e.sc.unique())); e['j'] = scs.get_indexer(e.sc).astype(np.int32)
    e['w'] = (1.0 / e.groupby('appl_no').sc.transform('size')).astype(np.float32)
    ai = pd.Series(np.arange(len(x)), index=x.appl_no.values)
    e['a'] = ai[e.appl_no.values].values.astype(np.int32)
    from r2 import app_pub
    fa = app_pub()[['applicant_cd', 'appl_no']].drop_duplicates()
    fa = fa[fa.appl_no.isin(ai.index)]; fa['a'] = ai[fa.appl_no.values].values.astype(np.int32)
    _IPX.update(ay=x.ay.values.astype(np.int16), vis=x.vis.values, k=x.ksic.astype(str).map(kidx).values.astype(np.int16),
                ea=e.a.values, ej=e.j.values, ew=e.w.values, nsc=len(scs), fa_firm=fa.applicant_cd.values.astype(str), fa_a=fa.a.values)
    np.savez(npz, **_IPX)
    return _IPX


def ext_ipcx(tr, te, c, final=False):
    X = ipcx_src()
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = int(f'{cut}1231')
            va = (X['ay'] <= cut) & (X['vis'] <= ce)                  # 원점 연말까지 공개·등록된 출원
            m = va[X['ea']]
            KP = np.zeros((50, X['nsc']), np.float64); np.add.at(KP, (X['k'][X['ea'][m]], X['ej'][m]), X['ew'][m])
            KP = KP / np.maximum(np.linalg.norm(KP, axis=1, keepdims=True), 1e-12)
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            sel = pd.Index(X['fa_firm']).isin(firms) & va[X['fa_a']]
            ff = fi[X['fa_firm'][sel]].values; aa = X['fa_a'][sel]
            # 기업–출원 → 출원–서브클래스 전개(희소 곱)
            A = sp.csr_matrix((np.ones(len(ff)), (ff, aa)), shape=(len(firms), len(X['ay'])))
            E = sp.csr_matrix((X['ew'], (X['ea'], X['ej'])), shape=(len(X['ay']), X['nsc']))
            FP = (A @ E).tocsr(); nrm = np.sqrt(np.asarray(FP.multiply(FP).sum(1)).ravel())
            C = np.asarray(FP @ KP.T) / np.maximum(nrm[:, None], 1e-12)
            f = fi[d.firm.values[idx]].values; has = nrm[f] > 0
            _put(d, 'ipcx_fit', np.where(has, C[f, k], np.nan), idx)
            rk = (-C).argsort(1).argsort(1); _put(d, 'ipcx_fit_rank', np.where(has, rk[f, k], np.nan), idx)
    return tr, te
IPCX_COLS = ['ipcx_fit', 'ipcx_fit_rank']
EXT['ipcx'] = ext_ipcx


def _ipcx_cos(X, firms, fi, am_k, am_f):
    """am_k: KSIC 프로파일에 쓸 출원 마스크, am_f: 기업 프로파일에 쓸 출원 마스크 → (기업×50 코사인, 기업 노름)"""
    m = am_k[X['ea']]
    KP = np.zeros((50, X['nsc']), np.float64); np.add.at(KP, (X['k'][X['ea'][m]], X['ej'][m]), X['ew'][m])
    KP = KP / np.maximum(np.linalg.norm(KP, axis=1, keepdims=True), 1e-12)
    sel = pd.Index(X['fa_firm']).isin(firms) & am_f[X['fa_a']]
    A = sp.csr_matrix((np.ones(int(sel.sum())), (fi[X['fa_firm'][sel]].values, X['fa_a'][sel])), shape=(len(firms), len(X['ay'])))
    E = sp.csr_matrix((X['ew'], (X['ea'], X['ej'])), shape=(len(X['ay']), X['nsc']))
    FP = (A @ E).tocsr(); nrm = np.sqrt(np.asarray(FP.multiply(FP).sum(1)).ravel())
    return np.asarray(FP @ KP.T) / np.maximum(nrm[:, None], 1e-12), nrm


def ext_ipcxd(tr, te, c, final=False):
    """기술 적합도의 '방향': 기업 최근 3년 출원 프로파일 적합도, 목표 KSIC 최근 3년 프로파일 쪽으로의 적합도 변화"""
    X = ipcx_src()
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = int(f'{cut}1231')
            va = (X['ay'] <= cut) & (X['vis'] <= ce); rec = va & (X['ay'] > cut - 3); old = va & (X['ay'] <= cut - 3)
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms); f = fi[d.firm.values[idx]].values
            Cr, nr = _ipcx_cos(X, firms, fi, va, rec)            # 기업 최근 3년 × KSIC 전체
            Ckr, na = _ipcx_cos(X, firms, fi, rec, va)           # 기업 전체 × KSIC 최근 3년
            Cko, _ = _ipcx_cos(X, firms, fi, old, va)            # 기업 전체 × KSIC 3년 이전
            _put(d, 'ipcx_rfit', np.where(nr[f] > 0, Cr[f, k], np.nan), idx)
            _put(d, 'ipcx_kdfit', np.where(na[f] > 0, Ckr[f, k] - Cko[f, k], np.nan), idx)
    return tr, te
IPCXD_COLS = ['ipcx_rfit', 'ipcx_kdfit']
EXT['ipcxd'] = ext_ipcxd


# ---------------- 공개·등록 출원(전체 IPC) 기반 메인그룹 프로파일 → oa·ros 재시험(적합도 결측 축소) ----------------
_IPXM = {}
def ipcx_src_mg():
    if _IPXM: return _IPXM
    npz = R3/'cache'/'src_kr_ipc_mg.npz'
    if npz.exists():
        z = np.load(npz, allow_pickle=True); _IPXM.update(mgs=pd.Index(z['mgs']), ea=z['ea'], ej=z['ej'], ew=z['ew']); return _IPXM
    X = ipcx_src()
    x = pd.read_parquet(R3/'cache'/'src_kr_ipc.parquet', columns=['appl_no', 'ksic', 'ipc_all'])
    x = x[x.ksic.astype(str).isin(K50())].reset_index(drop=True)
    assert len(x) == len(X['ay'])
    e = x[['appl_no', 'ipc_all']].assign(a=np.arange(len(x)), mg=x.ipc_all.fillna('').str.split('|')).explode('mg')
    e['mg'] = e.mg.fillna('').str.split('/').str[0].str.strip(); e = e[e.mg.str.len() >= 6].drop_duplicates(['a', 'mg'])
    mgs = pd.Index(sorted(e.mg.unique())); e['j'] = mgs.get_indexer(e.mg).astype(np.int32)
    e['w'] = (1.0 / e.groupby('a').mg.transform('size')).astype(np.float32)
    _IPXM.update(mgs=mgs, ea=e.a.values.astype(np.int32), ej=e.j.values, ew=e.w.values)
    np.savez(npz, mgs=np.array(mgs, dtype=object), ea=_IPXM['ea'], ej=_IPXM['ej'], ew=_IPXM['ew'])
    return _IPXM


_PPROF = {}
def mg_profiles_pub(cut):
    """기준시점 연말까지 공개·등록된 KR 출원(ay ≤ cut)의 전체 IPC 메인그룹(분할 가중): KSIC×MG 점유, 기업×MG 점유(희소)"""
    if cut in _PPROF: return _PPROF[cut]
    X = ipcx_src(); Y = ipcx_src_mg(); ce = int(f'{cut}1231')
    va = (X['ay'] <= cut) & (X['vis'] <= ce); m = va[Y['ea']]; nM = len(Y['mgs'])
    KP = np.zeros((50, nM), np.float64); np.add.at(KP, (X['k'][Y['ea'][m]], Y['ej'][m]), Y['ew'][m])
    KP = KP / np.maximum(KP.sum(1, keepdims=True), 1e-12)
    sel = va[X['fa_a']]; firms = pd.Index(pd.unique(X['fa_firm'][sel])); fi = pd.Series(np.arange(len(firms)), index=firms)
    A = sp.csr_matrix((np.ones(int(sel.sum())), (fi[X['fa_firm'][sel]].values, X['fa_a'][sel])), shape=(len(firms), len(X['ay'])))
    E = sp.csr_matrix((Y['ew'], (Y['ea'], Y['ej'])), shape=(len(X['ay']), nM))
    FP = (A @ E).tocsr(); tot = np.asarray(FP.sum(1)).ravel(); FP = (sp.diags(1 / np.maximum(tot, 1e-12)) @ FP).tocsr()
    _PPROF[cut] = dict(mgs=Y['mgs'], KP=KP, fi=fi, FP=FP)
    if len(_PPROF) > 3: _PPROF.pop(next(iter(_PPROF)))
    return _PPROF[cut]


def _rename(tr, te, old, new):
    for d in (tr, te): d.rename(columns=dict(zip(old, new)), inplace=True)
    return tr, te
OA2_COLS = [x + '2' for x in OA_COLS]
ROS2_COLS = [x + '2' for x in ROS_COLS]
EXT['oa2'] = lambda tr, te, c, final=False: _rename(*ext_oa(tr, te, c, prof=mg_profiles_pub), OA_COLS, OA2_COLS)
EXT['ros2'] = lambda tr, te, c, final=False: _rename(*ext_ros(tr, te, c, prof=mg_profiles_pub), ROS_COLS, ROS2_COLS)
