# T1 R3 B단계(재시도, 2026-09-29 오후) 추가 결합 — r3ext2 위에 더해 쓴다. 모든 피처는 각 원점(cut) 연말 시점 정보만.
#  oaco  : OpenAlex 기술셀 '기업 소속' 논문 변형(_shared/ext/r3oa_mg_t0_features.parquet, 재시도 수집분: company 523 + ALL 558 메인그룹)
#          → 목표 KSIC 메인그룹 구성(원점 연말까지 공개·등록된 KR 출원 전체 IPC, E20 방식) 가중 기업논문 점유 성장·기업 비중,
#            기업 메인그룹 × 목표 KSIC 메인그룹 겹침 위 기업논문 성장, 기업 자신 기술 대비 격차.
#          시점: 피처표의 t0 = cut−1 행(발행연도 ≤ cut−1 논문만 — 원점 연도 논문은 색인 지연으로 쓰지 않음). OpenAlex 스냅샷이라 준-정합.
#  ipcm  : 기업–목표 KSIC 기술 적합도(E20)를 메인그룹 해상도로(원점 연말까지 공개·등록된 KR 출원 전체 IPC 메인그룹, 분할 가중) 코사인·순위.
#  entfit: '최근 진입자 닮음' — 목표 KSIC k에 (cut−4, cut] 동안 첫 공개·등록 출원을 낸 법인들의 k 밖 기술 프로파일(서브클래스) 평균과
#          기업 프로파일의 코사인·순위. 진입 판정·프로파일 모두 원점 연말까지 공개·등록된 KR 출원만 쓴다.
import sys, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
import numpy as np, pandas as pd, scipy.sparse as sp
import r3ext2
from r3ext2 import _put, ipcx_src, ipcx_src_mg, mg_profiles_pub, EXTD
EXT = {}


# ---------------- OpenAlex 기업 소속 논문(메인그룹) ----------------
_OC = {}
def oc_table():
    """r3oa_work/r3oa_11_features.py와 같은 식을 모든 t0(1999~)로 확장: 발행연도 ≤ t0 논문만. 반환 {t0: DataFrame(mg → oa_co_g3, oa_co_share)}"""
    if _OC: return _OC
    A = pd.read_parquet(EXTD/'r3oa_mg_year_ALL.parquet').pivot_table(index='mg', columns='year', values='n_works', aggfunc='first')
    C = pd.read_parquet(EXTD/'r3oa_mg_year_company.parquet').pivot_table(index='mg', columns='year', values='n_works', aggfunc='first')
    E = pd.read_parquet(EXTD/'science_openalex_ksic_extra_year.parquet')
    tc = E[(E.ksic == 'TOTAL') & (E.variant == 'company_ALL')].set_index('year').n_works.sort_index()
    win = lambda P, t0, lo, hi: P[[y for y in range(t0 + lo, t0 + hi + 1)]].sum(1)
    for t0 in range(max(A.columns.min(), C.columns.min(), tc.index.min()) + 5, 2022):
        a3 = win(A, t0, -2, 0); c3 = win(C, t0, -2, 0); c0 = win(C, t0, -5, -3)
        _OC[t0] = pd.DataFrame({'oa_co_g3': (np.log((c3 + 1) / tc.loc[t0 - 2:t0].sum()) - np.log((c0 + 1) / tc.loc[t0 - 5:t0 - 3].sum())).values,
                                'oa_co_share': (np.log(c3 + 1) - np.log(a3.reindex(C.index) + 1)).values}, index=C.index)
    return _OC


def ext_oaco(tr, te, c, final=False):
    F = oc_table()
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]
            assert (cut - 1) in F, f'r3oa 기업논문 t0={cut - 1} 계산 불가'
            t = F[cut - 1].dropna(subset=['oa_co_g3', 'oa_co_share'])      # 🔒 발행연도 ≤ cut−1
            P = mg_profiles_pub(cut)
            cov = P['mgs'].isin(t.index); cj = np.nonzero(cov)[0]
            g = t.oa_co_g3.reindex(P['mgs'][cj]).values; sh = t.oa_co_share.reindex(P['mgs'][cj]).values
            KPc = P['KP'][:, cj]; kc = KPc.sum(1)
            k_g = np.where(kc > 0, (KPc @ g) / np.maximum(kc, 1e-12), np.nan)
            k_s = np.where(kc > 0, (KPc @ sh) / np.maximum(kc, 1e-12), np.nan)
            FPc = P['FP'][:, cj]
            den = np.asarray(FPc @ KPc.T); num = np.asarray(FPc @ (KPc * g[None, :]).T)
            fc = np.asarray(FPc.sum(1)).ravel(); f_g = np.where(fc > 0, np.asarray(FPc @ g).ravel() / np.maximum(fc, 1e-12), np.nan)
            f = pd.Series(d.firm.values[idx]).map(P['fi']).fillna(-1).astype(int).values; ok = f >= 0; fz = np.where(ok, f, 0)
            dd = np.where(ok, den[fz, k], 0.0); nn = np.where(ok, num[fz, k], 0.0)
            _put(d, 'oc_k_g', k_g[k], idx); _put(d, 'oc_k_sh', k_s[k], idx)
            _put(d, 'oc_fit_g', np.where(dd > 0, nn / np.maximum(dd, 1e-12), np.nan), idx)
            _put(d, 'oc_gap', k_g[k] - np.where(ok, f_g[fz], np.nan), idx)
    return tr, te
OC_COLS = ['oc_k_g', 'oc_k_sh', 'oc_fit_g', 'oc_gap']
EXT['oaco'] = ext_oaco


# ---------------- 기술 적합도(메인그룹 해상도) ----------------
def ext_ipcm(tr, te, c, final=False):
    X = ipcx_src(); Y = ipcx_src_mg(); nM = len(Y['mgs']); nA = len(X['ay'])
    E = sp.csr_matrix((Y['ew'], (Y['ea'], Y['ej'])), shape=(nA, nM))
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = int(f'{cut}1231')
            va = (X['ay'] <= cut) & (X['vis'] <= ce); m = va[Y['ea']]
            KP = np.zeros((50, nM), np.float64); np.add.at(KP, (X['k'][Y['ea'][m]], Y['ej'][m]), Y['ew'][m])
            KP = KP / np.maximum(np.linalg.norm(KP, axis=1, keepdims=True), 1e-12)
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            sel = pd.Index(X['fa_firm']).isin(firms) & va[X['fa_a']]
            A = sp.csr_matrix((np.ones(int(sel.sum())), (fi[X['fa_firm'][sel]].values, X['fa_a'][sel])), shape=(len(firms), nA))
            FP = (A @ E).tocsr(); nrm = np.sqrt(np.asarray(FP.multiply(FP).sum(1)).ravel())
            C = np.asarray(FP @ KP.T) / np.maximum(nrm[:, None], 1e-12)
            f = fi[d.firm.values[idx]].values; has = nrm[f] > 0
            _put(d, 'ipcm_fit', np.where(has, C[f, k], np.nan), idx)
            rk = (-C).argsort(1).argsort(1); _put(d, 'ipcm_fit_rank', np.where(has, rk[f, k], np.nan), idx)
    return tr, te
IPCM_COLS = ['ipcm_fit', 'ipcm_fit_rank']
EXT['ipcm'] = ext_ipcm


# ---------------- 최근 진입자 닮음 ----------------
def ext_entfit(tr, te, c, final=False, win=4):
    X = ipcx_src(); nA = len(X['ay'])
    E = sp.csr_matrix((X['ew'], (X['ea'], X['ej'])), shape=(nA, X['nsc']))
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = int(f'{cut}1231')
            va = (X['ay'] <= cut) & (X['vis'] <= ce)
            sel = va[X['fa_a']]; ffn = X['fa_firm'][sel]; aa = X['fa_a'][sel]
            firms = pd.Index(pd.unique(ffn)); fi = pd.Series(np.arange(len(firms)), index=firms); fx = fi[ffn].values
            A = sp.csr_matrix((np.ones(len(aa)), (fx, aa)), shape=(len(firms), nA))
            FP = (A @ E).tocsr()
            # 법인×KSIC 첫 공개·등록 출원연도(원점 연말 가시분) → (cut−win, cut] 진입자
            kk = X['k'][aa].astype(np.int64); yy = X['ay'][aa]
            key = fx.astype(np.int64) * 64 + kk
            fy = pd.Series(yy).groupby(key).min()
            ent = fy[(fy.values > cut - win)].index.values                       # (법인, KSIC) 진입 쌍
            er = pd.Series(np.arange(len(ent)), index=ent)
            m2 = pd.Index(key).isin(ent)
            A2 = sp.csr_matrix((np.ones(int(m2.sum())), (er[key[m2]].values, aa[m2])), shape=(len(ent), nA))
            V = (FP[ent // 64] - (A2 @ E)).tocsr()                               # 진입자의 k 밖 프로파일
            V.data = np.maximum(V.data, 0); V.eliminate_zeros()
            vn = np.sqrt(np.asarray(V.multiply(V).sum(1)).ravel()); okv = vn > 0
            V = sp.diags(np.where(okv, 1 / np.maximum(vn, 1e-12), 0)) @ V
            IK = sp.csr_matrix((np.ones(int(okv.sum())), ((ent % 64)[okv], np.nonzero(okv)[0])), shape=(50, len(ent)))
            EK = np.asarray((IK @ V).todense()); nk = np.linalg.norm(EK, axis=1); EK = EK / np.maximum(nk[:, None], 1e-12)
            # 위험집합 기업 프로파일
            rf = pd.Series(d.firm.values[idx]).map(fi).fillna(-1).astype(int).values
            ur = np.unique(rf[rf >= 0]); ui = pd.Series(np.arange(len(ur)), index=ur)
            FR = FP[ur]; fn = np.sqrt(np.asarray(FR.multiply(FR).sum(1)).ravel())
            C = np.asarray(FR @ EK.T) / np.maximum(fn[:, None], 1e-12)
            C[:, nk == 0] = np.nan
            ok = rf >= 0; j = np.where(ok, ui.reindex(np.where(ok, rf, ur[0] if len(ur) else 0)).values, 0)
            has = ok & (fn[j] > 0)
            _put(d, 'ent_fit', np.where(has, C[j, k], np.nan), idx)
            rk = (-np.nan_to_num(C, nan=-9)).argsort(1).argsort(1)
            _put(d, 'ent_fit_rank', np.where(has, rk[j, k], np.nan), idx)
    return tr, te
ENT_COLS = ['ent_fit', 'ent_fit_rank']
EXT['entfit'] = ext_entfit


# ---------------- 기술 이웃 기업의 보유(협업 필터링) ----------------
#  peer: 기업 서브클래스 프로파일(원점 연말까지 공개·등록된 KR 출원 전체 IPC) 코사인 상위 N개 이웃 법인(자기 제외) 중
#        목표 KSIC에 이미 공개·등록 출원을 가진 비율(유사도 가중)과 그 순위. '나와 기술이 닮은 기업들이 이미 가 있는 산업'.
def ext_peer(tr, te, c, final=False, N=50, chunk=1500, mode='hold', win=4):
    X = ipcx_src(); nA = len(X['ay'])
    E = sp.csr_matrix((X['ew'], (X['ea'], X['ej'])), shape=(nA, X['nsc']))
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; ce = int(f'{cut}1231')
            va = (X['ay'] <= cut) & (X['vis'] <= ce)
            sel = va[X['fa_a']]; ffn = X['fa_firm'][sel]; aa = X['fa_a'][sel]
            firms = pd.Index(pd.unique(ffn)); fi = pd.Series(np.arange(len(firms)), index=firms); fx = fi[ffn].values
            A = sp.csr_matrix((np.ones(len(aa)), (fx, aa)), shape=(len(firms), nA))
            FP = (A @ E).tocsr(); nr = np.sqrt(np.asarray(FP.multiply(FP).sum(1)).ravel())
            FN = (sp.diags(1 / np.maximum(nr, 1e-12)) @ FP).astype(np.float32).toarray()          # 법인 × 서브클래스(행 L2 정규화)
            H = np.zeros((len(firms), 50), np.float32); H[fx, X['k'][aa]] = 1.0                   # 원점 가시 KSIC 보유
            if mode == 'ent':                                                                       # (cut−win, cut] 첫 진입만
                key = fx.astype(np.int64) * 64 + X['k'][aa].astype(np.int64); fy = pd.Series(X['ay'][aa]).groupby(key).min()
                e = fy.index.values[fy.values > cut - win]; H = np.zeros((len(firms), 50), np.float32); H[e // 64, e % 64] = 1.0
            rf = pd.Series(d.firm.values[idx]).map(fi).fillna(-1).astype(int).values
            ur = np.unique(rf[rf >= 0]); PK = np.full((len(ur), 50), np.nan, np.float32)
            for s0 in range(0, len(ur), chunk):
                u = ur[s0:s0 + chunk]; S = FN[u] @ FN.T; S[np.arange(len(u)), u] = -1.0              # 자기 제외
                top = np.argpartition(-S, N, axis=1)[:, :N]; sv = np.maximum(np.take_along_axis(S, top, 1), 0)
                PK[s0:s0 + chunk] = np.einsum('ij,ijk->ik', sv, H[top]) / np.maximum(sv.sum(1, keepdims=True), 1e-12)
                del S
            ui = pd.Series(np.arange(len(ur)), index=ur); ok = (rf >= 0)
            j = np.where(ok, ui.reindex(np.where(ok, rf, ur[0] if len(ur) else 0)).values, 0)
            has = ok & (nr[np.where(ok, rf, 0)] > 0)
            nm = 'peer_k' if mode == 'hold' else 'peer_e'
            _put(d, nm, np.where(has, PK[j, k], np.nan), idx)
            rk = (-np.nan_to_num(PK, nan=-9)).argsort(1).argsort(1); _put(d, nm + '_rank', np.where(has, rk[j, k], np.nan), idx)
    return tr, te
PEER_COLS = ['peer_k', 'peer_k_rank']
EXT['peer'] = ext_peer


# peerent: 기술 이웃 상위 50개 법인 중 목표 KSIC에 (cut−4, cut] 동안 첫 공개·등록 출원을 낸(최근 진입한) 비율(유사도 가중)·순위
PEERE_COLS = ['peer_e', 'peer_e_rank']
EXT['peerent'] = lambda tr, te, c, final=False: ext_peer(tr, te, c, mode='ent')
