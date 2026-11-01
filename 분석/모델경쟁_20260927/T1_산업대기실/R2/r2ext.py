# T1 R2 외부 산업데이터 결합 — KSIC 라벨을 다리로 쓰는 연계
#  (1) io    : 한국은행 산업연관표(2010년 접속표, 2020 기준 통합소분류 165부문) × IO상품–KSIC 대조표 → 기업의 기존 산업 믹스와 목표 산업의 투입·산출 연결 강도(쌍 수준)
#  (2) trade : 관세청/UN Comtrade HS6 수입(분석/fusion/hs6_import_yearly.csv) × KSIC–HS6 연계표 → 목표 산업의 수입 규모·증가·중국 비중(원점 직전연도, 산업×연도)
#  (3) export: UN Comtrade 한국 HS6 수출(_shared/ext/R2global_kr_export_hs6.csv, 다른 트랙 수집 공유분) → 목표 산업 수출 증가(원점 직전연도)
#  (4) kosis : KOSIS 광업제조업조사 5자리(2008–) 산업 규모 — 개발 원점의 학습 원점(≤2007)에는 as-of 값이 없어 '구조 속성(2008년 값 고정)'으로만 시험
# 시점 규칙: 무역·KOSIS는 기준시점 t의 직전연도(t−1) 값까지만. IO표는 2010년 표(2014년 공표) — 개발 원점(2008–12)에는 구조 정보의 경미한 선견(결과변수 아님), 최종 원점(2017)에는 as-of 유효.
import sys, pathlib, json
R2 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R2))
import numpy as np, pandas as pd
from r2 import lib, CACHE, B, fc_of

EXTD = B/'공모전데이터/외부데이터'
_K = {}
def K50():
    if 'K' not in _K: _K['K'] = lib.load()['K']
    return _K['K']


# ---------------- (1) 산업연관표 ----------------
def io_link():
    p = R2/'cache'/'io_L.npz'
    if p.exists():
        z = np.load(p); return {k: z[k] for k in z.files}
    K = K50()
    x = pd.read_excel(EXTD/'산업연관표/IO코드와 KSIC 및 HS코드 대조표.xlsx', sheet_name=0, header=None, skiprows=2)
    x.columns = ['io', 'name', 'ksic']; x['io'] = x.io.ffill(); x['name'] = x.name.ffill(); x = x.dropna(subset=['ksic'])
    x['io4'] = x.io.astype(int).astype(str).str.zfill(4); x['k5'] = x.ksic.astype(str).str.strip().str[:5]
    cl = pd.read_excel(EXTD/'산업연관표/2010-2015-2020_부문분류표.xlsx', sheet_name=0, header=None, skiprows=3, dtype=str)
    cl = cl.iloc[:, :4]; cl.columns = ['io4', 'nm4', 'io3', 'nm3']; cl['io3'] = cl.io3.ffill(); cl['nm3'] = cl.nm3.ffill(); cl = cl.dropna(subset=['io4'])
    cl['io4'] = cl.io4.str.strip().str.zfill(4); m43 = dict(zip(cl.io4, cl.io3.str.strip().str.zfill(3))); n3 = dict(zip(cl.io3.str.strip().str.zfill(3), cl.nm3))
    x['io3'] = x.io4.map(m43)
    lk = x[x.k5.isin(K)].groupby(['k5', 'io3']).size().rename('n_products').reset_index()
    # 거래표(2010, 생산자가격, 통합소분류)
    xl = EXTD/'산업연관표/2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx'
    z = pd.read_excel(xl, sheet_name='10-A표_총거래표(생산자)', header=None)
    codes_r = z.iloc[6:, 0].astype(str).str.strip(); hdr = z.iloc[4, 2:].astype(str).str.strip()
    rr = [i for i, cd in zip(z.index[6:], codes_r) if cd.isdigit() and len(cd) == 3]
    cc = [j for j, cd in zip(z.columns[2:], hdr) if cd.isdigit() and len(cd) == 3]
    rcodes = [str(z.iat[i, 0]).strip() for i in rr]; ccodes = [str(z.iat[4, j]).strip() for j in cc]
    Z = z.loc[rr, cc].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    assert rcodes == ccodes, '거래표 행·열 코드 불일치'
    # 총산출(열 합계 행 '총투입' 또는 행 합계 '총산출') — 총투입계수표로 X 역산: A = Z / X
    a = pd.read_excel(xl, sheet_name='10-총투입계수(A)', header=None)
    ar = [i for i in a.index[6:] if str(a.iat[i, 0]).strip() in rcodes]; ac = [j for j in a.columns[2:] if str(a.iat[4, j]).strip() in ccodes]
    A = a.loc[ar, ac].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.nanmedian(np.where(A > 1e-6, Z / A, np.nan), axis=0)   # 부문별 총산출 = Z_ij / A_ij
    X = np.where(np.isfinite(X), X, Z.sum(0))
    idx = {cd: i for i, cd in enumerate(rcodes)}
    # KSIC(50) → io3 가중(제품코드 수 비례)
    Wk = np.zeros((50, len(rcodes)))
    for _, r in lk.iterrows():
        if r.io3 in idx: Wk[K.index(r.k5), idx[r.io3]] += r.n_products
    mapped = Wk.sum(1) > 0
    Wk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-12)
    Bsell = Z / np.maximum(X[:, None], 1e-9)            # i 산출 중 j에 판매되는 비중
    L_buy = Wk @ A @ Wk.T        # [j,k]: k 산업 1단위 산출에 필요한 j 산업 투입(k가 j로부터 구매) → j 보유 기업의 하류(전방) 진입
    L_sup = L_buy.T              # [j,k]: j 산업이 k로부터 구매(k가 j에 공급) → j 보유 기업의 상류(후방) 진입
    L_sell = Wk @ Bsell @ Wk.T   # [j,k]: j 산출 중 k로 판매 비중
    S_same = Wk @ Wk.T; np.fill_diagonal(S_same, 0)
    np.savez(p, L_buy=L_buy, L_sup=L_sup, L_sell=L_sell, S_same=S_same, mapped=mapped)
    # 연계표 저장(보고서 '데이터 연계 결과')
    lk['io3_name'] = lk.io3.map(n3); lk['io3_in_table'] = lk.io3.isin(idx)
    lk.to_csv(R2/'ext_link_io.csv', index=False, encoding='utf-8-sig')
    top = []
    for i, k in enumerate(K):
        o = np.argsort(-L_buy[:, i])[:3]; s = np.argsort(-L_sup[:, i])[:3]
        top.append(dict(ksic=k, mapped=bool(mapped[i]), top_input_ksic='|'.join(K[j] for j in o if j != i), top_output_to_ksic='|'.join(K[j] for j in s if j != i)))
    pd.DataFrame(top).to_csv(R2/'ext_link_io_ksic_neighbors.csv', index=False, encoding='utf-8-sig')
    return dict(L_buy=L_buy, L_sup=L_sup, L_sell=L_sell, S_same=S_same, mapped=mapped)


_MV = {}
def firm_mix(d, final=False):
    """(cut, firm)별 산업 믹스 50차원: 기준시점 연말까지 등록(가시)된 범위안 특허 수 + 범위밖 자사특허 제목분류 노출(위험쌍의 t_lsum 역변환). 행 정규화."""
    ins = lib.load()['ins']; K = K50(); kidx = {k: i for i, k in enumerate(K)}
    out = {}
    for cut, g in d.groupby('cut', sort=False):
        v = ins[(ins.ay <= cut) & (ins.rgst_date <= f'{cut}1231')].drop_duplicates(['applicant_cd', 'rgst_no'])
        firms = pd.Index(g.firm.unique()); fi = pd.Series(np.arange(len(firms)), index=firms)
        v = v[v.applicant_cd.isin(firms)]
        S = np.zeros((len(firms), 50), np.float32)
        np.add.at(S, (fi[v.applicant_cd].values, v.ksic.map(kidx).values), 1.0)
        np.add.at(S, (fi[g.firm].values, g.k.values), np.expm1(g.t_lsum.values).astype(np.float32))
        out[cut] = (fi, S / np.maximum(S.sum(1, keepdims=True), 1e-9), S)
    return out


def ext_io(tr, te, c, final=False):
    L = io_link()
    for d in (tr, te):
        mix = firm_mix(d, final)
        cols = {n: np.zeros(len(d), np.float32) for n in ('io_down', 'io_up', 'io_sell', 'io_max', 'io_same')}
        for cut, (fi, Sn, S) in mix.items():
            idx = np.nonzero(d.cut.values == cut)[0]; f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            Sn2 = Sn.copy()
            # 자기 산업(k) 노출은 제외하고 '다른 보유 산업'과의 연결만(대각 누수 방지: 텍스트 노출은 이미 t_* 피처)
            cols['io_down'][idx] = (Sn2 @ L['L_buy'])[f, k] - Sn2[f, k] * L['L_buy'][k, k]
            cols['io_up'][idx] = (Sn2 @ L['L_sup'])[f, k] - Sn2[f, k] * L['L_sup'][k, k]
            cols['io_sell'][idx] = (Sn2 @ L['L_sell'])[f, k] - Sn2[f, k] * L['L_sell'][k, k]
            cols['io_same'][idx] = (Sn2 @ L['S_same'])[f, k]
            held = (S > 0).astype(np.float32); Lmx = np.maximum(L['L_buy'], L['L_sup']); np.fill_diagonal(Lmx, 0)
            # 보유(노출>0) 산업 중 목표 산업과의 최대 연결강도
            mx = np.zeros((len(fi), 50), np.float32)
            for j in range(50):
                mx = np.maximum(mx, held[:, [j]] * Lmx[j][None, :])
            cols['io_max'][idx] = mx[f, k]
        for n, v in cols.items(): d[n] = v
    return tr, te
IO_COLS = ['io_down', 'io_up', 'io_sell', 'io_max', 'io_same']


# ---------------- (2)(3) 무역 ----------------
def trade_table():
    p = R2/'cache'/'trade_ky.parquet'
    if p.exists(): return pd.read_parquet(p)
    K = K50()
    mp = pd.read_csv(B/'분석/fusion/ksic_hs6_map.csv', dtype=str)[['k5', 'hs6']].drop_duplicates()
    mp['w'] = 1.0 / mp.groupby('hs6').k5.transform('nunique')          # 여러 KSIC에 걸친 HS6는 균등 분할
    im = pd.read_csv(B/'분석/fusion/hs6_import_yearly.csv', dtype={'hs6': str}); im['hs6'] = im.hs6.str.zfill(6)
    ex = pd.read_csv(B/'분석/모델경쟁_20260927/_shared/ext/R2global_kr_export_hs6.csv', dtype={'hs6': str}); ex['hs6'] = ex.hs6.str.zfill(6)
    rows = []
    for nm, df, part in (('imp', im[im.partner == 'WLD'], None), ('imp_cn', im[im.partner == 'CN'], None), ('exp', ex, None)):
        m = df.merge(mp, on='hs6'); m['v'] = m.value * m.w
        g = m.groupby(['k5', 'year']).v.sum().rename(nm); rows.append(g)
    t = pd.concat(rows, axis=1).reset_index().rename(columns={'k5': 'ksic'})
    t = t[t.ksic.isin(K)]; t.to_parquet(p, index=False)
    # 연계표
    cov = mp[mp.k5.isin(K)].groupby('k5').hs6.nunique().rename('n_hs6').reindex(K).fillna(0).astype(int).reset_index().rename(columns={'k5': 'ksic'})
    for nm in ('imp', 'imp_cn', 'exp'):
        for yy in (2005, 2010, 2015):
            cov[f'{nm}_{yy}_usd'] = cov.ksic.map(t[t.year == yy].set_index('ksic')[nm])
    hs_im = set(im.hs6); hs_ex = set(ex.hs6); mk = mp[mp.k5.isin(K)]
    cov['hs6_matched_import'] = cov.ksic.map(mk[mk.hs6.isin(hs_im)].groupby('k5').hs6.nunique()).fillna(0).astype(int)
    cov['hs6_matched_export'] = cov.ksic.map(mk[mk.hs6.isin(hs_ex)].groupby('k5').hs6.nunique()).fillna(0).astype(int)
    cov.to_csv(R2/'ext_link_trade.csv', index=False, encoding='utf-8-sig')
    return t


def _ky(cut, t, col, lag):
    s = t[t.year == cut - lag].set_index('ksic')[col]; return s


def ext_trade(tr, te, c, final=False, with_export=False):
    t = trade_table(); K = K50(); ki = {k: i for i, k in enumerate(K)}
    T = {}
    for col in ('imp', 'imp_cn', 'exp'):
        piv = t.pivot(index='year', columns='ksic', values=col).reindex(columns=K)
        T[col] = piv
    def val(col, yr):
        P = T[col]
        return P.loc[yr].values.astype(float) if yr in P.index else np.full(50, np.nan)
    for d in (tr, te):
        for n in (['tr_imp_l', 'tr_imp_g1', 'tr_imp_g3', 'tr_cn_sh', 'tr_cn_g3'] + (['tr_exp_l', 'tr_exp_g3', 'tr_nx'] if with_export else [])):
            d[n] = np.float32(np.nan)
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; y1 = cut - 1   # 🔒 기준시점 직전연도까지
            im1, im2, im4 = val('imp', y1), val('imp', y1 - 1), val('imp', y1 - 3)
            cn1, cn4 = val('imp_cn', y1), val('imp_cn', y1 - 3)
            f = dict(tr_imp_l=np.log1p(im1), tr_imp_g1=np.log1p(im1) - np.log1p(im2), tr_imp_g3=np.log1p(im1) - np.log1p(im4),
                     tr_cn_sh=cn1 / np.maximum(im1, 1), tr_cn_g3=np.log1p(cn1) - np.log1p(cn4))
            if with_export:
                e1, e4 = val('exp', y1), val('exp', y1 - 3)
                f.update(tr_exp_l=np.log1p(e1), tr_exp_g3=np.log1p(e1) - np.log1p(e4), tr_nx=(e1 - im1) / np.maximum(e1 + im1, 1))
            for n, v in f.items(): d.loc[d.index[idx], n] = v[k].astype(np.float32)
    return tr, te
TRADE_COLS = ['tr_imp_l', 'tr_imp_g1', 'tr_imp_g3', 'tr_cn_sh', 'tr_cn_g3']
EXPORT_COLS = ['tr_exp_l', 'tr_exp_g3', 'tr_nx']


# ---------------- (4) KOSIS 광업제조업조사(구조 속성) ----------------
def kosis_table():
    d = pd.read_csv(B/'공모전데이터/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
    return d


def ext_kosis(tr, te, c, final=False):
    d = kosis_table(); K = K50()
    base = d[d['연도'] == 2008].pivot_table(index='세세분류코드', columns='항목명', values='값', aggfunc='first').reindex(K)
    cov = pd.DataFrame({'ksic': K, 'kosis_items_2008': base.notna().sum(1).values}); cov.to_csv(R2/'ext_link_kosis.csv', index=False, encoding='utf-8-sig')
    pick = {}
    for nm in base.columns:
        if '사업체' in nm: pick['ks_est'] = nm
        elif '종사자' in nm: pick['ks_emp'] = nm
        elif '출하' in nm: pick['ks_ship'] = nm
        elif '부가' in nm: pick['ks_va'] = nm
    for d2 in (tr, te):
        for n, col in pick.items(): d2[n] = np.log1p(base[col].values.astype(float))[d2.k.values].astype(np.float32)
    return tr, te


EXT = {'io': ext_io, 'trade': ext_trade, 'trade_x': lambda tr, te, c, final=False: ext_trade(tr, te, c, final, with_export=True), 'kosis': ext_kosis}


# ---------------- (내부) 진입 흐름·다각화 모멘텀 — 제공 융합데이터의 KSIC 라벨만으로 ----------------
def _events(cut):
    """기준시점 연말까지 가시(등록 또는 공개)된 범위안 출원 사건(법인, 산업, 출원연도). 등록=registry 원출원인 코드, 공개=법인 별칭 매칭."""
    from r2 import app_pub
    ins = lib.load()['ins']; K = K50(); kidx = {k: i for i, k in enumerate(K)}; ce = f'{cut}1231'
    a = ins[(ins.ay <= cut) & (ins.rgst_date <= ce)][['applicant_cd', 'ksic', 'ay']].copy(); a['k'] = a.ksic.map(kidx)
    p = app_pub(); p = p[(p.ay <= cut) & (((p.pub_date.str.len() == 8) & (p.pub_date <= ce)) | ((p.rgst_date.str.len() == 8) & (p.rgst_date <= ce)))]
    e = pd.concat([a[['applicant_cd', 'k', 'ay']], p[['applicant_cd', 'k', 'ay']]])
    return e.groupby(['applicant_cd', 'k']).ay.min().reset_index()   # 법인×산업 첫 (가시) 출원연도


def flow_matrix(cut, win=3):
    """최근 win년 진입 흐름 Tr[j,k] = P(최근 win년 k 첫 출원 | win년 전 j 보유, k 미보유) — 모든 법인, 기준시점 가시 정보"""
    f = _events(cut); old = f[f.ay <= cut - win]; new = f[(f.ay > cut - win)]
    firms = pd.Index(f.applicant_cd.unique()); fi = pd.Series(np.arange(len(firms)), index=firms)
    H = np.zeros((len(firms), 50), np.float32); H[fi[old.applicant_cd].values, old.k.values] = 1
    N = np.zeros((len(firms), 50), np.float32); N[fi[new.applicant_cd].values, new.k.values] = 1
    N = N * (1 - H)
    num = H.T @ N; den = H.T @ (1 - H)
    Tr = (num + 0.5 * N.sum(0)[None, :] / max(len(firms), 1)) / (den + 1.0)
    np.fill_diagonal(Tr, 0)
    return Tr.astype(np.float32), f


def ext_flow5(tr, te, c, final=False):
    """진입 흐름 5년 창 변형(fl5_mix, fl5_max)"""
    for d in (tr, te):
        for n in ('fl5_mix', 'fl5_max'): d[n] = np.float32(np.nan)
        for cut in d.cut.unique():
            Tr, f = flow_matrix(cut, 5); idx = np.nonzero(d.cut.values == cut)[0]
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            g = f[f.applicant_cd.isin(firms)]
            H = np.zeros((len(firms), 50), np.float32); H[fi[g.applicant_cd].values, g.k.values] = 1
            fr_ = fi[d.firm.values[idx]].values; k = d.k.values[idx]; nh = H.sum(1)
            mix = (H @ Tr) / np.maximum(nh, 1)[:, None]; mx = np.zeros_like(H)
            for j in range(50): mx = np.maximum(mx, H[:, [j]] * Tr[j][None, :])
            d.loc[d.index[idx], 'fl5_mix'] = np.where(nh[fr_] > 0, mix[fr_, k], np.nan).astype(np.float32)
            d.loc[d.index[idx], 'fl5_max'] = np.where(nh[fr_] > 0, mx[fr_, k], np.nan).astype(np.float32)
    return tr, te


def ext_flow(tr, te, c, final=False):
    for d in (tr, te):
        for n in ('fl_mix', 'fl_max', 'dv_new3', 'dv_new_p3', 'dv_held'): d[n] = np.float32(np.nan)
        for cut in d.cut.unique():
            Tr, f = flow_matrix(cut); idx = np.nonzero(d.cut.values == cut)[0]
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            g = f[f.applicant_cd.isin(firms)]
            H = np.zeros((len(firms), 50), np.float32); H[fi[g.applicant_cd].values, g.k.values] = 1
            fr_ = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            nh = H.sum(1)
            mix = (H @ Tr) / np.maximum(nh, 1)[:, None]
            mx = np.zeros_like(H)
            for j in range(50): mx = np.maximum(mx, H[:, [j]] * Tr[j][None, :])
            n3 = g[g.ay > cut - 3].groupby('applicant_cd').size(); p3 = g[(g.ay > cut - 6) & (g.ay <= cut - 3)].groupby('applicant_cd').size()
            fm = pd.Series(d.firm.values[idx])
            d.loc[d.index[idx], 'fl_mix'] = np.where(nh[fr_] > 0, mix[fr_, k], np.nan).astype(np.float32)
            d.loc[d.index[idx], 'fl_max'] = np.where(nh[fr_] > 0, mx[fr_, k], np.nan).astype(np.float32)
            d.loc[d.index[idx], 'dv_new3'] = fm.map(n3).fillna(0).values.astype(np.float32)
            d.loc[d.index[idx], 'dv_new_p3'] = fm.map(p3).fillna(0).values.astype(np.float32)
            d.loc[d.index[idx], 'dv_held'] = nh[fr_].astype(np.float32)
    return tr, te
FLOW_COLS = ['fl_mix', 'fl_max', 'dv_new3', 'dv_new_p3', 'dv_held']
EXT['flow'] = ext_flow
EXT['flow5'] = ext_flow5


# ---------------- (1′) 산업연관표 간접 연결: 생산유발계수(레온티에프 역행렬) ----------------
def io_leontief():
    p = R2/'cache'/'io_leo.npz'
    if p.exists(): return np.load(p)['L']
    K = K50()
    xl = EXTD/'산업연관표/2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx'
    a = pd.read_excel(xl, sheet_name='10-생산유발계수', header=None)
    # 행·열 코드 위치 탐색(3자리 숫자)
    hr = None
    for i in range(12):
        row = a.iloc[i, 2:].astype(str).str.strip()
        if (row.str.fullmatch(r'\d{3}')).sum() > 100: hr = i; break
    cc = [j for j in a.columns[2:] if str(a.iat[hr, j]).strip().isdigit() and len(str(a.iat[hr, j]).strip()) == 3]
    rr = [i for i in a.index[hr + 1:] if str(a.iat[i, 0]).strip().isdigit() and len(str(a.iat[i, 0]).strip()) == 3]
    codes = [str(a.iat[i, 0]).strip() for i in rr]; ccodes = [str(a.iat[hr, j]).strip() for j in cc]
    assert codes == ccodes
    Lm = a.loc[rr, cc].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    lk = pd.read_csv(R2/'ext_link_io.csv', dtype=str); lk['n_products'] = lk.n_products.astype(float)
    idx = {cd: i for i, cd in enumerate(codes)}; Wk = np.zeros((50, len(codes)))
    for _, r in lk.iterrows():
        if r.io3 in idx: Wk[K.index(r.k5), idx[r.io3]] += r.n_products
    Wk = Wk / np.maximum(Wk.sum(1, keepdims=True), 1e-12)
    L = Wk @ Lm @ Wk.T            # [j,k]: k 최종수요 1단위가 j 산출을 유발하는 정도(직·간접)
    np.fill_diagonal(L, 0); np.savez(p, L=L); return L


def ext_leo(tr, te, c, final=False):
    L = io_leontief()
    for d in (tr, te):
        mix = firm_mix(d, final); v1 = np.zeros(len(d), np.float32); v2 = np.zeros(len(d), np.float32)
        for cut, (fi, Sn, S) in mix.items():
            idx = np.nonzero(d.cut.values == cut)[0]; f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            v1[idx] = (Sn @ L)[f, k]; v2[idx] = (Sn @ L.T)[f, k]
        d['leo_in'] = v1; d['leo_out'] = v2
    return tr, te
LEO_COLS = ['leo_in', 'leo_out']
EXT['leo'] = ext_leo


# ---------------- (내부 5개국) 목표 산업의 해외 출원 모멘텀(국가 간 선행) ----------------
def glob_table():
    p = R2/'cache'/'glob_kyc.parquet'
    if p.exists(): return pd.read_parquet(p)
    import duckdb
    M = (B/'분석/모델경쟁_20260927/_shared/meta').as_posix() + '/*.parquet'
    t = duckdb.sql(f"""select cc, ksic, appl_year as ay, substr(pub_date,1,4) as py, count(distinct appl_no) n
                      from read_parquet('{M}') where appl_year>=1995 and appl_year<=2022 and pub_date is not null and length(pub_date)>=4
                      group by 1,2,3,4""").df()
    t['py'] = pd.to_numeric(t.py, errors='coerce'); t.to_parquet(p, index=False); return t


def ext_glob(tr, te, c, final=False):
    t = glob_table(); K = K50(); t = t[t.ksic.astype(str).isin(K)]; t['k'] = t.ksic.astype(str).map({k: i for i, k in enumerate(K)})
    for d in (tr, te):
        for n in ('gl_us_g', 'gl_cn_g', 'gl_jp_g', 'gl_ep_g', 'gl_for_g', 'gl_kr_sh'): d[n] = np.float32(np.nan)
        for cut in d.cut.unique():
            v = t[(t.py <= cut)]      # 🔒 기준시점 연말까지 공개된 출원만
            def cnt(ccs, a0, a1):
                s = v[v.cc.isin(ccs) & (v.ay >= a0) & (v.ay <= a1)].groupby('k').n.sum(); return s.reindex(range(50)).fillna(0).values
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]
            out = {}
            for nm, ccs in (('us', ['US']), ('cn', ['CN']), ('jp', ['JP']), ('ep', ['EP']), ('for', ['US', 'CN', 'JP', 'EP'])):
                r = cnt(ccs, cut - 4, cut - 2); b = cnt(ccs, cut - 7, cut - 5)   # 공개지연 고려: 최근 2년 제외
                out[f'gl_{nm}_g'] = np.log1p(r) - np.log1p(b)
            kr = cnt(['KR'], cut - 4, cut - 2); al = cnt(['KR', 'US', 'CN', 'JP', 'EP'], cut - 4, cut - 2)
            out['gl_kr_sh'] = kr / np.maximum(al, 1)
            for n, val in out.items(): d.loc[d.index[idx], n] = val[k].astype(np.float32)
    return tr, te
GLOB_COLS = ['gl_us_g', 'gl_cn_g', 'gl_jp_g', 'gl_ep_g', 'gl_for_g', 'gl_kr_sh']
EXT['glob'] = ext_glob


# ---------------- (2″) 무역 쌍 수준: 기업 보유 산업의 수입 압력 vs 목표 산업 ----------------
def ext_trade_pair(tr, te, c, final=False):
    t = trade_table(); K = K50()
    piv = t.pivot(index='year', columns='ksic', values='imp').reindex(columns=K); cn = t.pivot(index='year', columns='ksic', values='imp_cn').reindex(columns=K)
    def g3(P, y1):
        if (y1 in P.index) and (y1 - 3 in P.index): return (np.log1p(P.loc[y1].values.astype(float)) - np.log1p(P.loc[y1 - 3].values.astype(float)))
        return np.full(50, np.nan)
    for d in (tr, te):
        mix = firm_mix(d, final)
        for n in ('tp_home_g3', 'tp_home_cn', 'tp_gap_g3'): d[n] = np.float32(np.nan)
        for cut, (fi, Sn, S) in mix.items():
            idx = np.nonzero(d.cut.values == cut)[0]; f = fi[d.firm.values[idx]].values; k = d.k.values[idx]; y1 = cut - 1   # 🔒 직전연도
            gi = g3(piv, y1); gc = g3(cn, y1)
            gi0 = np.nan_to_num(gi); gc0 = np.nan_to_num(gc); w = Sn * (~np.isnan(gi))[None, :]
            home = (w @ gi0) / np.maximum(w.sum(1), 1e-9); homec = (w @ gc0) / np.maximum(w.sum(1), 1e-9)
            ok = np.isfinite(gi).any()
            d.loc[d.index[idx], 'tp_home_g3'] = (home[f] if ok else np.nan)
            d.loc[d.index[idx], 'tp_home_cn'] = (homec[f] if ok else np.nan)
            d.loc[d.index[idx], 'tp_gap_g3'] = (gi[k] - home[f] if ok else np.nan)
    return tr, te
TP_COLS = ['tp_home_g3', 'tp_home_cn', 'tp_gap_g3']
EXT['trade_pair'] = ext_trade_pair


# ---------------- (내부 IPC–KSIC) 산업 간 기술 근접도: KSIC별 IPC 서브클래스 분포 코사인 ----------------
_IPCS = {}
def ipc_sim(cut):
    if cut in _IPCS: return _IPCS[cut]
    ins = lib.load()['ins']; K = K50(); kidx = {k: i for i, k in enumerate(K)}
    if 'mipc' not in ins.columns:
        ip = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['rgst_no', 'mipc']).drop_duplicates('rgst_no')
        ins = ins.merge(ip, on='rgst_no', how='left'); lib.load()['ins'] = ins
    v = ins[(ins.ay <= cut) & (ins.rgst_date <= f'{cut}1231')].drop_duplicates('rgst_no')
    v = v.assign(sc=v.mipc.fillna('').astype(str).str[:4]); v = v[v.sc.str.len() == 4]
    ct = pd.crosstab(v.ksic.map(kidx), v.sc).reindex(range(50), fill_value=0).values.astype(np.float32)
    ct = ct / np.maximum(np.linalg.norm(ct, axis=1, keepdims=True), 1e-9)
    S = ct @ ct.T; np.fill_diagonal(S, 0); _IPCS[cut] = S; return S


def ext_ipcsim(tr, te, c, final=False):
    for d in (tr, te):
        mix = firm_mix(d, final); a = np.zeros(len(d), np.float32); b = np.zeros(len(d), np.float32)
        for cut, (fi, Sn, S0) in mix.items():
            S = ipc_sim(cut); idx = np.nonzero(d.cut.values == cut)[0]; f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            a[idx] = (Sn @ S)[f, k] - 0.0
            held = (S0 > 0).astype(np.float32); mx = np.zeros_like(held)
            for j in range(50): mx = np.maximum(mx, held[:, [j]] * S[j][None, :])
            b[idx] = mx[f, k]
        d['ipc_mix'] = a; d['ipc_max'] = b
    return tr, te
EXT['ipcsim'] = ext_ipcsim


def ext_ipcfit(tr, te, c, final=False, level='sc', names=('ipc_fit', 'ipc_fit_rank')):
    """기업 자신의 IPC 서브클래스 프로파일(기준시점 가시 범위안 등록특허) × 목표 KSIC의 IPC 프로파일 코사인(직접 기술 적합도)"""
    ins = lib.load()['ins']; K = K50(); kidx = {k: i for i, k in enumerate(K)}
    cut_ipc = (lambda s: s.str[:4]) if level == 'sc' else (lambda s: s.str.split('/').str[0].str.replace(' ', '', regex=False))
    for d in (tr, te):
        out = np.full(len(d), np.nan, np.float32); top = np.full(len(d), np.nan, np.float32)
        for cut in d.cut.unique():
            v = ins[(ins.ay <= cut) & (ins.rgst_date <= f'{cut}1231')].drop_duplicates(['applicant_cd', 'rgst_no'])
            v = v.assign(sc=cut_ipc(v.mipc.fillna('').astype(str))); v = v[v.sc.str.len() >= 4]
            scs = pd.Index(v.sc.unique()); si = pd.Series(np.arange(len(scs)), index=scs)
            KP = np.zeros((50, len(scs)), np.float32); np.add.at(KP, (v.ksic.map(kidx).values, si[v.sc].values), 1)
            KP = KP / np.maximum(np.linalg.norm(KP, axis=1, keepdims=True), 1e-9)
            idx = np.nonzero(d.cut.values == cut)[0]; firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            g = v[v.applicant_cd.isin(firms)]
            FP = np.zeros((len(firms), len(scs)), np.float32); np.add.at(FP, (fi[g.applicant_cd].values, si[g.sc].values), 1)
            has = FP.sum(1) > 0; FP = FP / np.maximum(np.linalg.norm(FP, axis=1, keepdims=True), 1e-9)
            C = FP @ KP.T                     # 기업×KSIC 코사인
            f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            out[idx] = np.where(has[f], C[f, k], np.nan)
            rk = (-C).argsort(1).argsort(1); top[idx] = np.where(has[f], rk[f, k], np.nan)
        d[names[0]] = out; d[names[1]] = top
    return tr, te
EXT['ipcfit'] = ext_ipcfit
EXT['ipcfit_g'] = lambda tr, te, c, final=False: ext_ipcfit(tr, te, c, final, level='mg', names=('ipcg_fit', 'ipcg_fit_rank'))
