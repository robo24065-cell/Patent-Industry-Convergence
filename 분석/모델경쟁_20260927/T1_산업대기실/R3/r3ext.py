# T1 R3 외부데이터 결합(추가분) — r2ext.EXT에 더해 쓴다. 모든 피처는 각 원점(cut) 연말 시점에 알 수 있던 값만.
#  sci  : OpenAlex KSIC 키워드 논문(기업 소속 저자 전 세계·한국, 국가별 전체) — 목표 산업의 과학 활력(연도 ≤ cut−1), 기업 보유산업 대비 격차
#  price: ECOS 생산자물가(기본분류)·수출입물가(품목) — 목표 산업 가격 변화(연평균, 이듬해 1월 공표 → cut−1년까지)
#  design / um : KIPRIS 디자인권·범위밖 실용신안(자체 수집 business_rights) 명칭 → KSIC 분류기(cap=c-12-31) → 기업×산업 노출(등록일 ≤ cut 연말)
#  acq  : 등록원부 권리자 이력(자체 수집 registry_full.owner) — 원출원인이 아닌 법인이 그 KSIC 범위안 특허를 넘겨받은 사건(이전등록일 ≤ cut 연말)
#  lic  : 등록원부 전용·통상실시권 — 그 KSIC 특허의 실시권자(설정일 ≤ cut 연말)
import sys, pathlib
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; sys.path.insert(0, str(R3)); sys.path.insert(0, str(T1/'R2'))
import numpy as np, pandas as pd
from r2 import lib, B
import r2ext
EXTD = B/'분석/모델경쟁_20260927/_shared/ext'
EXT = {}
K50 = r2ext.K50
PDIR = {}   # 원점별 분류기 캐시 경로 재지정(최종·미래 실행기용). 비어 있으면 R3/cache/o<c>


def _mixw(d):
    """cut별 (fi, Sn, S) 기업 산업 믹스 — r2ext.firm_mix(기준시점 가시 범위안 등록 + 범위밖 제목 노출)"""
    return r2ext.firm_mix(d)


def _put(d, name, vals, idx):
    if name not in d.columns: d[name] = np.float32(np.nan)
    d.loc[d.index[idx], name] = vals.astype(np.float32)


# ---------------- OpenAlex 과학 활력 ----------------
_SCI = {}
def sci_tables():
    if _SCI: return _SCI
    K = K50()
    e = pd.read_parquet(EXTD/'science_openalex_ksic_extra_year.parquet')
    c = pd.read_parquet(EXTD/'science_openalex_ksic_country_year.parquet')
    for v in ('company_ALL', 'company_KR'):
        p = e[e.variant == v].pivot_table(index='year', columns='ksic', values='n_works', aggfunc='sum')
        _SCI[v] = p.reindex(columns=K).fillna(0); _SCI[v + '_TOT'] = p['TOTAL'] if 'TOTAL' in p.columns else p.reindex(columns=K).sum(1)
    p = c[c.country == 'ALL'].pivot_table(index='year', columns='ksic', values='n_works', aggfunc='sum')
    _SCI['ALL'] = p.reindex(columns=K).fillna(0)
    t = pd.read_parquet(EXTD/'science_openalex_country_total_year.parquet'); _SCI['ALL_TOT'] = t[t.country == 'ALL'].set_index('year').n_works_total
    return _SCI


def _win(P, a, b):
    yrs = [y for y in range(a, b + 1) if y in P.index]
    return P.loc[yrs].sum(0) if yrs else None


def _g3(P, T, cut):
    """점유 성장: log(최근 3년 합) − log(이전 3년 합), 총량 성장 차감. 연도 ≤ cut−1 (🔒 원점 연도 자료 미사용)"""
    r = _win(P, cut - 3, cut - 1); b = _win(P, cut - 6, cut - 4)
    rt = T.reindex(range(cut - 3, cut)).sum(); bt = T.reindex(range(cut - 6, cut - 3)).sum()
    return (np.log1p(r.values) - np.log1p(b.values)) - (np.log1p(rt) - np.log1p(bt)), np.log1p(r.values)


def ext_sci(tr, te, c, final=False):
    S = sci_tables()
    for d in (tr, te):
        mix = _mixw(d)
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]
            gc, _ = _g3(S['company_ALL'], S['company_ALL_TOT'], cut)
            gk, lk = _g3(S['company_KR'], S['company_KR_TOT'], cut)
            ga, _ = _g3(S['ALL'], S['ALL_TOT'], cut)
            fi, Sn, S0 = mix[cut]; f = fi[d.firm.values[idx]].values
            home = Sn @ gc   # 기업 보유산업 믹스 가중 기업논문 성장
            _put(d, 'sci_c_g3', gc[k], idx); _put(d, 'sci_kr_g3', gk[k], idx); _put(d, 'sci_kr_l', lk[k], idx)
            _put(d, 'sci_all_g3', ga[k], idx); _put(d, 'sci_gap', gc[k] - home[f], idx)
    return tr, te
SCI_COLS = ['sci_c_g3', 'sci_kr_g3', 'sci_kr_l', 'sci_all_g3', 'sci_gap']
EXT['sci'] = ext_sci


# ---------------- ECOS 물가 ----------------
def ext_price(tr, te, c, final=False):
    K = K50(); p = pd.read_parquet(EXTD/'demand_ksic_price_panel_1996_2021.parquet')
    P = {s: p[p.series == s].pivot_table(index='year', columns='ksic', values='lidx', aggfunc='first').reindex(columns=K) for s in ('ppi_basic', 'exp_item', 'imp_item')}
    def g(s, y1, lag):
        M = P[s]
        if y1 in M.index and (y1 - lag) in M.index: return (M.loc[y1] - M.loc[y1 - lag]).values.astype(float)
        return np.full(50, np.nan)
    for d in (tr, te):
        mix = _mixw(d)
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; k = d.k.values[idx]; y1 = cut - 1   # 🔒 연평균은 이듬해 1월 공표 → cut−1년까지
            ppi3 = g('ppi_basic', y1, 3); ppi1 = g('ppi_basic', y1, 1); ex3 = g('exp_item', y1, 3); im3 = g('imp_item', y1, 3)
            fi, Sn, S0 = mix[cut]; f = fi[d.firm.values[idx]].values
            ok = np.isfinite(ppi3); w = Sn * ok[None, :]; home = (w @ np.nan_to_num(ppi3)) / np.maximum(w.sum(1), 1e-9)
            _put(d, 'pr_ppi_g3', ppi3[k], idx); _put(d, 'pr_ppi_g1', ppi1[k], idx); _put(d, 'pr_exp_g3', ex3[k], idx); _put(d, 'pr_imp_g3', im3[k], idx)
            _put(d, 'pr_gap', np.where(ok[k], ppi3[k] - home[f], np.nan), idx)
    return tr, te
PRICE_COLS = ['pr_ppi_g3', 'pr_ppi_g1', 'pr_exp_g3', 'pr_imp_g3', 'pr_gap']
EXT['price'] = ext_price


# ---------------- KIPRIS 디자인권·실용신안 명칭 노출 ----------------
def _title_exposure(tr, te, c, kind, pre, thr=0.2):
    FC = PDIR.get(c, R3/'cache'/f'o{c}')
    P = np.load(FC/f'P_{kind}.npy'); ix = pd.read_parquet(FC/f'P_{kind}_index.parquet')
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            m = (ix.rgst_date.values <= f'{cut}1231') & ix.applicant_cd.isin(firms).values
            r = fi[ix.applicant_cd.values[m]].values; Pm = P[m].astype(np.float32); rec = ix.ay.values[m] > cut - 3
            nF = len(firms); SUM = np.zeros((nF, 50), np.float32); np.add.at(SUM, r, Pm)
            MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, r, Pm)
            N20 = np.zeros((nF, 50), np.float32); np.add.at(N20, r, (Pm > thr).astype(np.float32))
            REC = np.zeros((nF, 50), np.float32); np.add.at(REC, r[rec], Pm[rec])
            n = np.bincount(r, minlength=nF).astype(np.float32)
            f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            _put(d, f'{pre}_lsum', np.log1p(SUM[f, k]), idx); _put(d, f'{pre}_max', MX[f, k], idx); _put(d, f'{pre}_n20', np.log1p(N20[f, k]), idx)
            _put(d, f'{pre}_rec3', np.log1p(REC[f, k]), idx); _put(d, f'{pre}_n', np.log1p(n[f]), idx)
    return tr, te
DES_COLS = ['ds_lsum', 'ds_max', 'ds_n20', 'ds_rec3', 'ds_n']
UM_COLS = ['um_lsum', 'um_max', 'um_n20', 'um_rec3', 'um_n']
EXT['design'] = lambda tr, te, c, final=False: _title_exposure(tr, te, c, 'design', 'ds')
EXT['um'] = lambda tr, te, c, final=False: _title_exposure(tr, te, c, 'um', 'um')


# ---------------- 등록원부 권리 이전(특허 매입)·실시권 ----------------
def _events_pair(tr, te, src, pre, rec_years=3, asg_only=False):
    K = K50(); kidx = {k: i for i, k in enumerate(K)}
    e = pd.read_parquet(R3/'cache'/f'src_{src}.parquet'); e['k'] = e.ksic.astype(str).map(kidx); e = e.dropna(subset=['k']); e['k'] = e.k.astype(int)
    if asg_only and 'cs' in e.columns: e = e[e.cs.fillna('').str.contains('양도')]
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; ce = f'{cut}1231'; cr = f'{cut - rec_years}1231'
            v = e[e.date <= ce]
            ck = v.groupby(['applicant_cd', 'k']).size(); ckr = v[v.date > cr].groupby(['applicant_cd', 'k']).size(); n = v.groupby('applicant_cd').size()
            mi = pd.MultiIndex.from_arrays([d.firm.values[idx], d.k.values[idx]])
            _put(d, f'{pre}_k', np.log1p(ck.reindex(mi).fillna(0).values), idx)
            _put(d, f'{pre}_k_rec', np.log1p(ckr.reindex(mi).fillna(0).values), idx)
            _put(d, f'{pre}_n', np.log1p(pd.Series(d.firm.values[idx]).map(n).fillna(0).values), idx)
    return tr, te
ACQ_COLS = ['acq_k', 'acq_k_rec', 'acq_n']
LIC_COLS = ['lic_k', 'lic_k_rec', 'lic_n']
EXT['acq'] = lambda tr, te, c, final=False: _events_pair(tr, te, 'acq', 'acq')
EXT['acq_asg'] = lambda tr, te, c, final=False: _events_pair(tr, te, 'acq', 'acq', asg_only=True)
EXT['lic'] = lambda tr, te, c, final=False: _events_pair(tr, te, 'lic', 'lic')


# ---------------- KIPRIS 상표권 활동(기업 수준) ----------------
def ext_tm(tr, te, c, final=False):
    """기업의 상표 등록 활동(등록일 ≤ cut 연말): 누적 수, 최근 3년 출원분, 직전 3년 대비 증가 — 신제품·브랜드 확장(시장 다각화) 신호"""
    e = pd.read_parquet(R3/'cache'/'src_tm.parquet')
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]; v = e[e.rgst_date <= f'{cut}1231']
            n = v.groupby('applicant_cd').size(); r = v[v.ay > cut - 3].groupby('applicant_cd').size(); p = v[(v.ay > cut - 6) & (v.ay <= cut - 3)].groupby('applicant_cd').size()
            fm = pd.Series(d.firm.values[idx])
            nr = np.log1p(fm.map(r).fillna(0).values); npv = np.log1p(fm.map(p).fillna(0).values)
            _put(d, 'tm_n', np.log1p(fm.map(n).fillna(0).values), idx); _put(d, 'tm_rec3', nr, idx); _put(d, 'tm_g', nr - npv, idx)
    return tr, te
TM_COLS = ['tm_n', 'tm_rec3', 'tm_g']
EXT['tm'] = ext_tm


# ---------------- 디자인 변형: 주분류(argmax) 기준 수·기업 내 비중 ----------------
def ext_design_top(tr, te, c, final=False, thr=0.5):
    FC = PDIR.get(c, R3/'cache'/f'o{c}')
    P = np.load(FC/'P_design.npy'); ix = pd.read_parquet(FC/'P_design_index.parquet')
    am = P.argmax(1); pm = P.max(1).astype(np.float32)
    for d in (tr, te):
        for cut in d.cut.unique():
            idx = np.nonzero(d.cut.values == cut)[0]
            firms = pd.Index(d.firm.values[idx]).unique(); fi = pd.Series(np.arange(len(firms)), index=firms)
            m = (ix.rgst_date.values <= f'{cut}1231') & ix.applicant_cd.isin(firms).values
            r = fi[ix.applicant_cd.values[m]].values; a = am[m]; hi = pm[m] > thr
            nF = len(firms); TOP = np.zeros((nF, 50), np.float32); np.add.at(TOP, (r[hi], a[hi]), 1)
            SH = np.zeros((nF, 50), np.float32); np.add.at(SH, r, P[m].astype(np.float32))
            n = np.bincount(r, minlength=nF).astype(np.float32)
            f = fi[d.firm.values[idx]].values; k = d.k.values[idx]
            _put(d, 'ds_top', np.log1p(TOP[f, k]), idx); _put(d, 'ds_share', SH[f, k] / np.maximum(n[f], 1), idx)
    return tr, te
EXT['design_top'] = ext_design_top
