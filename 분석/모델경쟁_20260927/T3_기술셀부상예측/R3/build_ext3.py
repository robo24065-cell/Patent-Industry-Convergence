"""T3 3라운드 외부 데이터 결합층 (A단계: 이미 수집된 외부 데이터).
산출: ext_ksic_t0_{var}.parquet (KSIC×t0), ext_cell_t0_{var}.parquet (셀×t0), ext3_link_*.csv (연계표)
시점 규칙 — 공개연도 변형의 원칙 "t0년 말까지 공개된 자료만"을 두 변형 모두에 적용:
  - KOSIS 광공업생산지수(월별, 익월 공표): pub 변형은 t0-1년 12월 ~ t0년 11월 12개월 창(12월 지수는 이듬해 1월 공표) → 2년 성장·1년 성장·출하 2년 성장.
      (2020=100 재기준 계열의 수준값은 2라운드 감사에서 누수로 적발 → 성장률만. 링크 계열이라 성장률은 대체로 보존되나 '준 스냅샷')
  - OpenAlex 논문(KSIC 키워드): 논문 발행연도 ≤ t0 (현재 스냅샷을 소급 집계 → 준-정합). 국가 총량 대비 점유로 정규화.
  - ECOS 수출·수입·생산자 물가(연평균, 이듬해 1월 공표): 연도 ≤ t0-1.
  - NBER-CES 미국 제조업(약 1.5년 공표 시차, 2018 종료): 연도 ≤ t0-3 (최종 원점 t0=2021 도 2018 까지로 같은 시차 유지).
  - OECD IP5 패밀리(우선권연도, 완결까지 3~4년): 연도 ≤ t0-3. 현재 빈티지만 있음(준-정합).
  - KPSS 특허가치 xi: US 특허 issue_date ≤ t0-12-31 인 것만. 제공 특허의 family US 토큰으로 연결(quality_kpss_family_link).
  - WIPO 관청×공개연도×기술분야 공개건수: 공개연도 y 통계는 y+1년 공표 → 보정계수는 y ≤ t0-1 만.
사용: python build_ext3.py --var pub|app
"""
import os, sys, re, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument('--var', default='pub'); ap.add_argument('--t0max', type=int, default=None); a = ap.parse_args()
os.environ['T3R3_VAR'] = a.var
import numpy as np, pandas as pd
import r3lib as L
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
DATA = os.path.join(ROOT, '공모전데이터'); SHX = os.path.join(HERE, '..', '..', '_shared', 'ext')
T0MAX = a.t0max or max(L.DEV_T0)
assert T0MAX + 3 < L.F_YEAR or os.environ.get('T3R3_FINAL_UNLOCK') == '1'
T0S = list(range(2005, T0MAX + 1))
P = L.load(); cells = P['cells']; ksic_c = P['ksic']; mg_c = P['mg']; K50 = sorted(set(ksic_c))
n = P['A']['n']

def lg(a_, b_):
    return np.log(a_ / b_) if (a_ == a_ and b_ == b_ and a_ > 0 and b_ > 0) else np.nan

# ================= 1. KOSIS 광공업생산지수 (성장률 3피처) =================
GI = os.path.join(DATA, '외부데이터', '경기지표')
pi = pd.read_csv(os.path.join(GI, 'kosis_광공업생산지수_월별.csv'), dtype={'산업코드': str}); pi['값'] = pd.to_numeric(pi.값, errors='coerce')
pi['y'] = pi.연월 // 100; pi['m'] = pi.연월 % 100
pi = pi[pi.연월 <= (T0MAX * 100 + 12)]
# 연도 창: pub 변형 = (y-1)년 12월 ~ y년 11월 (y년 말까지 공표된 12개월), app 변형 = y년 1~12월 (2라운드와 동일)
if a.var == 'pub':
    pi['wy'] = np.where(pi.m == 12, pi.y + 1, pi.y)
else:
    pi['wy'] = pi.y
cnt = pi.groupby(['산업코드', '항목코드', 'wy']).값.count()
PA = pi.groupby(['산업코드', '항목코드', 'wy']).값.mean()[cnt == 12]
def pv(code, it, y):
    try: return PA.loc[(code, it, y)]
    except KeyError: return np.nan
rows = []; link_pi = []
for ks in K50:
    c3 = 'C' + ks[:3]; c2 = 'C' + ks[:2]
    code = c3 if (c3, 'T10', 2010) in PA.index else c2
    link_pi.append(dict(ksic=ks, prod_index_code=code, level='소분류3' if code == c3 else '중분류2'))
    for t0 in T0S:
        p = lambda it, y: pv(code, it, y)
        r = dict(ksic=ks, t0=t0)
        r['pi_prod_g2'] = lg((p('T10', t0) + p('T10', t0 - 1)) / 2, (p('T10', t0 - 2) + p('T10', t0 - 3)) / 2)
        r['pi_prod_g1'] = lg(p('T10', t0), p('T10', t0 - 1))
        r['pi_ship_g2'] = lg((p('T11', t0) + p('T11', t0 - 1)) / 2, (p('T11', t0 - 2) + p('T11', t0 - 3)) / 2)
        rows.append(r)
PIK = pd.DataFrame(rows)
pd.DataFrame(link_pi).to_csv(os.path.join(HERE, 'ext3_link_prodindex.csv'), index=False, encoding='utf-8-sig')

# ================= 2. OpenAlex (KSIC 키워드 × 저자국, 기업 소속 논문) =================
oc = pd.read_parquet(os.path.join(SHX, 'science_openalex_ksic_country_year.parquet'))
ox = pd.read_parquet(os.path.join(SHX, 'science_openalex_ksic_extra_year.parquet'))
ot = pd.read_parquet(os.path.join(SHX, 'science_openalex_country_total_year.parquet'))
TOT = ot.set_index(['country', 'year']).n_works_total
OC = oc.pivot_table(index=['ksic', 'year'], columns='country', values='n_works', aggfunc='sum').fillna(0)
OX = ox.pivot_table(index=['ksic', 'year'], columns='variant', values='n_works', aggfunc='sum').fillna(0)
def osum(tab, ks, col, a_, b_):
    try: s = tab.loc[ks][col]
    except KeyError: return np.nan
    return float(s.reindex(range(a_, b_ + 1)).fillna(0).sum())
def tsum(c, a_, b_):
    return float(sum(TOT.get((c, y), np.nan) for y in range(a_, b_ + 1)))
rows = []
for t0 in T0S:
    for ks in K50:
        r = dict(ksic=ks, t0=t0)
        def shg(tab, col, tc):   # 3년 점유 성장 (자기 국가 총량 대비)
            w0 = osum(tab, ks, col, t0 - 2, t0) / tsum(tc, t0 - 2, t0); w1 = osum(tab, ks, col, t0 - 5, t0 - 3) / tsum(tc, t0 - 5, t0 - 3)
            return np.log((w0 * 1e6 + 1) / (w1 * 1e6 + 1)) if (w0 == w0 and w1 == w1) else np.nan
        r['oa_all_g'] = shg(OC, 'ALL', 'ALL')
        r['oa_kr_g'] = shg(OC, 'KR', 'KR')
        fx0 = sum(osum(OC, ks, c, t0 - 2, t0) for c in ('US', 'JP', 'CN', 'DE')); fx1 = sum(osum(OC, ks, c, t0 - 5, t0 - 3) for c in ('US', 'JP', 'CN', 'DE'))
        ft0 = sum(tsum(c, t0 - 2, t0) for c in ('US', 'JP', 'CN', 'DE')); ft1 = sum(tsum(c, t0 - 5, t0 - 3) for c in ('US', 'JP', 'CN', 'DE'))
        r['oa_fx_g'] = np.log((fx0 / ft0 * 1e6 + 1) / (fx1 / ft1 * 1e6 + 1))
        r['oa_comp_g'] = shg(OX, 'company_ALL', 'ALL')
        r['oa_compkr_g'] = shg(OX, 'company_KR', 'KR')
        c0 = osum(OX, ks, 'company_ALL', t0 - 2, t0); a0 = osum(OC, ks, 'ALL', t0 - 2, t0)
        r['oa_comp_sh'] = np.log((c0 + 1) / (a0 + 1))                  # 기업 논문 비중(상업화 근접도)
        k0 = osum(OC, ks, 'KR', t0 - 2, t0)
        r['oa_kr_sh'] = np.log((k0 / tsum('KR', t0 - 2, t0) * 1e6 + 1) / (a0 / tsum('ALL', t0 - 2, t0) * 1e6 + 1))   # 한국 과학 특화도(RCA)
        rows.append(r)
OA = pd.DataFrame(rows)

# ================= 3. ECOS 물가 (수출·수입·생산자) =================
pr = pd.read_parquet(os.path.join(SHX, 'demand_ksic_price_panel_1996_2021.parquet'))
PR = pr.pivot_table(index=['ksic', 'year'], columns='series', values='lidx')
def pval(ks, s, y):
    try: return PR.loc[(ks, y), s]
    except KeyError: return np.nan
rows = []
for t0 in T0S:
    for ks in K50:
        r = dict(ksic=ks, t0=t0); y = t0 - 1
        for s, nm in (('exp_item', 'pr_exp'), ('imp_item', 'pr_imp'), ('ppi_basic', 'pr_ppi')):
            r[f'{nm}_g3'] = pval(ks, s, y) - pval(ks, s, y - 3)
            r[f'{nm}_g1'] = pval(ks, s, y) - pval(ks, s, y - 1)
        r['pr_tot'] = r['pr_exp_g3'] - r['pr_imp_g3']   # 교역조건 변화
        rows.append(r)
PRK = pd.DataFrame(rows)

# ================= 4. NBER-CES 미국 제조업 =================
us = pd.read_parquet(os.path.join(SHX, 'demand_ksic_us_mfg_panel_1958_2018.parquet')).set_index(['ksic', 'year'])
def uval(ks, c, y):
    try: return us.loc[(ks, y), c]
    except KeyError: return np.nan
rows = []
for t0 in T0S:
    for ks in K50:
        r = dict(ksic=ks, t0=t0); y = min(t0 - 3, 2018)
        r['us_rship_g3'] = lg(uval(ks, 'rship', y), uval(ks, 'rship', y - 3))
        r['us_emp_g3'] = lg(uval(ks, 'emp', y), uval(ks, 'emp', y - 3))
        r['us_tfp_g3'] = lg(uval(ks, 'tfp5', y), uval(ks, 'tfp5', y - 3))
        r['us_inv_int'] = lg(uval(ks, 'invest', y), uval(ks, 'vship', y))
        rows.append(r)
USK = pd.DataFrame(rows)

E = PIK.merge(OA, on=['ksic', 't0'], how='outer').merge(PRK, on=['ksic', 't0'], how='outer').merge(USK, on=['ksic', 't0'], how='outer')
E.to_parquet(os.path.join(HERE, f'ext_ksic_t0_{a.var}.parquet'))
print('ksic ext', E.shape, flush=True)

# ================= 5. 셀 수준: OECD IP5 한국 점유(WIPO35) · KPSS · WIPO 커버리지 =================
def mgn(s):
    s = str(s).replace(' ', '')
    m = re.match(r'^([A-H]\d\d[A-Z])(\d{1,4})/', s)
    if m: return f'{m.group(1)} {int(m.group(2))}'
    m = re.match(r'^([A-H]\d\d[A-Z])(\d{3,6})$', s)
    if m: return f'{m.group(1)} {int(s[4:len(s) - 2])}'
    return None
conc = pd.read_csv(os.path.join(SHX, 'quality_wipo_ipc_field_concordance.csv'))
sub_f = conc[conc.main_group.isna()].set_index('subclass').wipo_field.to_dict()
mg_f = {(r.subclass, int(r.main_group)): r.wipo_field for r in conc[conc.main_group.notna()].itertuples()}
ug = pd.Series(np.unique(mg_c))
def fld(m):
    sc, g = m.split(' '); return mg_f.get((sc, int(g)), sub_f.get(sc, np.nan))
MGF = pd.Series([fld(m) for m in ug], index=ug.values)
cell_f = MGF.reindex(mg_c).values
link_w = dict(n_cells=int(len(cells)), cells_mapped=int(np.isfinite(cell_f).sum()))
# IP5 (9P50_3, 발명자 기준, 우선권연도)
o = pd.read_parquet(os.path.join(SHX, 'quality_oecd_patents_wipo35.parquet'))
o = o[(o.PATENT_AUTHORITIES == '9P50_3') & (o.AGENT_ROLE == 'INVENTOR') & (o.WIPO != '_T')]
o['f'] = o.WIPO.str[-2:].astype(int)
IP = o.pivot_table(index=['f', 'TIME_PERIOD'], columns='REF_AREA', values='OBS_VALUE', aggfunc='sum')
def ipw(f, c, a_, b_):
    try: return float(IP.loc[f][c].reindex(range(a_, b_ + 1)).sum(min_count=b_ - a_ + 1))
    except KeyError: return np.nan
rowsf = []
for t0 in T0S:
    Lg = 3
    for f in range(1, 36):
        w1 = ipw(f, 'W', t0 - Lg - 2, t0 - Lg); w0 = ipw(f, 'W', t0 - Lg - 5, t0 - Lg - 3); k1 = ipw(f, 'KOR', t0 - Lg - 2, t0 - Lg); k0 = ipw(f, 'KOR', t0 - Lg - 5, t0 - Lg - 3)
        c1 = ipw(f, 'CHN', t0 - Lg - 2, t0 - Lg); c0 = ipw(f, 'CHN', t0 - Lg - 5, t0 - Lg - 3)
        rowsf.append(dict(f=f, t0=t0, ip5_kr_sh=k1 / w1, ip5_kr_sh_chg=k1 / w1 - k0 / w0, ip5_w_g=np.log(w1 / w0), ip5_cn_sh_chg=c1 / w1 - c0 / w0))
IPF = pd.DataFrame(rowsf)
# KPSS
kp = pd.read_parquet(os.path.join(SHX, 'quality_kpss_family_link.parquet'), columns=['cc', 'appl_no', 'appl_year', 'ksic', 'main_ipc', 'issue_date', 'xi_real'])
kp['mg'] = [mgn(x) for x in kp.main_ipc.values]; kp = kp.dropna(subset=['mg'])
kp['cell'] = kp.ksic + '|' + kp.mg; kp = kp[kp.cell.isin(set(cells))]
kp['iy'] = pd.to_datetime(kp.issue_date).dt.year; kp['lxi'] = np.log(kp.xi_real.clip(lower=1e-3))
link_w['kpss_rows_in_cells'] = int(len(kp)); link_w['kpss_cells'] = int(kp.cell.nunique())
rows = []
for t0 in T0S:
    k = kp[(kp.iy <= t0) & (kp.appl_year >= t0 - 5) & (kp.appl_year <= t0)]
    g = k.groupby('cell').agg(kp_n=('lxi', 'size'), kp_lxi=('lxi', 'mean'), kp_lxi_max=('lxi', 'max'))
    kk = k.groupby('ksic').lxi.mean()
    g = g.reindex(cells); g['kp_n'] = np.log1p(g.kp_n.fillna(0))
    g['kp_lxi_rel'] = g.kp_lxi - kk.reindex(ksic_c).values      # 같은 KSIC 대비 셀 가치 집약도
    g['cell'] = cells; g['t0'] = t0
    rows.append(g.reset_index(drop=True))
KP = pd.concat(rows, ignore_index=True)
# WIPO 커버리지: 이 변형의 셀 집계(공개연도 또는 출원연도)를 WIPO 분야별 공개건수와 비교 → 관청×분야×연도 보정계수 (y ≤ t0-1)
wp = pd.read_parquet(os.path.join(SHX, 'quality_wipo_vs_provided_pub.parquet'))
WP = wp.set_index(['cc', 'year', 'wipo_field']).wipo_pub
fdf = pd.DataFrame({'f': cell_f})
cov_rows = []
for j, c in enumerate(L.CCS):
    X = pd.DataFrame(n[:, j, :]); X['f'] = cell_f; G = X.groupby('f').sum()
    for f in G.index:
        for yi, y in enumerate(range(L.Y0, L.Y1 + 1)):
            w = WP.get((c, y, int(f)), np.nan)
            cov_rows.append(dict(cc=c, f=int(f), year=y, prov=float(G.loc[f, yi]), wipo=w, ratio=float(G.loc[f, yi]) / w if (w == w and w > 0) else np.nan))
COV = pd.DataFrame(cov_rows); COV.to_csv(os.path.join(HERE, f'ext3_wipo_coverage_{a.var}.csv'), index=False, encoding='utf-8-sig')
# 셀 피처: IP5
CF = pd.DataFrame({'cell': np.tile(cells, len(T0S)), 't0': np.repeat(T0S, len(cells)), 'f': np.tile(cell_f, len(T0S))})
CF = CF.merge(IPF, on=['f', 't0'], how='left').merge(KP, on=['cell', 't0'], how='left')
CF.drop(columns=['f']).to_parquet(os.path.join(HERE, f'ext_cell_t0_{a.var}.parquet'))
link_w['ip5_cells_with_value_t0_2016'] = int(CF[(CF.t0 == min(2016, T0MAX))].ip5_kr_sh.notna().sum())
link_w['kpss_cells_with_value_t0_2016'] = int(CF[(CF.t0 == min(2016, T0MAX))].kp_lxi.notna().sum())
pd.Series(link_w).to_csv(os.path.join(HERE, f'ext3_link_cell_{a.var}.csv'), encoding='utf-8-sig')
print('cell ext', CF.shape, link_w, flush=True)
# 연계율 요약 (KSIC)
cov = {c: int(E.groupby('ksic')[c].apply(lambda s: s.notna().any()).sum()) for c in ['pi_prod_g2', 'oa_all_g', 'oa_comp_g', 'pr_exp_g3', 'pr_ppi_g3', 'us_rship_g3']}
nn = {c: round(float(E[E.t0.between(2008, T0MAX)][c].notna().mean()), 3) for c in cov}
pd.DataFrame(dict(ksic_linked=cov, nonmissing_t0_2008plus=nn)).to_csv(os.path.join(HERE, f'ext3_link_ksic_{a.var}.csv'), encoding='utf-8-sig')
print(pd.DataFrame(dict(ksic_linked=cov, nonmissing_t0_2008plus=nn)))
