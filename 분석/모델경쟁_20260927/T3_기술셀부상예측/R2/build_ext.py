"""T3 2라운드 외부 데이터 결합층 (KSIC 연결).  산출: ext_ksic_t0.parquet, ext_cell_t0.parquet, ext_link_*.csv
시점 규칙(원점 t0 에서 쓸 수 있던 값만):
  - KOSIS 광업제조업조사(세세분류 5자리): 조사연도 Y 는 Y+1년 말 공표 → 원점 t0 에서는 Y ≤ t0-1 만. 2010년은 경제총조사라 결측.
  - 무역(UN Comtrade HS6, 한국 수입·수출): 연간 값은 다음 해 초 공개 → Y ≤ t0 (특허 출원 집계와 같은 기준).
  - 산업연관표: 2010년 접속표(2014 공표)를 개발 원점 전체에 구조 가중치로 사용(한계: 2010~2013 원점에는 공표 전 구조). 최종/미래 원점은 2015년 표.
사용: python build_ext.py [--maxyear 2019]  (개발: ≤2019, 🔒 2020~ 는 final_run_r2 에서만)
"""
import os, sys, argparse, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--maxyear', type=int, default=2019); ap.add_argument('--io_year', default='10')
ap.add_argument('--out_suffix', default=''); a = ap.parse_args()
import r2lib as L
assert a.maxyear <= L.Y1
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))   # 공모전 루트
DATA = os.path.join(ROOT, '공모전데이터'); FUS = os.path.join(ROOT, '분석', 'fusion')
SHX = os.path.join(HERE, '..', '..', '_shared', 'ext')
MAXY = a.maxyear
T0S = list(range(2005, MAXY + 1))

P = L.load(); cells = P['cells']; ksic_c = P['ksic']; mg_c = P['mg']
n = P['A']['n']; kr = n[:, 0, :]; fx = n[:, 1:, :].sum(1); tot = n.sum(1)
K50 = sorted(set(ksic_c))
def yi(y): return y - L.Y0

# ================= 1. KOSIS 광업제조업조사 =================
k = pd.read_csv(os.path.join(DATA, '작업데이터', 'kosis_광업제조업_50개산업.csv'), dtype={'세세분류코드': str}, encoding='utf-8-sig')
k = k[k.연도 <= MAXY - 1]  # 원점 ≤MAXY 에서 쓸 수 있는 조사연도(≤t0-1)만 읽음
items = {'T01': 'estab', 'T02': 'emp', 'T04': 'ship', 'T06': 'va', 'T07': 'asset'}
k = k[k.항목코드.isin(items)]
w10 = k[k.기준 == '10차'].pivot_table(index=['세세분류코드', '항목코드'], columns='연도', values='값')
w11 = k[k.기준 == '11차'].pivot_table(index=['세세분류코드', '항목코드'], columns='연도', values='값') if (k.기준 == '11차').any() else None
S = w10.copy()
if w11 is not None and 2020 in w11.columns and 2020 in w10.columns:   # 차수 연결: 2020 중첩연도 비율로 11차를 10차 수준에 맞춤
    ratio = (w10[2020] / w11[2020]).replace([np.inf, -np.inf], np.nan)
    for y in [c for c in w11.columns if c > 2020]: S[y] = w11[y] * ratio.reindex(w11.index).values
S = S.sort_index(axis=1)
kos_years = [int(c) for c in S.columns]
def kval(ks, it, y):
    try: v = S.loc[(ks, it), y]
    except KeyError: return np.nan
    return v if (v == v and v > 0) else np.nan
rows = []
# KSIC별 KR 특허(셀 합, 출원 단위는 셀 내 dedupe) — 특허 집약도용
kr_ks = pd.DataFrame(kr).groupby(ksic_c).sum()
for t0 in T0S:
    for ks in K50:
        r = dict(ksic=ks, t0=t0)
        avail = [y for y in kos_years if y <= t0 - 1]
        a_ = max(avail) if avail else None
        for it, nm in items.items():
            if a_ is None: r[f'ks_{nm}_g'] = np.nan; r[f'ks_{nm}_lvl'] = np.nan; continue
            xa = kval(ks, it, a_)
            bs = [b for b in (a_ - 3, a_ - 4, a_ - 2) if b in kos_years]
            g = np.nan
            for b in bs:
                xb = kval(ks, it, b)
                if xa == xa and xb == xb: g = np.log(xa / xb) / (a_ - b); break
            r[f'ks_{nm}_g'] = g; r[f'ks_{nm}_lvl'] = np.log(xa) if xa == xa else np.nan
        sh = kval(ks, 'T04', a_) if a_ else np.nan; va = kval(ks, 'T06', a_) if a_ else np.nan
        r['ks_va_ratio'] = va / sh if (sh == sh and va == va) else np.nan
        pat = kr_ks.loc[ks].values[yi(t0) - 2:yi(t0) + 1].sum()
        r['ks_pat_int'] = np.log((pat + 1) / sh) if sh == sh else np.nan     # 특허 3년/출하액(백만원)
        emp = kval(ks, 'T02', a_) if a_ else np.nan
        r['ks_pat_per_emp'] = np.log((pat + 1) / emp) if emp == emp else np.nan
        r['kosis_year_used'] = a_ if a_ else np.nan
        rows.append(r)
KOS = pd.DataFrame(rows)
link_k = KOS.groupby('ksic').agg(first_t0_with_growth=('ks_ship_g', lambda s: KOS.loc[s.dropna().index, 't0'].min() if s.notna().any() else np.nan))
link_k['kosis_years'] = [','.join(str(y) for y in kos_years if not np.isnan(kval(ks, 'T04', y))) for ks in link_k.index]
link_k['n_years_ship'] = [sum(1 for y in kos_years if not np.isnan(kval(ks, 'T04', y))) for ks in link_k.index]
link_k['masked_cells'] = [int(k[(k.세세분류코드 == ks)].마스킹.sum()) for ks in link_k.index]
link_k.to_csv(os.path.join(HERE, 'ext_link_kosis.csv'), encoding='utf-8-sig')

# ================= 2. 무역 (KSIC ↔ HS6, 한국 수입·수출) =================
mp = pd.read_csv(os.path.join(FUS, 'ksic_hs6_map.csv'), dtype=str)
imp = pd.read_csv(os.path.join(FUS, 'hs6_import_yearly.csv'), dtype={'hs6': str}); imp = imp[imp.year <= MAXY]
imp['hs6'] = imp.hs6.str.zfill(6)
M = imp.pivot_table(index=['hs6', 'year'], columns='partner', values='value', aggfunc='sum').reset_index()
M = M.merge(mp[['k5', 'hs6']].drop_duplicates(), on='hs6')   # 1 HS6 → 여러 KSIC 이면 각 KSIC 에 전액 (중복 허용: KSIC 규모 지표)
MK = M.groupby(['k5', 'year'])[['WLD', 'CN']].sum().reset_index()
exp_file = os.path.join(SHX, 'R2global_kr_export_hs6.csv')
EX = None
if os.path.exists(exp_file):
    ex = pd.read_csv(exp_file, dtype={'hs6': str}); ex['hs6'] = ex.hs6.str.zfill(6); ex = ex[ex.year <= MAXY]
    ex_years = sorted(ex.year.unique())
    EX = ex.merge(mp[['k5', 'hs6']].drop_duplicates(), on='hs6').groupby(['k5', 'year']).value.sum().reset_index()
def ser(df, ks, col):
    s = df[df.k5 == ks].set_index('year')[col] if df is not None else pd.Series(dtype=float)
    return s
rows = []
for t0 in T0S:
    for ks in K50:
        r = dict(ksic=ks, t0=t0)
        s = ser(MK, ks, 'WLD'); c = ser(MK, ks, 'CN')
        W = lambda x, a_, b: x.reindex(range(t0 + a_, t0 + b + 1)).sum(min_count=b - a_ + 1)
        if len(s):
            w0, w1 = W(s, -2, 0), W(s, -5, -3); c0, c1 = W(c, -2, 0), W(c, -5, -3)
            r['tr_imp_lvl'] = np.log1p(w0) if w0 == w0 else np.nan
            r['tr_imp_g'] = np.log((w0 + 1) / (w1 + 1)) if (w0 == w0 and w1 == w1) else np.nan
            r['tr_imp_cn_g'] = np.log((c0 + 1) / (c1 + 1)) if (c0 == c0 and c1 == c1) else np.nan
            r['tr_cn_sh'] = c0 / w0 if (w0 == w0 and w0 > 0) else np.nan
            r['tr_cn_sh_chg'] = r['tr_cn_sh'] - c1 / w1 if (w1 == w1 and w1 > 0 and r['tr_cn_sh'] == r['tr_cn_sh']) else np.nan
        if EX is not None:
            e = ser(EX, ks, 'value')
            if len(e) and t0 - 5 >= min(ex_years) and t0 <= max(ex_years):
                e0, e1 = W(e, -2, 0), W(e, -5, -3)
                r['tr_exp_lvl'] = np.log1p(e0) if e0 == e0 else np.nan
                r['tr_exp_g'] = np.log((e0 + 1) / (e1 + 1)) if (e0 == e0 and e1 == e1) else np.nan
                if len(s):
                    w0 = W(s, -2, 0); w1 = W(s, -5, -3)
                    r['tr_net_exp'] = (e0 - w0) / (e0 + w0) if (e0 == e0 and w0 == w0 and e0 + w0 > 0) else np.nan   # 무역특화지수
                    r['tr_net_exp_chg'] = r['tr_net_exp'] - (e1 - w1) / (e1 + w1) if (e1 == e1 and w1 == w1 and e1 + w1 > 0 and r['tr_net_exp'] == r['tr_net_exp']) else np.nan
        rows.append(r)
TR = pd.DataFrame(rows)
lk = mp.groupby('k5').agg(n_hs6=('hs6', 'nunique'), io=('io', lambda x: ','.join(sorted(set(x)))))
lk = lk.reindex(K50)
hs_multi = mp.groupby('hs6').k5.nunique()
lk['n_hs6_unique_to_ksic'] = [int((hs_multi.reindex(mp[mp.k5 == ks].hs6.unique()) == 1).sum()) if ks in set(mp.k5) else 0 for ks in K50]
lk['imp_hs6_with_data'] = [int(imp[imp.hs6.isin(mp[mp.k5 == ks].hs6)].hs6.nunique()) for ks in K50]
lk['imp_WLD_2015_usd'] = [float(MK[(MK.k5 == ks) & (MK.year == min(2015, MAXY))].WLD.sum()) for ks in K50]
if EX is not None:
    lk['exp_hs6_with_data'] = [int(ex[ex.hs6.isin(mp[mp.k5 == ks].hs6)].hs6.nunique()) for ks in K50]
    lk['exp_years'] = f'{min(ex_years)}-{max(ex_years)}'
lk.index.name = 'ksic'; lk.to_csv(os.path.join(HERE, 'ext_link_trade.csv'), encoding='utf-8-sig')

# ================= 3. 산업연관표: 전후방 연결 =================
conc = os.path.join(DATA, '외부데이터', '산업연관표', 'IO코드와 KSIC 및 HS코드 대조표.xlsx')
kk = pd.read_excel(conc, sheet_name='IO코드와 KSIC 비교표', header=1, dtype=str); kk.columns = ['io', 'nm', 'ksic']
kk['io'] = kk.io.ffill().str.strip(); kk = kk.dropna(subset=['ksic']); kk['k5'] = kk.ksic.str.strip().str[:5]
io_map = kk[kk.k5.isin(K50)][['k5', 'io']].drop_duplicates(); io_map['s3'] = io_map.io.str[:3]
fio = os.path.join(DATA, '외부데이터', '산업연관표', '2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx')
A = pd.read_excel(fio, sheet_name=f'{a.io_year}-총투입계수(A)', header=None)
codes_col = A.iloc[4, 2:].astype(str).str.strip().values; codes_row = A.iloc[6:, 0].astype(str).str.strip().values
Am = A.iloc[6:, 2:].apply(pd.to_numeric, errors='coerce'); Am.index = codes_row; Am.columns = codes_col
Am = Am.loc[[c for c in Am.index if len(c) == 3 and c.isdigit()], [c for c in Am.columns if len(c) == 3 and c.isdigit()]].fillna(0)
ks_s = io_map.groupby('k5').s3.apply(lambda x: sorted(set(x))).to_dict()
# KSIC i 의 후방(공급) 가중 u[i,j] = Σ A[s_j, s_i] (j 부문이 i 1단위 생산에 투입), 전방(수요) d[i,j] = Σ A[s_i, s_j]
U = pd.DataFrame(0.0, index=K50, columns=K50); Dn = U.copy()
for i in K50:
    for j in K50:
        if i == j or i not in ks_s or j not in ks_s: continue
        U.loc[i, j] = sum(Am.loc[sj, si] for si in ks_s[i] for sj in ks_s[j] if si in Am.index and sj in Am.columns and sj in Am.index and si in Am.columns)
        Dn.loc[i, j] = sum(Am.loc[si, sj] for si in ks_s[i] for sj in ks_s[j] if si in Am.index and sj in Am.columns)
U = U.clip(lower=0); Dn = Dn.clip(lower=0)   # 음(-) 계수(재고·보조금 조정 잔차, 최소 -0.005)는 0으로
lio = io_map.groupby('k5').agg(io_basic=('io', lambda x: ','.join(sorted(set(x)))), io_small=('s3', lambda x: ','.join(sorted(set(x))))).reindex(K50)
lio['n_upstream_ksic'] = (U > 0).sum(1).values; lio['n_downstream_ksic'] = (Dn > 0).sum(1).values
lio['up_weight_sum'] = U.sum(1).values; lio['dn_weight_sum'] = Dn.sum(1).values
lio['top_upstream'] = [','.join(U.loc[i].sort_values(ascending=False).index[:3][U.loc[i].sort_values(ascending=False).values[:3] > 0]) for i in K50]
lio['top_downstream'] = [','.join(Dn.loc[i].sort_values(ascending=False).index[:3][Dn.loc[i].sort_values(ascending=False).values[:3] > 0]) for i in K50]
lio.index.name = 'ksic'; lio.to_csv(os.path.join(HERE, 'ext_link_io.csv'), encoding='utf-8-sig')
U.to_csv(os.path.join(HERE, f'ext_io_upstream_w_{a.io_year}.csv'), encoding='utf-8-sig')

# KSIC 수준: 전후방 산업의 KR/해외 특허 모멘텀 가중평균
def g3(x, t0):
    return np.log((x[..., yi(t0) - 2:yi(t0) + 1].sum(-1) + 1) / (x[..., yi(t0) - 5:yi(t0) - 2].sum(-1) + 1))
krK = pd.DataFrame(kr).groupby(ksic_c).sum().reindex(K50); fxK = pd.DataFrame(fx).groupby(ksic_c).sum().reindex(K50)
rows = []
for t0 in T0S:
    km = pd.Series(g3(krK.values, t0), index=K50); fm = pd.Series(g3(fxK.values, t0), index=K50)
    for i in K50:
        r = dict(ksic=i, t0=t0)
        for nm, Wt in (('up', U), ('dn', Dn)):
            w = Wt.loc[i]; s = w.sum()
            r[f'io_{nm}_kmom'] = float((w * km).sum() / s) if s > 0 else np.nan
            r[f'io_{nm}_fmom'] = float((w * fm).sum() / s) if s > 0 else np.nan
        rows.append(r)
IOK = pd.DataFrame(rows)
# ================= 4. KOSIS 광공업생산지수·가동률(월별, 소분류 3자리, 2005.01~) =================
# 월별 지수는 다음 달 공표 → 원점 t0 에서는 t0년 12월까지 사용. 2020=100 재기준 계열(과거 소급 개편값)이라는 한계는 보고서에 명시.
GI = os.path.join(DATA, '외부데이터', '경기지표')
pi = pd.read_csv(os.path.join(GI, 'kosis_광공업생산지수_월별.csv'), dtype={'산업코드': str}); pi['y'] = pi.연월 // 100; pi = pi[pi.y <= MAXY]
pi['값'] = pd.to_numeric(pi.값, errors='coerce')
PA = pi.groupby(['산업코드', '항목코드', 'y']).값.mean()
ut = pd.read_csv(os.path.join(GI, 'kosis_가동률지수_소분류.csv'), dtype={'C1': str}); ut['y'] = ut.PRD_DE // 100; ut = ut[ut.y <= MAXY]
ut['DT'] = pd.to_numeric(ut.DT, errors='coerce'); UA = ut.groupby(['C1', 'y']).DT.mean()
def pv(code, it, y):
    try: return PA.loc[(code, it, y)]
    except KeyError: return np.nan
def uv(code, y):
    try: return UA.loc[(code, y)]
    except KeyError: return np.nan
rows = []; link_pi = []
for ks in K50:
    c3 = 'C' + ks[:3]; c2 = 'C' + ks[:2]
    code = c3 if (c3, 'T10', 2010) in PA.index else c2
    ucode = c3 if (c3, 2010) in UA.index else c2
    link_pi.append(dict(ksic=ks, prod_index_code=code, util_code=ucode, level='소분류3' if code == c3 else '중분류2'))
    for t0 in T0S:
        r = dict(ksic=ks, t0=t0)
        p = lambda it, y: pv(code, it, y)
        a2 = (p('T10', t0) + p('T10', t0 - 1)) / 2; b2 = (p('T10', t0 - 2) + p('T10', t0 - 3)) / 2
        r['pi_prod_g2'] = np.log(a2 / b2) if (a2 == a2 and b2 == b2) else np.nan
        r['pi_prod_g1'] = np.log(p('T10', t0) / p('T10', t0 - 1)) if (p('T10', t0) == p('T10', t0) and p('T10', t0 - 1) == p('T10', t0 - 1)) else np.nan
        s2 = (p('T11', t0) + p('T11', t0 - 1)) / 2; s2b = (p('T11', t0 - 2) + p('T11', t0 - 3)) / 2
        r['pi_ship_g2'] = np.log(s2 / s2b) if (s2 == s2 and s2b == s2b) else np.nan
        r['pi_inv_ship'] = np.log(p('T12', t0) / p('T11', t0)) if (p('T12', t0) == p('T12', t0) and p('T11', t0) == p('T11', t0)) else np.nan
        u0 = uv(ucode, t0); u1 = uv(ucode, t0 - 2)
        g1p = np.log(p('T10', t0 - 1) / p('T10', t0 - 2)) if (p('T10', t0 - 1) == p('T10', t0 - 1) and p('T10', t0 - 2) == p('T10', t0 - 2)) else np.nan
        r['pi_prod_acc'] = r['pi_prod_g1'] - g1p if (r['pi_prod_g1'] == r['pi_prod_g1'] and g1p == g1p) else np.nan
        r['pi_util'] = u0; r['pi_util_chg'] = u0 - u1 if (u0 == u0 and u1 == u1) else np.nan
        rows.append(r)
PIK = pd.DataFrame(rows)
# 제조업 전체(C) 대비 상대 성장 (원점별 경기 제거)
for t0 in T0S:
    a2 = (pv('C', 'T10', t0) + pv('C', 'T10', t0 - 1)) / 2; b2 = (pv('C', 'T10', t0 - 2) + pv('C', 'T10', t0 - 3)) / 2
    PIK.loc[PIK.t0 == t0, 'pi_prod_g2_rel'] = PIK.loc[PIK.t0 == t0, 'pi_prod_g2'] - (np.log(a2 / b2) if (a2 == a2 and b2 == b2) else np.nan)
# 산업연관 × 생산지수: 전방(수요) 산업의 생산 성장 = 수요 견인, 후방(공급) 산업의 생산 성장
for t0 in T0S:
    g = PIK[PIK.t0 == t0].set_index('ksic').pi_prod_g2.reindex(K50)
    for nm, Wt in (('up', U), ('dn', Dn)):
        W_ = Wt.values * g.notna().values[None, :]
        v = (W_ * np.nan_to_num(g.values)[None, :]).sum(1) / np.where(W_.sum(1) > 0, W_.sum(1), np.nan)
        PIK.loc[PIK.t0 == t0, f'io_{nm}_pi_g2'] = pd.Series(v, index=K50).reindex(PIK.loc[PIK.t0 == t0, 'ksic']).values
# 내수·수출 출하지수(중분류 2자리, C26은 소분류 3자리): 수출출하 비중 변화
ex_ = pd.read_csv(os.path.join(GI, 'kosis_내수수출출하지수.csv'), dtype={'C1': str}); ex_['y'] = ex_.PRD_DE // 100; ex_ = ex_[ex_.y <= MAXY]
ex_['DT'] = pd.to_numeric(ex_.DT, errors='coerce'); XA = ex_.groupby(['C1', 'ITM_ID', 'y']).DT.mean()
def xv(code, it, y):
    try: return XA.loc[(code, it, y)]
    except KeyError: return np.nan
for i, r in PIK.iterrows():
    ks, t0 = r.ksic, r.t0; code = 'C' + ks[:3] if ('C' + ks[:3], 'T11', 2010) in XA.index else 'C' + ks[:2]
    d0, e0, d1, e1 = xv(code, 'T10', t0), xv(code, 'T11', t0), xv(code, 'T10', t0 - 2), xv(code, 'T11', t0 - 2)
    PIK.loc[i, 'pi_exp_dom_g'] = np.log(e0 / e1) - np.log(d0 / d1) if all(v == v for v in (d0, e0, d1, e1)) else np.nan
pd.DataFrame(link_pi).to_csv(os.path.join(HERE, 'ext_link_prodindex.csv'), index=False, encoding='utf-8-sig')
E = KOS.merge(TR, on=['ksic', 't0'], how='outer').merge(IOK, on=['ksic', 't0'], how='outer').merge(PIK, on=['ksic', 't0'], how='outer')
# 무역·KOSIS 결합: 수입침투 변화(수입 성장 − 출하 성장, 연율 비교)
E['tr_imp_minus_ship'] = E.tr_imp_g / 3 - E.ks_ship_g
# ================= 5. 5개국 수출 경쟁(KR·CN·JP·US·DE, HS6→KSIC): 한국 수출 점유율 =================
# _shared/ext/R2ind_comtrade_exports_5c_hs6_2004_2021.csv (다른 트랙 수집). 5개국 모두 완결된 연도만 사용(청크 부분 수집 연도 제외).
f5 = os.path.join(SHX, 'R2ind_comtrade_exports_5c_hs6_2004_2021.csv')
if os.path.exists(f5):
    x5 = pd.read_csv(f5, dtype={'hs6': str}); x5['hs6'] = x5.hs6.str.zfill(6); x5 = x5[x5.year <= MAXY]
    cov = x5.groupby(['year', 'reporter']).hs6.nunique().unstack()
    full_years = [int(y) for y in cov.index if cov.loc[y].min() >= 0.85 * len(set(mp.hs6)) and cov.loc[y].notna().all()]
    x5 = x5[x5.year.isin(full_years)].merge(mp[['k5', 'hs6']].drop_duplicates(), on='hs6')
    X5 = x5.groupby(['k5', 'year', 'reporter']).export_usd.sum().unstack().fillna(0)
    X5['tot'] = X5.sum(1); X5['kr_sh'] = X5[410] / X5.tot; X5['cn_sh'] = X5[156] / X5.tot
    krall = X5.groupby(level='year')[410].sum() / X5.groupby(level='year').tot.sum()
    # KSIC 수준 KR 특허 점유율(5개국 출원 중 KR, t0-2..t0) — 기술-수출 괴리용
    totK = pd.DataFrame(tot).groupby(ksic_c).sum().reindex(K50)
    rows = []
    for t0 in T0S:
        av = [y for y in full_years if y <= t0]
        for ks in K50:
            r = dict(ksic=ks, t0=t0)
            if av and (ks, av[-1]) in X5.index:
                a_ = av[-1]; r['ex5_kr_sh'] = X5.loc[(ks, a_), 'kr_sh']; r['ex5_kr_rca'] = np.log((X5.loc[(ks, a_), 'kr_sh'] + 1e-4) / krall[a_])
                ps = krK.loc[ks].values[yi(t0) - 2:yi(t0) + 1].sum() / max(totK.loc[ks].values[yi(t0) - 2:yi(t0) + 1].sum(), 1)
                r['ex5_pat_gap'] = np.log((ps + 1e-3) / (r['ex5_kr_sh'] + 1e-3))
                if len(av) >= 2 and (ks, av[-2]) in X5.index:
                    b_ = av[-2]
                    r['ex5_kr_sh_chg'] = (X5.loc[(ks, a_), 'kr_sh'] - X5.loc[(ks, b_), 'kr_sh']) / (a_ - b_)
                    r['ex5_cn_sh_chg'] = (X5.loc[(ks, a_), 'cn_sh'] - X5.loc[(ks, b_), 'cn_sh']) / (a_ - b_)
                r['ex5_year_used'] = a_
            rows.append(r)
    EX5 = pd.DataFrame(rows); E = E.merge(EX5, on=['ksic', 't0'], how='left')
    pd.DataFrame(dict(full_years=[','.join(map(str, full_years))], hs6_linked=[x5.hs6.nunique()], ksic_linked=[x5.k5.nunique()])).to_csv(
        os.path.join(HERE, 'ext_link_export5c.csv'), index=False, encoding='utf-8-sig')
    print('5개국 수출 완결연도', full_years)
E.to_parquet(os.path.join(HERE, f'ext_ksic_t0{a.out_suffix}.parquet'))

# 셀 수준: 같은 IPC 메인그룹이 전후방 산업(KSIC)에서 얼마나 성장하나 (기술이 공급망을 따라 이동)
cid = pd.DataFrame(dict(cell=cells, ksic=ksic_c, mg=mg_c))
rows = []
for t0 in T0S:
    km = g3(kr, t0); fm = g3(fx, t0); kw = kr[:, yi(t0) - 2:yi(t0) + 1].sum(1); fw = fx[:, yi(t0) - 2:yi(t0) + 1].sum(1)
    X = cid.assign(km=km, fm=fm, kw=kw, fw=fw)
    piv = {c: X.pivot_table(index='mg', columns='ksic', values=c, aggfunc='first').reindex(columns=K50) for c in ('km', 'fm', 'kw', 'fw')}
    out = {}
    for nm, Wt in (('up', U), ('dn', Dn)):
        Wm = Wt.values  # [i, j]
        pres = piv['kw'].notna().values.astype(float)                  # mg×j 존재
        wsum = pres @ Wm.T                                              # mg×i: Σ_j w_ij·존재
        for c in ('km', 'fm'):
            v = np.nan_to_num(piv[c].values) @ Wm.T
            out[f'io_{nm}_cell_{c}'] = pd.DataFrame(np.where(wsum > 0, v / np.maximum(wsum, 1e-12), np.nan), index=piv['km'].index, columns=K50)
        vk = np.nan_to_num(piv['kw'].values) @ Wm.T
        out[f'io_{nm}_cell_klvl'] = pd.DataFrame(np.log1p(vk), index=piv['km'].index, columns=K50)
    long = None
    for key, df in out.items():
        s = df.stack(future_stack=True).rename(key)
        long = s.to_frame() if long is None else long.join(s)
    long = long.reset_index(); long.columns = ['mg', 'ksic'] + list(out.keys())
    long['cell'] = long.ksic + '|' + long.mg; long = long[long.cell.isin(set(cells))]
    long['t0'] = t0; rows.append(long.drop(columns=['mg', 'ksic']))
CE = pd.concat(rows, ignore_index=True)
CE.to_parquet(os.path.join(HERE, f'ext_cell_t0{a.out_suffix}.parquet'))
print('KOSIS', KOS.shape, 'TRADE', TR.shape, 'IO', IOK.shape, 'CELL', CE.shape)
print(E.groupby('t0')[['ks_ship_g', 'tr_imp_g', 'tr_exp_g' if 'tr_exp_g' in E else 'tr_imp_g', 'io_up_kmom']].apply(lambda d: d.notna().mean()).round(2).to_string())
