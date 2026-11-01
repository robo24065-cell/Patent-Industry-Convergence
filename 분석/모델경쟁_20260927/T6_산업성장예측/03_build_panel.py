# 패널 구축: (KSIC 5자리, 목표연도 T) 행. 모든 피처는 T-1 이전 자료만.
# 1단계 안전장치: KOSIS 광업제조업조사는 연도<=2019만 로드(로드 즉시 필터), 외부 월별/연간 자료는 <=2018만 사용.
import pandas as pd, numpy as np
D = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/공모전데이터"
MAXY = 2019
k = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_2008_2024.csv', dtype={'세세분류코드': str})
k = k[(k.연도 <= MAXY) & (k.기준 == '10차')].copy()
k50 = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
k50 = k50[k50.연도 <= MAXY]
S50 = sorted(k50.세세분류코드.unique())
item = {'T01': 'estab', 'T02': 'emp', 'T03': 'wage', 'T04': 'ship', 'T05': 'cost', 'T06': 'va', 'T07': 'asset'}
k['v'] = k.항목코드.map(item)
w = k.pivot_table(index=['세세분류코드', '연도'], columns='v', values='값', aggfunc='first').reset_index()
w.columns.name = None
w = w.rename(columns={'세세분류코드': 'ksic', '연도': 'year'})
codes = w.ksic.unique()
grid = pd.MultiIndex.from_product([codes, range(2008, MAXY + 1)], names=['ksic', 'year']).to_frame(index=False)
w = grid.merge(w, on=['ksic', 'year'], how='left').sort_values(['ksic', 'year']).reset_index(drop=True)
for c in item.values():
    w['l_' + c] = np.log(w[c].where(w[c] > 0))


def growth(col):
    g = w.groupby('ksic')[col].diff()
    two = (w[col] - w.groupby('ksic')[col].shift(2)) / 2  # 2011: (2011-2009)/2
    g = pd.Series(np.where(w.year == 2011, two, g), index=w.index)
    nxt = g.groupby(w.ksic).shift(-1)
    g[w.year == 2010] = nxt[w.year == 2010]  # 2010 = 2011과 같은 연율화 값(시차 피처 연속성용)
    return g


for c in item.values():
    w['g_' + c] = growth('l_' + c)
w['y'] = w['g_ship']
w['imputed_2y'] = w.year.isin([2010, 2011]).astype(int)

# 상위 3·2자리 동료 성장(자기 제외, 합계 기준)
w['k3'] = w.ksic.str[:3]
w['k2'] = w.ksic.str[:2]
w['k4'] = w.ksic.str[:4]
prev = w.groupby('ksic')['ship'].shift(1)
prev2 = w.groupby('ksic')['ship'].shift(2)
w['ship_prev'] = np.where(w.year == 2011, prev2, prev)
ok = w.ship.notna() & w.ship_prev.notna() & (w.ship > 0) & (w.ship_prev > 0)
for lv in ['k4', 'k3', 'k2']:
    a = w[ok].groupby([lv, 'year'])[['ship', 'ship_prev']].sum().rename(columns={'ship': 'S', 'ship_prev': 'P'})
    t = w.join(a, on=[lv, 'year'])
    S = t.S - np.where(ok, w.ship, 0)
    P = t.P - np.where(ok, w.ship_prev, 0)
    div = np.where(w.year == 2011, 2, 1)
    pg = pd.Series(np.log(S.where(S > 0) / P.where(P > 0)) / div, index=w.index)
    nxt = pg.groupby(w.ksic).shift(-1)
    pg[w.year == 2010] = nxt[w.year == 2010]
    w['peer_' + lv] = pg
w['in50'] = w.ksic.isin(S50).astype(int)
w = w.join(w.groupby('year').y.mean().rename('cm_all'), on='year')
w = w.join(w[w.in50 == 1].groupby('year').y.mean().rename('cm_50'), on='year')
w = w.join(w.groupby('year').y.median().rename('cmed_all'), on='year')

# ---- 피처(목표연도 T 행에 T-1 이전 값) ----
F = pd.DataFrame({'ksic': w.ksic, 'year': w.year, 'y': w.y, 'in50': w.in50, 'imputed_2y': w.imputed_2y,
                  'k3': w.k3, 'k2': w.k2, 'ship': w.ship})
gs = w.groupby('ksic')
for L in [1, 2, 3]:
    F[f'g{L}'] = gs['y'].shift(L)
for c in ['va', 'emp', 'estab', 'asset', 'wage', 'cost']:
    F[f'g{c}1'] = gs['g_' + c].shift(1)
F['lsize1'] = gs['l_ship'].shift(1)
F['va_ratio1'] = gs['l_va'].shift(1) - gs['l_ship'].shift(1)
F['cost_ratio1'] = gs['l_cost'].shift(1) - gs['l_ship'].shift(1)
F['asset_int1'] = gs['l_asset'].shift(1) - gs['l_ship'].shift(1)
F['wage_pe1'] = gs['l_wage'].shift(1) - gs['l_emp'].shift(1)
F['peer3_1'] = gs['peer_k3'].shift(1)
F['peer2_1'] = gs['peer_k2'].shift(1)
F['peer4_1'] = gs['peer_k4'].shift(1)
F['peer3_2'] = gs['peer_k3'].shift(2)
# 4자리 동료 존재 여부(자기 외 형제 산업 수)
F['n_sib4'] = w.groupby(['k4', 'year']).ksic.transform('count').groupby(w.ksic).shift(1) - 1
F['cm_all1'] = gs['cm_all'].shift(1)
F['cm_501'] = gs['cm_50'].shift(1)
F['cmed_all1'] = gs['cmed_all'].shift(1)
F['gmean_exp'] = gs['y'].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
F['gsd_exp'] = gs['y'].transform(lambda s: s.shift(1).expanding(min_periods=2).std())
F['g3avg'] = gs['y'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
F['ldev3'] = gs['l_ship'].transform(lambda s: s.shift(1) - s.shift(1).rolling(3, min_periods=2).mean())

# ---- 광공업생산지수(3자리·2자리·제조업, 월별) : T-1 연간 성장, T-1 4분기 전년동기비, 재고/출하 변화 ----
e = pd.read_csv(f'{D}/외부데이터/경기지표/kosis_광공업생산지수_월별.csv', dtype={'산업코드': str})
e = e[e.연월 <= 201812].copy()
e['값'] = pd.to_numeric(e['값'], errors='coerce')
e['yy'] = e.연월 // 100
e['mm'] = e.연월 % 100
e['it'] = e.항목코드.map({'T10': 'prod', 'T11': 'shipi', 'T12': 'inv'})
e = e.dropna(subset=['it'])


def idxfeat(x):
    a = x.groupby(['산업코드', 'it', 'yy']).값.mean().unstack('it')
    q4 = x[x.mm >= 10].groupby(['산업코드', 'it', 'yy']).값.mean().unstack('it')
    dec = x[x.mm == 12].groupby(['산업코드', 'it', 'yy']).값.mean().unstack('it')
    out = pd.DataFrame(index=a.index)
    la, lq = np.log(a), np.log(q4)
    g0 = lambda s: s - s.groupby(level=0).shift(1)
    out['ip_g'] = g0(la['prod'])
    out['is_g'] = g0(la['shipi'])
    out['ip_q4'] = g0(lq['prod'])
    out['is_q4'] = g0(lq['shipi'])
    if 'inv' in dec:
        out['inv_ship'] = g0(np.log(dec['inv'] / q4['shipi']))
    return out.reset_index().rename(columns={'산업코드': 'icode', 'yy': 'fy'})


i3 = idxfeat(e[e.산업코드.str.len() == 4])
i2 = idxfeat(e[e.산업코드.str.len() == 3])
tot = idxfeat(e[e.산업코드 == 'C'])
mining = F.ksic.str[:2].astype(int) <= 8
F['icode3'] = np.where(mining, 'B', 'C') + F.ksic.str[:3]
F['icode2'] = np.where(mining, 'B', 'C') + F.ksic.str[:2]
F['fy'] = F.year - 1
F = F.merge(i3.rename(columns={c: c + '3' for c in i3.columns if c not in ('icode', 'fy')}).rename(columns={'icode': 'icode3'}),
            on=['icode3', 'fy'], how='left')
F = F.merge(i2.rename(columns={c: c + '2' for c in i2.columns if c not in ('icode', 'fy')}).rename(columns={'icode': 'icode2'}),
            on=['icode2', 'fy'], how='left')
F = F.merge(tot.drop(columns='icode').rename(columns={c: 'mfg_' + c for c in tot.columns if c not in ('icode', 'fy')}),
            on='fy', how='left')
# ---- ECOS BSI 전망(제조업, T-1년 12월 조사) ----
b = pd.read_csv(f'{D}/외부데이터/경기지표/ecos_BSI_전망.csv')
b = b[(b.ITEM_NAME1 == '제 조 업') & (b.TIME % 100 == 12) & (b.TIME <= 201812)].copy()
b['fy'] = b.TIME // 100
bb = b.pivot_table(index='fy', columns='ITEM_NAME2', values='DATA_VALUE', aggfunc='first')
bb = bb[['매출전망BSI 2)', '업황전망BSI 1)', '수출전망BSI 2)']]
bb.columns = ['bsi_sales', 'bsi_biz', 'bsi_exp']
F = F.merge(bb.reset_index(), on='fy', how='left')
# ---- 수입점유비(2자리 업종, T-1 수준과 변화) ----
m = pd.read_csv(f'{D}/외부데이터/제3축/수입점유비_업종별.csv')
m = m[m.PRD_DE <= 2018].rename(columns={'PRD_DE': 'fy', 'DT': 'imp_share', 'C1': 'icode2'})[['icode2', 'fy', 'imp_share']]
m['imp_share'] = pd.to_numeric(m['imp_share'], errors='coerce')
m = m.sort_values(['icode2', 'fy'])
m['imp_share_d'] = m.groupby('icode2').imp_share.diff()
F = F.merge(m, on=['icode2', 'fy'], how='left').drop(columns=['fy'])
F.to_parquet('cache/panel_base.parquet', index=False)
F50 = F[F.in50 == 1]
print('rows', len(F), 'industries', F.ksic.nunique(), '50-set rows', len(F50), '50 industries', F50.ksic.nunique())
print(pd.concat([F.groupby('year').y.count().rename('all'), F50.groupby('year').y.count().rename('in50')], axis=1).to_string())
v = F50[F50.year.between(2012, 2019)]
print(v.describe().T[['count', 'mean', 'std', 'min', 'max']].round(3).to_string())
