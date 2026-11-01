# -*- coding: utf-8 -*-
"""T6 산업 성장 예측 — 동결 실행기 (2단계 감사·개선 후 고정)

사용:
  python final_run.py --mode dryrun    # 검증 구간(목표 2012-2019, 자료 <=2019)으로 학습·평가 → dryrun_result.json
  python final_run.py --mode holdout   # 🔒 3단계 평가자 전용: 목표 2021-2024 1년 앞 롤링 → holdout_result.json
  python final_run.py --mode future    # 전체 자료로 재학습 → 2025(주)·2026(보조) 전망 → future_forecast.csv

모형 정의·하이퍼파라미터는 final_spec.json 에서 읽는다(기본 경로: 이 파일 옆).
원칙: 원점 T의 학습 레이블은 목표연도 < T 만, 피처는 모두 T-1 이전 자료(조사 T-1, 월별 지수 T-1년 12월까지).
KSIC 차수: 10차(2008-2020)와 11차(2020-2024)를 2020 중복 연도로 연결한다.
  - y_2020 = ln(10차 2020 / 10차 2019), y_t(t>=2021) = ln(11차 t / 11차 t-1)
  - 수준 피처(ldev3)는 10차 2020 수준에 11차 변화분을 이어 붙인 연쇄 로그수준을 쓴다.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
D = ROOT + "/공모전데이터"
SHARED = ROOT + "/분석/모델경쟁_20260927/_shared"
OV = 2020  # 10차·11차 중복 연도
ITEM = {'T01': 'estab', 'T02': 'emp', 'T03': 'wage', 'T04': 'ship', 'T05': 'cost', 'T06': 'va', 'T07': 'asset'}
VAL_YEARS = list(range(2012, 2020))
HOLDOUT_YEARS = [2021, 2022, 2023, 2024]


# ============================================================ 1. 광업제조업조사 패널
def _basis_panel(k, years, annualize_2011):
    """한 KSIC 차수의 (산업, 연도) 로그수준·성장·동료성장."""
    k = k.copy()
    k['v'] = k.항목코드.map(ITEM)
    w = k.pivot_table(index=['세세분류코드', '연도'], columns='v', values='값', aggfunc='first').reset_index()
    w.columns.name = None
    w = w.rename(columns={'세세분류코드': 'ksic', '연도': 'year'})
    codes = w.ksic.unique()
    grid = pd.MultiIndex.from_product([codes, list(years)], names=['ksic', 'year']).to_frame(index=False)
    w = grid.merge(w, on=['ksic', 'year'], how='left').sort_values(['ksic', 'year']).reset_index(drop=True)
    for c in ITEM.values():
        if c not in w:
            w[c] = np.nan
        w['l_' + c] = np.log(w[c].where(w[c] > 0))

    def growth(col):
        g = w.groupby('ksic')[col].diff()
        if annualize_2011:
            two = (w[col] - w.groupby('ksic')[col].shift(2)) / 2  # 2011: (2011-2009)/2 (2010 경제총조사로 결측)
            g = pd.Series(np.where(w.year == 2011, two, g), index=w.index)
            nxt = g.groupby(w.ksic).shift(-1)
            g[w.year == 2010] = nxt[w.year == 2010]  # 2010 행은 시차 피처 연속성용(학습 레이블로 안 씀)
        return g

    for c in ITEM.values():
        w['g_' + c] = growth('l_' + c)
    w['k4'], w['k3'], w['k2'] = w.ksic.str[:4], w.ksic.str[:3], w.ksic.str[:2]
    prev = w.groupby('ksic')['ship'].shift(1)
    prev2 = w.groupby('ksic')['ship'].shift(2)
    w['ship_prev'] = np.where((w.year == 2011) & annualize_2011, prev2, prev)
    ok = w.ship.notna() & w.ship_prev.notna() & (w.ship > 0) & (w.ship_prev > 0)
    for lv in ['k4', 'k3', 'k2']:
        a = w[ok].groupby([lv, 'year'])[['ship', 'ship_prev']].sum().rename(columns={'ship': 'S', 'ship_prev': 'P'})
        t = w.join(a, on=[lv, 'year'])
        S = t.S - np.where(ok, w.ship, 0)
        P = t.P - np.where(ok, w.ship_prev, 0)
        div = np.where((w.year == 2011) & annualize_2011, 2, 1)
        pg = pd.Series(np.log(S.where(S > 0) / P.where(P > 0)) / div, index=w.index)
        if annualize_2011:
            nxt = pg.groupby(w.ksic).shift(-1)
            pg[w.year == 2010] = nxt[w.year == 2010]
        w['peer_' + lv] = pg
    return w


def load_survey(survey_maxy, rows_maxy, k_all=None, k50_codes=None, ov=OV):
    """10차(<=ov)·11차(>ov) 연결 패널. survey_maxy 이후 조사값은 로드 즉시 제거."""
    if k_all is None:
        k_all = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_2008_2024.csv', dtype={'세세분류코드': str})
    k_all = k_all[k_all.연도 <= survey_maxy].copy()  # 🔒 안전장치: 기준시점 이후 조사값은 메모리에도 올리지 않음
    if k50_codes is None:
        k50 = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
        k50_codes = sorted(k50.세세분류코드.unique())
    k10 = k_all[(k_all.기준 == '10차') & (k_all.연도 <= ov)]
    w10 = _basis_panel(k10, range(2008, min(survey_maxy, ov) + 1), True)
    parts = [w10]
    if survey_maxy > ov:
        k11 = k_all[(k_all.기준 == '11차') & (k_all.연도 >= ov)]
        w11 = _basis_panel(k11, range(ov, survey_maxy + 1), False)
        # 11차 ov 행에서 성장은 NaN(전년 없음). 10차에 없는 코드의 ov 수준만 보충.
        add_ov = w11[(w11.year == ov) & ~w11.ksic.isin(w10.ksic.unique())]
        top = w11[w11.year > ov].copy()
        l10 = w10[w10.year == ov].set_index('ksic').l_ship
        l11 = w11[w11.year == ov].set_index('ksic').l_ship
        off = (l10 - l11).reindex(top.ksic.values).fillna(0).values
        top['l_ship_chain'] = top.l_ship.values + off
        parts += [add_ov, top]
    w = pd.concat(parts, ignore_index=True)
    if 'l_ship_chain' not in w:
        w['l_ship_chain'] = np.nan
    w['l_ship_chain'] = w.l_ship_chain.fillna(w.l_ship)
    codes = w.ksic.unique()
    grid = pd.MultiIndex.from_product([codes, range(2008, rows_maxy + 1)], names=['ksic', 'year']).to_frame(index=False)
    w = grid.merge(w, on=['ksic', 'year'], how='left').sort_values(['ksic', 'year']).reset_index(drop=True)
    w['k4'], w['k3'], w['k2'] = w.ksic.str[:4], w.ksic.str[:3], w.ksic.str[:2]
    w['y'] = w['g_ship']
    w['imputed_2y'] = w.year.isin([2010, 2011]).astype(int)
    w['in50'] = w.ksic.isin(k50_codes).astype(int)
    return w


# ============================================================ 2. 외부 월별·연간 지표
def _idxfeat(x, val='값', code='산업코드', ym='연월', items=None):
    x = x.copy()
    x['yy'], x['mm'] = x[ym] // 100, x[ym] % 100
    a = x.groupby([code, 'it', 'yy'])[val].mean().unstack('it')
    q4 = x[x.mm >= 10].groupby([code, 'it', 'yy'])[val].mean().unstack('it')
    dec = x[x.mm == 12].groupby([code, 'it', 'yy'])[val].mean().unstack('it')
    out = pd.DataFrame(index=a.index)
    la, lq = np.log(a), np.log(q4)
    g0 = lambda s: s - s.groupby(level=0).shift(1)
    for it, nm in (items or {'prod': 'ip', 'shipi': 'is'}).items():
        if it in la:
            out[nm + '_g'] = g0(la[it])
            out[nm + '_q4'] = g0(lq[it])
    if 'inv' in dec and 'shipi' in q4:
        out['inv_ship'] = g0(np.log(dec['inv'] / q4['shipi']))
    return out.reset_index().rename(columns={code: 'icode', 'yy': 'fy'})


def add_external(F, rows_maxy, extra=True):
    cut_m = (rows_maxy - 1) * 100 + 12  # 목표연도 T의 피처는 T-1년 12월까지 → 최대 rows_maxy-1
    mining = F.ksic.str[:2].astype(int) <= 8
    F['icode3'] = np.where(mining, 'B', 'C') + F.ksic.str[:3]
    F['icode2'] = np.where(mining, 'B', 'C') + F.ksic.str[:2]
    F['fy'] = F.year - 1
    e = pd.read_csv(f'{D}/외부데이터/경기지표/kosis_광공업생산지수_월별.csv', dtype={'산업코드': str})
    e = e[e.연월 <= cut_m].copy()
    e['값'] = pd.to_numeric(e['값'], errors='coerce')
    e['it'] = e.항목코드.map({'T10': 'prod', 'T11': 'shipi', 'T12': 'inv'})
    e = e.dropna(subset=['it'])
    i3 = _idxfeat(e[e.산업코드.str.len() == 4])
    i2 = _idxfeat(e[e.산업코드.str.len() == 3])
    tot = _idxfeat(e[e.산업코드 == 'C'])
    F = F.merge(i3.rename(columns={c: c + '3' for c in i3.columns if c not in ('icode', 'fy')}).rename(columns={'icode': 'icode3'}),
                on=['icode3', 'fy'], how='left')
    F = F.merge(i2.rename(columns={c: c + '2' for c in i2.columns if c not in ('icode', 'fy')}).rename(columns={'icode': 'icode2'}),
                on=['icode2', 'fy'], how='left')
    F = F.merge(tot.drop(columns='icode').rename(columns={c: 'mfg_' + c for c in tot.columns if c not in ('icode', 'fy')}),
                on='fy', how='left')
    b = pd.read_csv(f'{D}/외부데이터/경기지표/ecos_BSI_전망.csv')
    b = b[(b.ITEM_NAME1 == '제 조 업') & (b.TIME % 100 == 12) & (b.TIME <= cut_m)].copy()
    b['fy'] = b.TIME // 100
    bb = b.pivot_table(index='fy', columns='ITEM_NAME2', values='DATA_VALUE', aggfunc='first')
    bb = bb[['매출전망BSI 2)', '업황전망BSI 1)', '수출전망BSI 2)']]
    bb.columns = ['bsi_sales', 'bsi_biz', 'bsi_exp']
    F = F.merge(bb.reset_index(), on='fy', how='left')
    m = pd.read_csv(f'{D}/외부데이터/제3축/수입점유비_업종별.csv')
    m = m[m.PRD_DE <= rows_maxy - 1].rename(columns={'PRD_DE': 'fy', 'DT': 'imp_share', 'C1': 'icode2'})[['icode2', 'fy', 'imp_share']]
    m['imp_share'] = pd.to_numeric(m['imp_share'], errors='coerce')
    m = m.sort_values(['icode2', 'fy'])
    m['imp_share_d'] = m.groupby('icode2').imp_share.diff()
    F = F.merge(m, on=['icode2', 'fy'], how='left')
    if extra:
        # 가동률지수(소분류 3자리, 없으면 2자리): T-1 연간·4분기 전년동기비
        cu = pd.read_csv(f'{D}/외부데이터/경기지표/kosis_가동률지수_소분류.csv', dtype={'C1': str})
        cu = cu[cu.PRD_DE <= cut_m].rename(columns={'PRD_DE': '연월', 'C1': '산업코드', 'DT': '값'})
        cu['값'] = pd.to_numeric(cu['값'], errors='coerce')
        cu['it'] = 'cu'
        c = _idxfeat(cu, items={'cu': 'cu'})
        c3 = c.rename(columns={'icode': 'icode3', 'cu_g': 'cu_g3', 'cu_q4': 'cu_q43'})
        c2 = c.rename(columns={'icode': 'icode2', 'cu_g': 'cu_g2', 'cu_q4': 'cu_q42'})
        F = F.merge(c3, on=['icode3', 'fy'], how='left').merge(c2, on=['icode2', 'fy'], how='left')
        F['cu_q4x'] = F.cu_q43.fillna(F.cu_q42)
        F['cu_gx'] = F.cu_g3.fillna(F.cu_g2)
        # 내수·수출 출하지수(2자리, C26은 3자리): 수출 4분기 모멘텀, 수출/내수 비율 변화
        ex = pd.read_csv(f'{D}/외부데이터/경기지표/kosis_내수수출출하지수.csv', dtype={'C1': str})
        ex = ex[ex.PRD_DE <= cut_m].rename(columns={'PRD_DE': '연월', 'C1': '산업코드', 'DT': '값'})
        ex['값'] = pd.to_numeric(ex['값'], errors='coerce')
        ex['it'] = ex.ITM_ID.map({'T10': 'dom', 'T11': 'exp'})
        x = _idxfeat(ex, items={'dom': 'dom', 'exp': 'exp'})
        x['expsh_g'] = x.exp_g - x.dom_g
        x3 = x.rename(columns={'icode': 'icode3'}).rename(columns={k: k + '3' for k in ['dom_g', 'dom_q4', 'exp_g', 'exp_q4', 'expsh_g']})
        x2 = x.rename(columns={'icode': 'icode2'}).rename(columns={k: k + '2' for k in ['dom_g', 'dom_q4', 'exp_g', 'exp_q4', 'expsh_g']})
        F = F.merge(x3, on=['icode3', 'fy'], how='left').merge(x2, on=['icode2', 'fy'], how='left')
        for k in ['dom_q4', 'exp_q4', 'expsh_g']:
            F[k + 'x'] = F[k + '3'].fillna(F[k + '2'])
        # ECOS 기업경영분석 성장성(종합): 3자리 없으면 2자리, T-1년 값
        g = pd.read_csv(f'{D}/외부데이터/재무/ecos_501Y005_성장성지표.csv', dtype={'ITEM_CODE1': str})
        g = g[(g.ITEM_CODE2 == 'A') & (g.TIME <= rows_maxy - 1)]
        g = g.pivot_table(index=['ITEM_CODE1', 'TIME'], columns='ITEM_NAME3', values='DATA_VALUE', aggfunc='first').reset_index()
        g = g.rename(columns={'ITEM_CODE1': 'ic', 'TIME': 'fy', '매출액증가율': 'fin_sales', '재고자산증가율': 'fin_inv', '유형자산증가율': 'fin_tang'})
        g = g[['ic', 'fy', 'fin_sales', 'fin_inv', 'fin_tang']]
        g3 = g.rename(columns={'ic': 'icode3', 'fin_sales': 'fs3', 'fin_inv': 'fi3', 'fin_tang': 'ft3'})
        g2 = g.rename(columns={'ic': 'icode2', 'fin_sales': 'fs2', 'fin_inv': 'fi2', 'fin_tang': 'ft2'})
        F = F.merge(g3, on=['icode3', 'fy'], how='left').merge(g2, on=['icode2', 'fy'], how='left')
        F['fin_sales'] = F.fs3.fillna(F.fs2)
        F['fin_inv'] = F.fi3.fillna(F.fi2)
        F['fin_tang'] = F.ft3.fillna(F.ft2)
        F['fin_inv_gap'] = F.fin_inv - F.fin_sales
    return F.drop(columns=['fy'])


# ============================================================ 3. 피처 패널
def build_panel(survey_maxy, rows_maxy, lag=1, extra=True, k_all=None, k50_codes=None, ov=OV):
    """(KSIC 5자리, 목표연도) 행. 조사 피처는 lag년 전(기본 T-1), 월별·연간 외부 피처는 T-1."""
    w = load_survey(survey_maxy, rows_maxy, k_all=k_all, k50_codes=k50_codes, ov=ov)
    w = w.join(w.groupby('year').y.mean().rename('cm_all'), on='year')
    w = w.join(w[w.in50 == 1].groupby('year').y.mean().rename('cm_50'), on='year')
    w = w.join(w.groupby('year').y.median().rename('cmed_all'), on='year')
    F = pd.DataFrame({'ksic': w.ksic, 'year': w.year, 'y': w.y, 'in50': w.in50, 'imputed_2y': w.imputed_2y,
                      'k3': w.k3, 'k2': w.k2, 'ship': w.ship})
    gs = w.groupby('ksic')
    L = lag
    for j in [1, 2, 3]:
        F[f'g{j}'] = gs['y'].shift(L + j - 1)
    for c in ['va', 'emp', 'estab', 'asset', 'wage', 'cost']:
        F[f'g{c}1'] = gs['g_' + c].shift(L)
    F['lsize1'] = gs['l_ship'].shift(L)
    F['va_ratio1'] = gs['l_va'].shift(L) - gs['l_ship'].shift(L)
    F['cost_ratio1'] = gs['l_cost'].shift(L) - gs['l_ship'].shift(L)
    F['asset_int1'] = gs['l_asset'].shift(L) - gs['l_ship'].shift(L)
    F['wage_pe1'] = gs['l_wage'].shift(L) - gs['l_emp'].shift(L)
    F['peer3_1'] = gs['peer_k3'].shift(L)
    F['peer2_1'] = gs['peer_k2'].shift(L)
    F['peer4_1'] = gs['peer_k4'].shift(L)
    F['peer3_2'] = gs['peer_k3'].shift(L + 1)
    F['n_sib4'] = w.groupby(['k4', 'year']).ksic.transform('count').groupby(w.ksic).shift(L) - 1
    F['cm_all1'] = gs['cm_all'].shift(L)
    F['cm_501'] = gs['cm_50'].shift(L)
    F['cmed_all1'] = gs['cmed_all'].shift(L)
    F['gmean_exp'] = gs['y'].transform(lambda s: s.shift(L).expanding(min_periods=1).mean())
    F['gsd_exp'] = gs['y'].transform(lambda s: s.shift(L).expanding(min_periods=2).std())
    F['g3avg'] = gs['y'].transform(lambda s: s.shift(L).rolling(3, min_periods=1).mean())
    F['ldev3'] = gs['l_ship_chain'].transform(lambda s: s.shift(L) - s.shift(L).rolling(3, min_periods=2).mean())
    F = add_external(F, rows_maxy, extra=extra)
    # 파생(모두 T-1 이전 값의 조합)
    F['dmargin1'] = F.gva1 - F.g1
    F['inv_rel1'] = F.gasset1 - F.g1
    F['prod_lab1'] = F.g1 - F.gemp1
    F['div_sector1'] = F.g1 - F.is_g3
    F['div_peer1'] = F.g1 - F.peer3_1
    F['estab_size1'] = F.gestab1 - F.gemp1
    return F


# ============================================================ 4. 특허 피처(절제 비교용, 50개 산업만)
def build_patent_features(rows_maxy):
    """출원 기반: L2 = T-2년 출원까지(공개지연 보수), 출원연도 상한 min(rows_maxy-1, 2022)(2023~ 공개지연 절단).
    등록원부 사건(소멸·이전·설정등록)은 사건일 기준 T-1년까지."""
    pc = os.path.join(HERE, 'cache/pat_ksic_cc_year.parquet')
    rc = os.path.join(HERE, 'cache/kr_registry_ksic_year.parquet')
    if not os.path.exists(pc):
        _build_patent_cache(pc)
    if not os.path.exists(rc):
        _build_registry_cache(rc)
    p = pd.read_parquet(pc)
    p = p[p.year <= min(rows_maxy - 1, 2022)]
    cnt = p.pivot_table(index=['ksic', 'year'], columns='cc', values='n_app', aggfunc='sum').fillna(0)
    cnt.columns = ['n_' + c.lower() for c in cnt.columns]
    cnt['n_tot'] = cnt.sum(axis=1)
    fpu = p[p.cc == 'US'].set_index(['ksic', 'year'])[['n_foreignprio', 'n_app']]
    cnt = cnt.join(fpu.rename(columns={'n_foreignprio': 'us_fp', 'n_app': '_us'}))
    c = cnt.reset_index().sort_values(['ksic', 'year'])
    lg = np.log1p
    base = pd.DataFrame({'ksic': c.ksic.astype(str), 'year': c.year})
    for cc in ['tot', 'kr', 'cn']:
        x = lg(c['n_' + cc])
        base[f'pg3_{cc}'] = (x - x.groupby(c.ksic).shift(3)) / 3
    base['cn_share'] = c.n_cn / c.n_tot
    base['cn_share_d3'] = base.cn_share - base.groupby('ksic').cn_share.shift(3)
    base['cn_vs_kr3'] = base.pg3_cn - base.pg3_kr
    base['us_fp_share'] = c.us_fp / c._us
    for v in ['pg3_tot', 'pg3_kr', 'cn_share_d3', 'cn_vs_kr3']:
        base[v + '_rel'] = base[v] - base.groupby('year')[v].transform('mean')
    b = base.copy()
    b['year'] = b.year + 2
    b = b.rename(columns={col: f'{col}_L2' for col in b.columns if col not in ('ksic', 'year')}).set_index(['ksic', 'year'])
    r = pd.read_parquet(rc)
    r['ksic'] = r.ksic.astype(str)
    r = r[r.year <= rows_maxy - 1].sort_values(['ksic', 'year'])
    full = pd.MultiIndex.from_product([r.ksic.unique(), range(2000, rows_maxy)], names=['ksic', 'year']).to_frame(index=False)
    r = full.merge(r, on=['ksic', 'year'], how='left').fillna(0)
    gr = r.groupby('ksic')
    r['stock'] = gr.kr_new_rgst.cumsum() - gr.kr_lapse.cumsum()
    r['stock_prev'] = r.groupby('ksic').stock.shift(1)
    reg = pd.DataFrame({'ksic': r.ksic, 'year': r.year})
    reg['kr_lapse_rate'] = r.kr_lapse / r.stock_prev.where(r.stock_prev > 50)
    reg['kr_transfer_rate'] = r.kr_transfer / r.stock_prev.where(r.stock_prev > 50)
    for v in ['kr_lapse_rate', 'kr_transfer_rate']:
        reg[v + '_rel'] = reg[v] - reg.groupby('year')[v].transform('mean')
    reg['year'] = reg.year + 1
    P = pd.concat([b, reg.set_index(['ksic', 'year'])], axis=1).reset_index()
    return P[P.year.between(2009, rows_maxy)]


def _build_patent_cache(path):
    import duckdb
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = duckdb.connect()
    con.sql("SET memory_limit='3GB'; SET threads=2")
    con.sql(f"""copy (select ksic, cc, appl_year::INT as year, count(distinct appl_no) as n_app,
      count(distinct case when prio_cc is not null and prio_cc<>'' and split_part(prio_cc,'|',1)<>cc then appl_no end) as n_foreignprio
      from read_parquet('{SHARED}/meta/*.parquet') where appl_year between 2000 and 2022 group by 1,2,3)
      to '{path}' (format parquet)""")


def _build_registry_cache(path):
    import sqlite3
    c = sqlite3.connect('file:' + ROOT + '/수집파이프라인/registry_full.sqlite?mode=ro', uri=True)
    pat = pd.read_sql('select rgst_no, ksic, substr(rgstDate,1,4) ry from patent', c)
    lap = pd.read_sql("select rgst_no, substr(rgstCsDate,1,4) y from right where rgstCsName='소멸등록' and rgstCsReason='등록료불납'", c)
    tr = pd.read_sql("select rgst_no, substr(ownerRgstCsDate,1,4) y from owner where ownerRgstCsReason like '%이전%'", c)
    pat['ksic'] = pat.ksic.astype(str)
    agg = lambda ev, nm: ev.merge(pat[['rgst_no', 'ksic']], on='rgst_no').groupby(['ksic', 'y']).size().rename(nm)
    reg = pat.groupby(['ksic', 'ry']).size().rename('kr_new_rgst')
    reg.index.names = ['ksic', 'y']
    out = pd.concat([reg, agg(lap, 'kr_lapse'), agg(tr, 'kr_transfer')], axis=1).fillna(0).reset_index()
    out['y'] = pd.to_numeric(out.y, errors='coerce')
    out = out.dropna(subset=['y'])
    out['y'] = out.y.astype(int)
    out[(out.y >= 2000) & (out.y <= 2024)].rename(columns={'y': 'year'}).to_parquet(path, index=False)


# ============================================================ 5. 모형
def rank_features(F, feats, sectors, rank50=False):
    """연도 안 백분위 순위(피처만 사용, y 관측 여부와 무관).
    - 기본 기준집단: 그해 패널의 '학습 업종(sectors)' 5자리 산업 전체.
    - rank50=True: 제공 50개 산업 행(학습·예측 모두, 모든 연도)은 그해 50개 산업 안에서 순위화(대상 집단 내 상대 위치).
      1단계 v31은 예측행만 '50개 중 y 관측 행' 안에서 순위화했음 → 감사에서 y 필터를 제거하고 학습행 50개에도 같은 규칙 적용."""
    R = pd.DataFrame(index=F.index)
    ref = F.k2.isin(sectors) if sectors else pd.Series(True, index=F.index)
    X = F.loc[ref, feats].astype(float)
    R.loc[ref, feats] = X.groupby(F.loc[ref, 'year'].values).rank(pct=True).values
    if rank50:
        t = F.in50 == 1
        R.loc[t, feats] = F.loc[t, feats].astype(float).groupby(F.loc[t, 'year'].values).rank(pct=True).values
    return R[feats].astype(float)


def _train_rows(F, T, spec):
    tr = F[(F.year < T) & (F.year >= spec['min_year']) & F.y.notna() & ~F.year.isin(spec['drop_years'])]
    if spec.get('sectors'):
        tr = tr[tr.k2.isin(spec['sectors'])]
    return tr


def _target(tr, spec):
    y = np.clip(tr.y.values.astype(float), -spec['ycap'], spec['ycap'])
    med = pd.Series(y, index=tr.index).groupby(tr.year.values).transform('median').values  # 학습 연도별 단면 중앙값
    return y - med


def fit_models(tr, R, spec, feats=None):
    import xgboost as xg
    feats = feats or spec['features']
    y = _target(tr, spec)
    sw = None
    if spec.get('decay'):
        sw = spec['decay'] ** (tr.year.max() - tr.year.values)
    P = dict(spec['xgb'])
    if spec.get('monotone'):  # 피처별 사전 부호(없는 피처는 0=자유)
        P['monotone_constraints'] = '(' + ','.join(str(int(spec['monotone'].get(f, 0))) for f in feats) + ')'
    ms = []
    for s in spec['seeds']:
        m = xg.XGBRegressor(**P, random_state=s)
        m.fit(R.loc[tr.index, feats], y, sample_weight=sw)
        ms.append(m)
    return ms


def predict_models(ms, R, idx, feats):
    return np.mean([m.predict(R.loc[idx, feats]) for m in ms], axis=0) if ms else np.zeros(len(idx))


def rolling(F, years, spec, feats=None, R=None, require_y=True, return_models=False):
    feats = feats or spec['features']
    if R is None:
        R = rank_features(F, feats, spec.get('sectors'), spec.get('rank50', False))
    out, models = [], {}
    for T in years:
        tr = _train_rows(F, T, spec)
        te = F[(F.year == T) & (F.in50 == 1)]
        if require_y:
            te = te[te.y.notna()]
        if len(tr) < spec['min_rows']:
            p, ms = np.zeros(len(te)), []
        else:
            ms = fit_models(tr, R, spec, feats)
            p = predict_models(ms, R, te.index, feats)
        o = te[['ksic', 'year', 'y']].copy()
        o['pred'] = p
        o['n_train'] = len(tr)
        out.append(o)
        models[T] = (ms, te.index)
    res = pd.concat(out, ignore_index=True)
    return (res, models, R) if return_models else res


# ============================================================ 6. 지표·불확실성
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    sp, sy = np.sign(p), np.sign(y)
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                DIR=float(np.mean(np.where(sp == 0, 0.5, (sp == sy).astype(float)))),
                MedAE=float(np.median(np.abs(e))), n=int(len(y)))


def boot(pred, base_pred, B=5000, seed=0, cluster='ksic'):
    """군집 부트스트랩: skill = 1 - MAE_m/MAE_b, MAE 차이(m-b)의 95% CI."""
    rng = np.random.default_rng(seed)
    m = pred.merge(base_pred[['ksic', 'year', 'pred']].rename(columns={'pred': 'pb'}), on=['ksic', 'year'])
    m['am'], m['ab'] = (m.y - m.pred).abs(), (m.y - m.pb).abs()
    g = m.groupby(cluster)[['am', 'ab']].agg(['sum', 'count'])
    A, Bb, N = g[('am', 'sum')].values, g[('ab', 'sum')].values, g[('am', 'count')].values
    k = len(A)
    sk, dd = [], []
    for _ in range(B):
        i = rng.integers(0, k, k)
        sk.append(1 - A[i].sum() / Bb[i].sum())
        dd.append((A[i].sum() - Bb[i].sum()) / N[i].sum())
    return dict(skill_ci=[float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))],
                mae_diff_ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], n_clusters=int(k))


def baseline_preds(F, years):
    """고정 기준선 후보. 확정 규칙(zero)은 검증에서 선택되어 final_spec에 고정됨."""
    te = F[F.year.isin(years) & (F.in50 == 1) & F.y.notna()]
    cands = {'zero': np.zeros(len(te)), 'last_growth': te.g1.fillna(0).values,
             'cm50_last': te.cm_501.fillna(0).values, 'cmall_last': te.cm_all1.fillna(0).values}
    out = {}
    for k, v in cands.items():
        d = te[['ksic', 'year', 'y']].copy()
        d['pred'] = v
        out[k] = d.reset_index(drop=True)
    return out


def load_spec(path=None):
    return json.load(open(path or os.path.join(HERE, 'final_spec.json'), encoding='utf-8'))


def model_spec(spec):
    m = dict(spec['model'])
    return m


# ============================================================ 7. 모드
def evaluate(F, years, spec, tag, out_dir=HERE):
    ms = model_spec(spec)
    pred = rolling(F, years, ms)
    bases = baseline_preds(F, years)
    rule = spec['baseline']['rule']
    bp = bases[rule]
    chk = pred.merge(bp[['ksic', 'year']], on=['ksic', 'year'])
    assert len(chk) == len(pred) == len(bp), '평가 표본 불일치'
    mm, mb = metrics(pred.y, pred.pred), metrics(bp.y, bp.pred)
    res = dict(tag=tag, years=years, n=len(pred), n_industries=int(pred.ksic.nunique()),
               model=mm, baseline_rule=rule, baseline=mb, skill=1 - mm['MAE'] / mb['MAE'],
               boot_industry=boot(pred, bp, cluster='ksic'),
               baseline_candidates={k: metrics(v.y, v.pred) for k, v in bases.items()})
    if len(years) >= 3:
        res['boot_year'] = boot(pred, bp, cluster='year', seed=1)
    yr = pred.assign(a=(pred.y - pred.pred).abs(), a0=(bp.set_index(['ksic', 'year']).loc[list(zip(pred.ksic, pred.year))].pred.values - pred.y).abs())
    res['by_year'] = {int(t): dict(n=int(len(g)), MAE=float(g.a.mean()), MAE_base=float(g.a0.mean()), skill=float(1 - g.a.mean() / g.a0.mean()),
                                   n_train=int(g.n_train.iloc[0])) for t, g in yr.groupby('year')}
    # 특허 절제: 같은 모형 + 특허 피처(50개 산업만 값 있음)
    if spec.get('patent_ablation'):
        P = build_patent_features(int(max(years)))
        F2 = F.merge(P, on=['ksic', 'year'], how='left')
        F2.index = F.index
        feats2 = ms['features'] + spec['patent_ablation']['features']
        pp = rolling(F2, years, ms, feats=feats2)
        mp = metrics(pp.y, pp.pred)
        q = pp.merge(pred[['ksic', 'year', 'pred']].rename(columns={'pred': 'p0'}), on=['ksic', 'year'])
        q['d'] = (q.y - q.pred).abs() - (q.y - q.p0).abs()
        g = q.groupby('ksic').d.agg(['sum', 'count'])
        rng = np.random.default_rng(2)
        bs = []
        for _ in range(5000):
            i = rng.integers(0, len(g), len(g))
            bs.append(g['sum'].values[i].sum() / g['count'].values[i].sum())
        res['patent_ablation'] = dict(features_added=spec['patent_ablation']['features'], model_with_patent=mp,
                                      skill_with_patent=1 - mp['MAE'] / mb['MAE'],
                                      mae_diff_with_minus_without=float(q.d.mean()),
                                      mae_diff_ci_industry=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        pp.to_csv(os.path.join(out_dir, f'{tag}_preds_patent.csv'), index=False, encoding='utf-8-sig')
    pred.merge(bp[['ksic', 'year', 'pred']].rename(columns={'pred': 'pred_baseline'}), on=['ksic', 'year']) \
        .to_csv(os.path.join(out_dir, f'{tag}_preds.csv'), index=False, encoding='utf-8-sig')
    return res


def mode_dryrun(spec):
    t = time.time()
    F = build_panel(2019, 2019, extra=spec['model'].get('extra', True))
    assert F.year.max() <= 2019 and F[F.y.notna()].year.max() <= 2019
    res = evaluate(F, VAL_YEARS, spec, 'dryrun')
    ref = spec.get('validation', {})
    res['spec_validation_MAE'] = ref.get('MAE')
    res['reproduced'] = bool(ref.get('MAE') is not None and abs(res['model']['MAE'] - ref['MAE']) < 1e-6)
    res['sec'] = time.time() - t
    json.dump(res, open(os.path.join(HERE, 'dryrun_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ['n', 'n_industries', 'model', 'baseline', 'skill', 'boot_industry', 'reproduced']}, ensure_ascii=False, indent=1))
    if 'patent_ablation' in res:
        print('특허 절제:', json.dumps(res['patent_ablation'], ensure_ascii=False))
    print('연도별:', {k: round(v['skill'], 4) for k, v in res['by_year'].items()}, f"{res['sec']:.0f}s")


def mode_holdout(spec, survey_maxy=2024, years=None, k_all=None, k50_codes=None, ov=OV, out_dir=HERE):
    """🔒 3단계 평가자 전용. 목표연도 T∈{2021..2024}: 학습 레이블은 목표연도<T(2009, 2012..T-1; 2010·2011 제외, 2020은 10차 기준 포함),
    피처는 T-1 이전. 한 번의 패널(조사 <=2024)로 만들지만 각 원점의 학습·피처는 T-1 이전만 사용된다."""
    t = time.time()
    years = years or HOLDOUT_YEARS
    F = build_panel(survey_maxy, survey_maxy, extra=spec['model'].get('extra', True), k_all=k_all, k50_codes=k50_codes, ov=ov)
    res = evaluate(F, years, spec, 'holdout', out_dir=out_dir)
    ytr = F[F.y.notna() & (F.year < min(years))].year.unique()
    res['label_years_available_before_first_origin'] = sorted(int(v) for v in ytr)
    res['note'] = '기준선 규칙은 검증(2012-2019)에서 고정(zero). 부트스트랩 군집=산업(주), 연도(보조, 4개뿐이라 참고용).'
    res['sec'] = time.time() - t
    json.dump(res, open(os.path.join(out_dir, 'holdout_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ['n', 'n_industries', 'model', 'baseline', 'skill', 'boot_industry']}, ensure_ascii=False, indent=1))


def evaluate_strict(spec, years, full_maxy, tag, out_dir=HERE):
    """[3단계 규칙 준수 수정, 모형 불변] 원점 T마다 조사자료를 T-1까지만 로드한 패널(build_panel(T-1, T))로 학습·예측.
    원래 holdout/dryrun은 전체 기간 격자 패널 하나로 순위를 매겨, 나중 조사에 처음 나타나는 코드의 '유령 행'(과거 연도,
    외부 지수·ECOS 값만 있음)이 과거 학습행의 순위 기준집단에 섞였다(y 미사용이지만 미래 코드 목록 = 기준시점 이후 정보).
    평가 y만 전체 패널(조사 <=full_maxy)에서 붙인다. 모형·피처·하이퍼파라미터·기준선·지표는 동일."""
    t0 = time.time()
    ms = model_spec(spec)
    k_all = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_2008_2024.csv', dtype={'세세분류코드': str})
    k_all = k_all[k_all.연도 <= full_maxy]
    Ffull = build_panel(full_maxy, full_maxy, extra=ms.get('extra', True), k_all=k_all)
    te = Ffull[Ffull.year.isin(years) & (Ffull.in50 == 1) & Ffull.y.notna()][['ksic', 'year', 'y']].reset_index(drop=True)
    pf = spec['patent_ablation']['features'] if spec.get('patent_ablation') else []
    P0, P1 = [], []
    for T in years:
        FT = build_panel(T - 1, T, extra=ms.get('extra', True), k_all=k_all)
        assert FT[FT.y.notna()].year.max() <= T - 1
        P0.append(rolling(FT, [T], ms, require_y=False).drop(columns='y'))
        if pf:
            PT = build_patent_features(T)
            F2 = FT.merge(PT, on=['ksic', 'year'], how='left')
            F2.index = FT.index
            P1.append(rolling(F2, [T], ms, feats=ms['features'] + pf, require_y=False).drop(columns='y'))
    pred = te.merge(pd.concat(P0), on=['ksic', 'year'], how='left')
    assert pred.pred.notna().all()
    bp = te.copy()
    bp['pred'] = 0.0  # 고정 기준선: zero (검증에서 선택)
    mm, mb = metrics(pred.y, pred.pred), metrics(bp.y, bp.pred)
    res = dict(tag=tag, strict_per_origin_panel=True, years=years, n=len(pred), n_industries=int(pred.ksic.nunique()),
               model=mm, baseline_rule=spec['baseline']['rule'], baseline=mb, skill=1 - mm['MAE'] / mb['MAE'],
               boot_industry=boot(pred, bp, cluster='ksic'))
    if len(years) >= 3:
        res['boot_year'] = boot(pred, bp, cluster='year', seed=1)
    yr = pred.assign(a=(pred.y - pred.pred).abs(), a0=pred.y.abs())
    res['by_year'] = {int(t): dict(n=int(len(g)), MAE=float(g.a.mean()), MAE_base=float(g.a0.mean()),
                                   skill=float(1 - g.a.mean() / g.a0.mean()), n_train=int(g.n_train.iloc[0])) for t, g in yr.groupby('year')}
    if pf:
        pp = te.merge(pd.concat(P1), on=['ksic', 'year'], how='left')
        mp = metrics(pp.y, pp.pred)
        q = pp.merge(pred[['ksic', 'year', 'pred']].rename(columns={'pred': 'p0'}), on=['ksic', 'year'])
        q['d'] = (q.y - q.pred).abs() - (q.y - q.p0).abs()
        g = q.groupby('ksic').d.agg(['sum', 'count'])
        rng = np.random.default_rng(2)
        bs = []
        for _ in range(5000):
            i = rng.integers(0, len(g), len(g))
            bs.append(g['sum'].values[i].sum() / g['count'].values[i].sum())
        res['patent_ablation'] = dict(features_added=pf, model_with_patent=mp, skill_with_patent=1 - mp['MAE'] / mb['MAE'],
                                      mae_diff_with_minus_without=float(q.d.mean()),
                                      mae_diff_ci_industry=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        pp.to_csv(os.path.join(out_dir, f'{tag}_preds_patent.csv'), index=False, encoding='utf-8-sig')
    pred.assign(pred_baseline=0.0).to_csv(os.path.join(out_dir, f'{tag}_preds.csv'), index=False, encoding='utf-8-sig')
    res['sec'] = time.time() - t0
    json.dump(res, open(os.path.join(out_dir, f'{tag}_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ['n', 'n_industries', 'model', 'baseline', 'skill', 'boot_industry']}, ensure_ascii=False, indent=1))
    if pf:
        print('특허 절제:', json.dumps(res['patent_ablation'], ensure_ascii=False))
    print('연도별:', {k: round(v['skill'], 4) for k, v in res['by_year'].items()}, f"{res['sec']:.0f}s")
    return res


def mode_future(spec, survey_maxy=2024, targets=None, k_all=None, k50_codes=None, ov=OV, out_dir=HERE):
    """전체 자료(조사 <=2024)로 재학습. 2025 = 주 모형(조사 T-1=2024, 지수 2024).
    2026 = 보조 모형(조사 피처를 2년 전(2024)으로 당긴 lag=2 모형 + 지수 T-1=2025). 공통 수준은 0(상대 성장 전망)."""
    import xgboost as xg
    ms = model_spec(spec)
    nm = pd.read_csv(f'{D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str})
    nm = nm.sort_values('연도').drop_duplicates('세세분류코드', keep='last')[['세세분류코드', '산업명']].rename(columns={'세세분류코드': 'ksic', '산업명': 'name'})
    outs = []
    targets = targets or [(survey_maxy + 1, 1), (survey_maxy + 2, 2)]
    for target, lag in targets:
        F = build_panel(survey_maxy, target, lag=lag, extra=ms.get('extra', True), k_all=k_all, k50_codes=k50_codes, ov=ov)
        pr, models, R = rolling(F, [target], ms, require_y=False, return_models=True)
        mlist, idx = models[target]
        C = np.mean([m.get_booster().predict(xg.DMatrix(R.loc[idx, ms['features']]), pred_contribs=True) for m in mlist], axis=0)
        cdf = pd.DataFrame(C[:, :-1], columns=['c_' + f for f in ms['features']])
        o = pr[['ksic', 'year', 'pred']].reset_index(drop=True).rename(columns={'year': 'target_year', 'pred': 'pred_rel_growth'})
        o['model'] = 'main_lag1' if lag == 1 else 'aux_lag2(조사 T-2, 지수 T-1)'
        o['n_train'] = pr.n_train.values
        top = cdf.abs().values.argsort(axis=1)[:, ::-1][:, :3]
        o['top_drivers'] = ['; '.join(f"{ms['features'][j]}({cdf.iloc[i, j]:+.3f})" for j in top[i]) for i in range(len(o))]
        o = pd.concat([o, cdf], axis=1)
        # 특허 절제 모형(같은 설정 + 특허 피처): 예측과 특허 피처의 |기여| 비중
        if spec.get('patent_ablation') and lag == 1:
            P = build_patent_features(min(target, survey_maxy + 1))
            F2 = F.merge(P, on=['ksic', 'year'], how='left')
            F2.index = F.index
            f2 = ms['features'] + spec['patent_ablation']['features']
            pr2, m2, R2 = rolling(F2, [target], ms, feats=f2, require_y=False, return_models=True)
            ml2, idx2 = m2[target]
            C2 = np.mean([m.get_booster().predict(xg.DMatrix(R2.loc[idx2, f2]), pred_contribs=True) for m in ml2], axis=0)[:, :-1]
            pf = [f2.index(f) for f in spec['patent_ablation']['features']]
            o['pred_with_patent'] = pr2.pred.values
            o['patent_contrib_sum'] = C2[:, pf].sum(axis=1)
            o['patent_contrib_share'] = np.abs(C2[:, pf]).sum(axis=1) / np.maximum(np.abs(C2).sum(axis=1), 1e-12)
        outs.append(o)
    out = pd.concat(outs, ignore_index=True).merge(nm, on='ksic', how='left')
    out['note'] = '상대 성장 전망(연도 중앙값 대비, 공통 수준=0 기준선). 특허 기여는 절제 모형 기준 참고치.'
    out.to_csv(os.path.join(out_dir, 'future_forecast.csv'), index=False, encoding='utf-8-sig')
    print(out[['ksic', 'name', 'target_year', 'pred_rel_growth']].head(10).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dryrun', 'holdout', 'future', 'holdout_strict', 'dryrun_strict'])
    ap.add_argument('--spec', default=None)
    a = ap.parse_args()
    S = load_spec(a.spec)
    if a.mode == 'holdout_strict':
        evaluate_strict(S, HOLDOUT_YEARS, 2024, 'holdout_strict')
    elif a.mode == 'dryrun_strict':
        evaluate_strict(S, VAL_YEARS, 2019, 'dryrun_strict')
    else:
        {'dryrun': mode_dryrun, 'holdout': mode_holdout, 'future': mode_future}[a.mode](S)
