# -*- coding: utf-8 -*-
"""의미 검정 (science_ 트랙, 2026-09-28) — 개발 원점 t0<=2016 (결과 <=2019)만 사용.
검정 A: 같은 나라 논문 성장 -> 1~3년 뒤 같은 관청 특허 성장 (KSIC x 5개국 = 250 단위/원점)
검정 B: 해외(ALL-KR) 논문 성장 -> KR 특허 성장 (50 KSIC/원점)
검정 C: 기업 R&D(ANBERD, ISIC 그룹 x 5개국) 성장 -> 특허 성장
검정 D: ECOS 연구개발비/매출액(KR, 업종) 수준·변화 -> KR 특허 성장
검정 E: KSIC 논문 성장 -> KR 기술셀(KSIC x IPC 메인그룹) 성장 (셀 모멘텀 통제)
모든 검정: 결과 = log(특허 점유 t0+1..t0+3) - log(t0-3..t0-1)  [관청-연도 점유율 기준, 관청 전체 성장 제거]
통제 = 특허 모멘텀 log(t0-3..t0-1) - log(t0-6..t0-4). 원점별 편상관(Spearman 순위 후 잔차 상관),
관청(국가) 내 중심화. 순열 귀무분포(신호를 KSIC 간 무작위 섞기 200회)로 p값.
"""
import os, json
import numpy as np, pandas as pd, duckdb
from scipy import stats

EXT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(EXT, '..', 'meta', '*.parquet')
rng = np.random.default_rng(0)
ORIGINS = list(range(2006, 2017))
C2O = {'KR': 'KR', 'US': 'US', 'JP': 'JP', 'CN': 'CN', 'DE': 'EP'}

pat = pd.read_parquet(os.path.join(EXT, 'science_patent_counts_ksic_cc_year.parquet'))
pat['share'] = pat.n_pat / pat.groupby(['cc', 'year']).n_pat.transform('sum')
P = pat.pivot_table(index=['ksic', 'cc'], columns='year', values='share', fill_value=0)
oa = pd.read_parquet(os.path.join(EXT, 'science_openalex_ksic_country_year.parquet'))
tot = pd.read_parquet(os.path.join(EXT, 'science_openalex_country_total_year.parquet'))
oa = oa.merge(tot, on=['country', 'year'])
oa['share'] = oa.n_works / oa.n_works_total
W = oa.pivot_table(index=['ksic', 'country'], columns='year', values='share', fill_value=0)
Wn = oa.pivot_table(index=['ksic', 'country'], columns='year', values='n_works', fill_value=0)


def win(M, a, b):
    return M.loc[:, [y for y in range(a, b + 1)]].sum(axis=1)


def lg(x, eps):
    return np.log(x + eps)


def pcorr(y, x, z_list, groups=None):
    """순위 변환 후 통제변수 잔차화 -> 편상관"""
    df = pd.DataFrame({'y': y, 'x': x, **{f'z{i}': z for i, z in enumerate(z_list)}}).dropna()
    if groups is not None:
        g = pd.Series(groups, index=y.index).loc[df.index]
        df = df.groupby(g).transform(lambda s: s.rank() - s.rank().mean())
    else:
        df = df.rank()
    Z = np.column_stack([np.ones(len(df))] + [df[c].values for c in df.columns if c.startswith('z')])
    ry = df.y.values - Z @ np.linalg.lstsq(Z, df.y.values, rcond=None)[0]
    rx = df.x.values - Z @ np.linalg.lstsq(Z, df.x.values, rcond=None)[0]
    return np.corrcoef(ry, rx)[0, 1], len(df)


def run_test(name, build, n_perm=200):
    res = []
    for t0 in ORIGINS:
        d = build(t0)
        if d is None or len(d) < 10:
            continue
        r, n = pcorr(d.y, d.x, [d[c] for c in d.columns if c.startswith('ctl')], d.get('grp'))
        # 순열: x 를 (그룹 내) 무작위 섞기
        null = []
        for _ in range(n_perm):
            xp = d.x.copy()
            if 'grp' in d:
                xp = d.groupby('grp').x.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
            else:
                xp = pd.Series(rng.permutation(d.x.values), index=d.index)
            null.append(pcorr(d.y, xp, [d[c] for c in d.columns if c.startswith('ctl')], d.get('grp'))[0])
        res.append({'test': name, 't0': t0, 'n': n, 'pcorr': r, 'null_sd': np.std(null), 'perm_p': (np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1)})
    return pd.DataFrame(res)


def outcome(M, t0):
    return lg(win(M, t0 + 1, t0 + 3), 1e-6) - lg(win(M, t0 - 3, t0 - 1), 1e-6)


def momentum(M, t0):
    return lg(win(M, t0 - 3, t0 - 1), 1e-6) - lg(win(M, t0 - 6, t0 - 4), 1e-6)


# ---- A
def build_A(t0):
    rows = []
    y = outcome(P, t0); m = momentum(P, t0)
    xg = lg(win(W, t0 - 2, t0), 1e-7) - lg(win(W, t0 - 5, t0 - 3), 1e-7)
    for (k, c), v in xg.items():
        if c not in C2O:
            continue
        o = C2O[c]
        if (k, o) not in y.index:
            continue
        if win(Wn, t0 - 5, t0).get((k, c), 0) < 30:  # 논문 너무 적은 셀 제외
            continue
        rows.append({'ksic': k, 'grp': o, 'y': y[(k, o)], 'x': v, 'ctl_m': m[(k, o)]})
    return pd.DataFrame(rows)


# ---- B
def build_B(t0):
    y = outcome(P, t0); m = momentum(P, t0)
    WA = W.xs('ALL', level='country'); WK = W.xs('KR', level='country')
    NA = Wn.xs('ALL', level='country'); NK = Wn.xs('KR', level='country')
    # 해외 = 전체 - KR (건수 기반, 전체 대비 점유로 정규화)
    TA = tot[tot.country == 'ALL'].set_index('year').n_works_total; TK = tot[tot.country == 'KR'].set_index('year').n_works_total
    NF = NA - NK
    TF = TA - TK
    SF = NF.div(TF.reindex(NF.columns).values, axis=1)
    xf = lg(win(SF, t0 - 2, t0), 1e-7) - lg(win(SF, t0 - 5, t0 - 3), 1e-7)
    xk = lg(win(WK, t0 - 2, t0), 1e-7) - lg(win(WK, t0 - 5, t0 - 3), 1e-7)
    rows = []
    for k in xf.index:
        rows.append({'ksic': k, 'y': y[(k, 'KR')], 'x': xf[k], 'ctl_m': m[(k, 'KR')], 'ctl_kr': xk[k]})
    return pd.DataFrame(rows)


# ---- C  ANBERD
ISIC = {'15220': 'C15', '20202': 'C20', '20411': 'C20', '20423': 'C20', '20491': 'C20', '21220': 'C21', '23992': 'C23',
        '24191': 'C241_2431', '24219': 'C242_2432', '25914': 'C25X252', '25922': 'C25X252', '25924': 'C25X252', '25932': 'C25X252',
        '26111': 'C261', '26112': 'C261', '26121': 'C261', '26129': 'C261', '26211': 'C261', '26212': 'C261', '26223': 'C261',
        '26224': 'C261', '26291': 'C261', '26322': 'C262', '26323': 'C262', '26422': 'C263', '26429': 'C263', '26519': 'C264',
        '26529': 'C264', '27199': 'C325', '27211': 'C265', '27212': 'C265', '27301': 'C267', '28111': 'C27', '28122': 'C27',
        '28511': 'C27', '28902': 'C27', '29131': 'C28', '29132': 'C28', '29133': 'C28', '29150': 'C28', '29162': 'C28',
        '29210': 'C28', '29229': 'C28', '29241': 'C28', '29280': 'C28', '30393': 'C29', '31202': 'C302', '31312': 'C303',
        '33301': 'C32X325', '33309': 'C32X325'}
A2O = {'KOR': 'KR', 'USA': 'US', 'JPN': 'JP', 'CHN': 'CN', 'DEU': 'EP'}
an = pd.read_parquet(os.path.join(EXT, 'science_oecd_anberd_isic4.parquet'))
an = an[(an.CRITERIA == 'MA') & (an.UNIT_MEASURE == 'USD_PPP') & (an.PRICE_BASE == 'Q') & an.REF_AREA.isin(A2O)]
AN = an.pivot_table(index=['ACTIVITY', 'REF_AREA'], columns='TIME_PERIOD', values='OBS_VALUE')
patraw = pat.copy(); patraw['isic'] = patraw.ksic.map(ISIC)
PG = patraw.groupby(['isic', 'cc', 'year']).n_pat.sum().reset_index()
PG['share'] = PG.n_pat / PG.groupby(['cc', 'year']).n_pat.transform('sum')
PGm = PG.pivot_table(index=['isic', 'cc'], columns='year', values='share', fill_value=0)


def build_C(t0):
    y = outcome(PGm, t0); m = momentum(PGm, t0)
    rows = []
    for (isic, ra), s in AN.iterrows():
        o = A2O[ra]
        if (isic, o) not in y.index:
            continue
        a1 = s.reindex(range(t0 - 4, t0 - 1)); a0 = s.reindex(range(t0 - 7, t0 - 4))  # 발표시차 2년
        if a1.isna().any() or a0.isna().any() or a0.sum() <= 0:
            continue
        rows.append({'isic': isic, 'grp': o, 'y': y[(isic, o)], 'x': np.log(a1.sum() / a0.sum()), 'ctl_m': m[(isic, o)]})
    return pd.DataFrame(rows)


# ---- D  ECOS R&D intensity (KR)
ECOS = {'15220': 'C15', '20202': 'C202', '20411': 'C204', '20423': 'C204', '20491': 'C204', '21220': 'C21', '23992': 'C239',
        '24191': 'C241', '24219': 'C242', '25914': 'C25', '25922': 'C25', '25924': 'C25', '25932': 'C25',
        '26111': 'C26', '26112': 'C26', '26121': 'C26', '26129': 'C26', '26211': 'C262', '26212': 'C262', '26223': 'C262',
        '26224': 'C262', '26291': 'C262', '26322': 'C263', '26323': 'C263', '26422': 'C264', '26429': 'C264', '26519': 'C26',
        '26529': 'C26', '27199': 'C27', '27211': 'C27', '27212': 'C27', '27301': 'C27', '28111': 'C281', '28122': 'C281',
        '28511': 'C285', '28902': 'C289', '29131': 'C291', '29132': 'C291', '29133': 'C291', '29150': 'C291', '29162': 'C291',
        '29210': 'C292', '29229': 'C292', '29241': 'C292', '29280': 'C292', '30393': 'C303', '31202': 'C31', '31312': 'C31',
        '33301': 'C33', '33309': 'C33'}
ec = pd.read_parquet(os.path.join(EXT, 'science_ecos_rd_intensity.parquet'))
ec = ec[ec['size'] == 'A']
e6 = ec[ec.stat_code == '501Y006']; e26 = ec[(ec.stat_code == '501Y026') & (ec.year < 2009)]
E = pd.concat([e6, e26]).pivot_table(index='ind_code', columns='year', values='rd_to_sales_pct')
patraw['ecos'] = patraw.ksic.map(ECOS)
PE = patraw[patraw.cc == 'KR'].groupby(['ecos', 'year']).n_pat.sum().reset_index()
PE['share'] = PE.n_pat / PE.groupby('year').n_pat.transform('sum')
PEm = PE.pivot_table(index='ecos', columns='year', values='share', fill_value=0)


def build_D(t0, mode):
    y = outcome(PEm, t0); m = momentum(PEm, t0)
    rows = []
    for g in y.index:
        if g not in E.index:
            continue
        s = E.loc[g]
        if mode == 'level':
            v = s.reindex(range(t0 - 2, t0)).mean()  # 발표시차 1년: t0-1까지
        else:
            a1 = s.reindex(range(t0 - 2, t0)).mean(); a0 = s.reindex(range(t0 - 4, t0 - 2)).mean()
            v = np.log(a1 / a0) if a0 and a0 > 0 else np.nan
        if pd.isna(v):
            continue
        rows.append({'g': g, 'y': y[g], 'x': v, 'ctl_m': m[g]})
    return pd.DataFrame(rows)


# ---- E  KR 기술셀
def cells():
    con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
    q = f"""select ksic, split_part(main_ipc,'/',1) as mg, appl_year as year, count(distinct appl_no) n
            from read_parquet('{META}') where cc='KR' and appl_year between 1998 and 2019 and main_ipc is not null
            group by 1,2,3"""
    return con.sql(q).df()


def build_E_factory(C):
    C = C.copy(); C['cell'] = C.ksic + '|' + C.mg
    M = C.pivot_table(index='cell', columns='year', values='n', fill_value=0)
    ks = pd.Series([c.split('|')[0] for c in M.index], index=M.index)

    def build(t0):
        past = win(M, t0 - 3, t0 - 1)
        keep = past >= 10
        Mk = M[keep]
        y = np.log(win(Mk, t0 + 1, t0 + 3) + 1) - np.log(win(Mk, t0 - 3, t0 - 1) + 1)
        m = np.log(win(Mk, t0 - 3, t0 - 1) + 1) - np.log(win(Mk, t0 - 6, t0 - 4) + 1)
        WA = W.xs('ALL', level='country')
        xg = lg(win(WA, t0 - 2, t0), 1e-7) - lg(win(WA, t0 - 5, t0 - 3), 1e-7)
        # KSIC 전체 특허 모멘텀도 통제 (논문 효과가 단순 산업 모멘텀이 아님을 확인)
        km = momentum(P, t0).xs('KR', level='cc')
        d = pd.DataFrame({'y': y, 'x': ks[keep].map(xg).values, 'ctl_m': m, 'ctl_km': ks[keep].map(km).values}, index=Mk.index)
        return d
    return build


if __name__ == "__main__":
    out = []
    out.append(run_test('A_same_country_paper->patent', build_A))
    out.append(run_test('B_foreign_paper->KR_patent', build_B))
    out.append(run_test('C_ANBERD_BERD->patent', build_C))
    out.append(run_test('D_ECOS_RDintensity_level->KR_patent', lambda t: build_D(t, 'level')))
    out.append(run_test('D_ECOS_RDintensity_change->KR_patent', lambda t: build_D(t, 'change')))
    out.append(run_test('E_KSIC_paper->KR_cell', build_E_factory(cells()), n_perm=50))
    R = pd.concat(out)
    R.to_csv(os.path.join(EXT, 'science_meaning_test_results.csv'), index=False, encoding='utf-8-sig')
    S = R.groupby('test').agg(n_origins=('t0', 'size'), mean_n=('n', 'mean'), mean_pcorr=('pcorr', 'mean'),
                              share_pos=('pcorr', lambda s: (s > 0).mean()), mean_null_sd=('null_sd', 'mean'),
                              n_perm_p05=('perm_p', lambda s: (s < 0.05).sum()))
    S['z_mean'] = S.mean_pcorr / (S.mean_null_sd / np.sqrt(S.n_origins))  # 원점 독립 가정 근사
    print(S.round(3).to_string())
    print(R.round(3).to_string())
