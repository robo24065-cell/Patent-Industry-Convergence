"""T3 3라운드 B단계(재시도) 외부 데이터 결합층 C — 재시도 수집분을 공개연도 변형 셀×원점에 붙인다.
산출(R3 폴더): extC_cell_pub{suffix}.parquet (cell×t0), extC_ksic_pub{suffix}.parquet (ksic×t0), extC_link{suffix}.csv

원천(_shared/ext, 읽기만):
  - r3oa_mg_year_ALL.parquet (IPC 메인그룹 558개 × 발행연도, 1차 460 + KSIC 균형 보강 98)
  - r3oa_mg_year_company.parquet (기업 소속 논문, 메인그룹 523개)
  - science_openalex_country_total_year.parquet (전 세계 논문 총량), science_openalex_ksic_extra_year.parquet (TOTAL company_ALL = 기업 논문 총량)
  - r3kosis_ksic50_items_chain.parquet (KOSIS 광업제조업조사 연간 품목편 연쇄 성장)
시점 규칙("t0년 말까지 공개된 자료만"):
  - OpenAlex: 발행연도 ≤ t0 논문만(창 t0-5..t0). 2026-09 스냅샷 소급 집계 → 준-정합. 수준값 대신 같은 스냅샷 총량 대비 점유 성장.
    정의는 _shared/ext/r3oa_work/r3oa_11_features.py 와 같다(원자료에서 다시 계산해 t0=2024 까지 만들 수 있게 함).
  - KOSIS 품목편: 자료연도 Y 확정 공표가 Y+2년 1–2월 → Y ≤ t0-2. 금액 원값(재기준 없음).
  - ETSI 표준필수특허(SEP) 선언(quality_etsi_sep_link.parquet, 제공 레코드↔선언 연결, 로컬 전용): 최초 선언일 ≤ t0-12-31 이고
    레코드 출원연도 ≤ t0-2(18개월 공개 보수 규칙)인 것만 셀별로 센다. ETSI IPR DB는 선언일에 공개된다.
사용: python build_ext_r3c.py [--t0max 2018] [--suffix '']
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
SHX = os.path.join(HERE, '..', '..', '_shared', 'ext')


def oa_feats(t0s, cells, mg_c):
    A = pd.read_parquet(os.path.join(SHX, 'r3oa_mg_year_ALL.parquet')).pivot_table(index='mg', columns='year', values='n_works', aggfunc='first')
    Cc = pd.read_parquet(os.path.join(SHX, 'r3oa_mg_year_company.parquet')).pivot_table(index='mg', columns='year', values='n_works', aggfunc='first')
    T = pd.read_parquet(os.path.join(SHX, 'science_openalex_country_total_year.parquet'))
    ta = T[T.country == 'ALL'].set_index('year').n_works_total.sort_index()
    E = pd.read_parquet(os.path.join(SHX, 'science_openalex_ksic_extra_year.parquet'))
    tc = E[(E.ksic == 'TOTAL') & (E.variant == 'company_ALL')].set_index('year').n_works.sort_index()
    win = lambda P, t0, lo, hi: P[[y for y in range(t0 + lo, t0 + hi + 1)]].fillna(0).sum(1)
    out = []
    for t0 in t0s:
        a3 = win(A, t0, -2, 0); a0 = win(A, t0, -5, -3)
        ag = np.log((a3 + 1) / ta.loc[t0 - 2:t0].sum()) - np.log((a0 + 1) / ta.loc[t0 - 5:t0 - 3].sum())
        c3 = win(Cc, t0, -2, 0); c0 = win(Cc, t0, -5, -3)
        cg = np.log((c3 + 1) / tc.loc[t0 - 2:t0].sum()) - np.log((c0 + 1) / tc.loc[t0 - 5:t0 - 3].sum())
        cs = np.log(c3 + 1) - np.log(a3.reindex(Cc.index) + 1)
        m = pd.Series(mg_c)
        out.append(pd.DataFrame(dict(cell=cells, t0=t0,
                                     oa2_all_g3=ag.reindex(mg_c).values, oa2_all_lv3=np.log1p(a3).reindex(mg_c).values,
                                     oa2_co_g3=cg.reindex(mg_c).values, oa2_co_share=cs.reindex(mg_c).values,
                                     oa2_all_link=m.isin(A.index).values.astype(np.int8), oa2_co_link=m.isin(Cc.index).values.astype(np.int8))))
    return pd.concat(out, ignore_index=True)


def kosis_pub_feats(t0s, K50):
    ch = pd.read_parquet(os.path.join(SHX, 'r3kosis_ksic50_items_chain.parquet')).set_index(['ksic', 'year'])
    def gv(ks, y, col):
        try: v = ch.loc[(ks, y), col]
        except KeyError: return np.nan
        v = float(v) if np.isscalar(v) else float(np.asarray(v)[0])
        return v
    rows = []
    for ks in K50:
        for t0 in t0s:
            Y = t0 - 2
            gp = [gv(ks, Y, 'g_prod'), gv(ks, Y - 1, 'g_prod')]; gs = [gv(ks, Y, 'g_ship'), gv(ks, Y - 1, 'g_ship')]
            s2 = lambda g: float(np.sum(np.clip(g, -1, 1))) if all(x == x for x in g) else np.nan
            iv = gv(ks, Y, 'inv_ship1')
            rows.append(dict(ksic=ks, t0=t0, kp_gprod2=s2(gp), kp_gship2=s2(gs), kp_invlvl=(np.clip(iv, 0, 2) if iv == iv else np.nan)))
    K = pd.DataFrame(rows); K['kp_miss'] = K.kp_gprod2.isna().astype(np.int8)
    return K


def sep_feats(t0s, cells):
    S = pd.read_parquet(os.path.join(SHX, 'quality_etsi_sep_link.parquet'))
    S['mg'] = S.main_ipc.str.split('/').str[0].str.strip(); S['cell'] = S.ksic + '|' + S.mg
    S['dy'] = pd.to_datetime(S.first_decl, errors='coerce').dt.year
    S = S[S.dy.notna() & S.cell.isin(set(cells))]
    out = []
    for t0 in t0s:
        b = S[(S.dy <= t0) & (S.appl_year <= t0 - 2)]
        cum = b.groupby('cell').size().reindex(cells).fillna(0).values
        new = b[b.dy >= t0 - 2].groupby('cell').size().reindex(cells).fillna(0).values
        out.append(pd.DataFrame(dict(cell=cells, t0=t0, sep_cum=np.log1p(cum), sep_new3=np.log1p(new))))
    return pd.concat(out, ignore_index=True)


def build(L, t0s, suffix=''):
    P = L.load(); cells = P['cells']; ksic_c = P['ksic']; mg_c = P['mg']; K50 = sorted(set(ksic_c))
    assert max(t0s) + 3 < L.F_YEAR or os.environ.get('T3R3_FINAL_UNLOCK') == '1' or max(t0s) <= 2021
    tot = P['A']['n'].sum(1); totW = {t0: L.wsum(tot, t0, -2, 0) for t0 in t0s}
    O = oa_feats(t0s, cells, mg_c).merge(sep_feats(t0s, cells), on=['cell', 't0'], how='left')
    tw = np.concatenate([totW[t0] for t0 in t0s])
    O = O[tw >= 10].reset_index(drop=True)
    K = kosis_pub_feats(t0s, K50)
    O.to_parquet(os.path.join(HERE, f'extC_cell_pub{suffix}.parquet')); K.to_parquet(os.path.join(HERE, f'extC_ksic_pub{suffix}.parquet'))
    el = O.merge(pd.DataFrame(dict(cell=np.tile(cells, len(t0s)), t0=np.repeat(t0s, len(cells)), tot_W=tw)), on=['cell', 't0'])
    el = el[el.tot_W >= 50]; el['ksic'] = el.cell.str[:5]
    lk = el.groupby('t0').agg(n_elig=('cell', 'size'), oa_all_link=('oa2_all_link', 'mean'), oa_co_link=('oa2_co_link', 'mean'),
                              oa_co_g_nonmiss=('oa2_co_g3', lambda s: s.notna().mean()),
                              sep_any=('sep_cum', lambda s: (s > 0).mean()))
    lk = lk.join(K.groupby('t0').kp_gprod2.agg(lambda s: s.notna().sum()).rename('nksic_kp'))
    lk.to_csv(os.path.join(HERE, f'extC_link{suffix}.csv'), encoding='utf-8-sig')
    lks = el.groupby('ksic').agg(oa_all_link=('oa2_all_link', 'mean'), oa_co_link=('oa2_co_link', 'mean'), sep_any=('sep_cum', lambda s: (s > 0).mean()))
    lks.to_csv(os.path.join(HERE, f'extC_link_ksic{suffix}.csv'), encoding='utf-8-sig')
    print(lk.round(3).to_string())
    print('KSIC 연결(적격 셀 ≥1): OpenAlex ALL', int((lks.oa_all_link > 0).sum()), '/50, company', int((lks.oa_co_link > 0).sum()), '/50;',
          'company 셀 연결 ≥50% KSIC', int((lks.oa_co_link >= 0.5).sum()), '/50;', 'SEP 셀 ≥1 KSIC', int((lks.sep_any > 0).sum()), '/50')
    return O, K


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--t0max', type=int, default=2018); ap.add_argument('--suffix', default=''); a = ap.parse_args()
    os.environ.setdefault('T3R3_VAR', 'pub')
    import r3lib as L
    assert a.t0max + 3 < L.F_YEAR or os.environ.get('T3R3_FINAL_UNLOCK') == '1'
    build(L, list(range(2005, a.t0max + 1)), a.suffix)
