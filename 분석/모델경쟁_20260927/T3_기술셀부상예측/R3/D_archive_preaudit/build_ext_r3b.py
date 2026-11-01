"""T3 3라운드 B단계 외부 데이터 결합층 — 3라운드 신규 수집분(_shared/ext/r3oa_*, r3ros_*, r3kosis_*)을 공개연도 변형 셀×원점에 붙인다.
산출(R3 폴더): extB_cell_pub{suffix}.parquet (cell×t0), extB_ksic_pub{suffix}.parquet (ksic×t0), extB_link_{suffix}.csv (연계율)

시점 규칙("t0년 말까지 공개된 자료만"):
  - RoS(Reliance on Science) 특허→논문 인용: US 특허 발행연도 issue_year ∈ [t0-9, t0] (인용은 발행 시 확정, 사후 수정 없음).
      셀 배정은 제공 레코드의 family US 토큰으로 이뤄지므로, 연결 레코드의 최초 출원연도 first_rec_year ≤ t0-2 (18개월 공개 지연 보수 규칙)만 쓴다.
  - OpenAlex 셀(IPC 메인그룹) 논문: 발행연도 ≤ t0 (2026-09 현재 스냅샷 소급 집계 → 준-정합). 전 세계 논문 총수로 나눈 점유 성장.
  - KOSIS 연간 품목편(광업제조업조사): 자료연도 Y 확정 공표가 Y+2년 1–2월 → Y ≤ t0-2. 금액 원값(재기준 없음).
  - KOSIS 생산능력·가동률, 제조업 국내공급지수(2020=100 현재 빈티지): 성장률만, 연평균은 이듬해 1월 공표 → year ≤ t0-1 (준-정합).
사용: python build_ext_r3b.py [--t0max 2018] [--suffix ''] (공개연도 2022 이후 자료가 필요한 t0는 T3R3_FINAL_UNLOCK=1 에서만)
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
SHX = os.path.join(HERE, '..', '..', '_shared', 'ext')


def ros_feats(t0s, cells, ksic_c, mg_c, k=5.0):
    ci = pd.read_parquet(os.path.join(SHX, 'r3ros_cell_issueyear.parquet'))
    ci = ci[ci.issue_year >= min(t0s) - 9]
    ci['cell'] = ci.ksic + '|' + ci.mg
    out = []
    for t0 in t0s:
        base = ci[(ci.first_rec_year <= t0 - 2) & (ci.issue_year <= t0)]
        w10 = base[base.issue_year >= t0 - 9]; wr = base[base.issue_year >= t0 - 4]; wo = w10[w10.issue_year <= t0 - 5]
        def agg(d):
            g = d.groupby('cell')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum()
            return g.reindex(cells).fillna(0.0)
        A, R, O = agg(w10), agg(wr), agg(wo)
        ks = w10.groupby('ksic')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum()
        ks_sh = (ks.n_us_pat_sci / ks.n_us_pat.clip(lower=1)).reindex(ksic_c).fillna(0.2).values
        ks_pp = (ks.n_paper_cit / ks.n_us_pat.clip(lower=1)).reindex(ksic_c).fillna(1.0).values
        n = A.n_us_pat.values
        sci = (A.n_us_pat_sci.values + k * ks_sh) / (n + k)
        ppp = np.log1p((A.n_paper_cit.values + k * ks_pp) / (n + k))
        sr = (R.n_us_pat_sci.values + k * ks_sh) / (R.n_us_pat.values + k)
        so = (O.n_us_pat_sci.values + k * ks_sh) / (O.n_us_pat.values + k)
        out.append(pd.DataFrame(dict(cell=cells, t0=t0, ros_lnus=np.log1p(n), ros_sci=sci, ros_sci_rel=sci - ks_sh, ros_ppp=ppp,
                                     ros_sci_chg=sr - so, ros_ks_sci=ks_sh, ros_linked=(n > 0).astype(np.int8))))
    return pd.concat(out, ignore_index=True)


def oa_feats(t0s, cells, mg_c, tot_W_by_t0):
    oa = pd.read_parquet(os.path.join(SHX, 'r3oa_mg_year_ALL.parquet'))
    tot = pd.read_parquet(os.path.join(SHX, 'science_openalex_country_total_year.parquet'))
    tot = tot[tot.country == 'ALL'].set_index('year').n_works_total
    W = oa.pivot_table(index='mg', columns='year', values='n_works', aggfunc='sum').fillna(0)
    S = W.div(tot.reindex(W.columns).values, axis=1)
    out = []
    for t0 in t0s:
        assert max(W.columns[W.columns <= t0]) <= t0
        s_r = S[[y for y in range(t0 - 2, t0 + 1)]].sum(1); s_o = S[[y for y in range(t0 - 5, t0 - 2)]].sum(1)
        w_r = W[[y for y in range(t0 - 2, t0 + 1)]].sum(1); w_o = W[[y for y in range(t0 - 5, t0 - 2)]].sum(1)
        g = np.log((s_r + 1e-12) / (s_o + 1e-12)).where((w_r + w_o) >= 20)
        lvl = np.log1p(w_r)
        mg_g = g.reindex(mg_c).values; mg_l = lvl.reindex(mg_c).values
        cov = pd.Series(mg_c).isin(W.index).values
        out.append(pd.DataFrame(dict(cell=cells, t0=t0, oa_mg_g=mg_g, oa_mg_lvl=mg_l,
                                     oa_mg_int=mg_l - np.log1p(tot_W_by_t0[t0]), oa_cov=cov.astype(np.int8))))
    return pd.concat(out, ignore_index=True)


def kosis_feats(t0s, K50):
    ch = pd.read_parquet(os.path.join(SHX, 'r3kosis_ksic50_items_chain.parquet')).set_index(['ksic', 'year'])
    cu = pd.read_parquet(os.path.join(SHX, 'r3kosis_ksic50_capacity_util_annual.parquet')).dropna(subset=['year'])
    cu['year'] = cu.year.astype(int); cu = cu.set_index(['ksic', 'year'])
    ds = pd.read_parquet(os.path.join(SHX, 'r3kosis_ksic50_domsupply_annual.parquet')).set_index(['ksic', 'year'])
    def gv(tab, ks, y, col):
        try: v = tab.loc[(ks, y), col]
        except KeyError: return np.nan
        return float(v) if np.isscalar(v) else float(np.asarray(v)[0])
    rows = []
    for ks in K50:
        for t0 in t0s:
            Y = t0 - 2; r = dict(ksic=ks, t0=t0)
            g1, g0 = gv(ch, ks, Y, 'g_prod'), gv(ch, ks, Y - 1, 'g_prod')
            r['ki_gprod'] = np.clip(g1, -1, 1) if g1 == g1 else np.nan
            r['ki_gprod2'] = np.nanmean([np.clip(g1, -1, 1), np.clip(g0, -1, 1)]) if (g1 == g1 or g0 == g0) else np.nan
            gs = gv(ch, ks, Y, 'g_ship'); r['ki_gship'] = np.clip(gs, -1, 1) if gs == gs else np.nan
            di = gv(ch, ks, Y, 'd_inv_ship'); r['ki_dinv'] = np.clip(di, -0.5, 0.5) if di == di else np.nan
            iv = gv(ch, ks, Y, 'inv_ship1'); r['ki_invship'] = np.clip(iv, 0, 2) if iv == iv else np.nan
            y1 = t0 - 1
            r['cu_cap_g1'] = gv(cu, ks, y1, 'g1_cap'); r['cu_util_g1'] = gv(cu, ks, y1, 'g1_util'); r['cu_cap_g3'] = gv(cu, ks, y1, 'g3_cap')
            r['ds_imp_g1'] = gv(ds, ks, y1, 'g1_imported'); r['ds_dom_g1'] = gv(ds, ks, y1, 'g1_dom_made'); r['ds_impdom_g3'] = gv(ds, ks, y1, 'g3_import_minus_dom')
            rows.append(r)
    return pd.DataFrame(rows)


def build(L, t0s, suffix=''):
    P = L.load(); cells = P['cells']; ksic_c = P['ksic']; mg_c = P['mg']; K50 = sorted(set(ksic_c))
    tot = P['A']['n'].sum(1)
    totW = {t0: L.wsum(tot, t0, -2, 0) for t0 in t0s}
    R = ros_feats(t0s, cells, ksic_c, mg_c)
    O = oa_feats(t0s, cells, mg_c, totW)
    CF = R.merge(O, on=['cell', 't0'], how='left')
    tw = np.concatenate([totW[t0] for t0 in t0s])
    CF = CF[tw >= 10].reset_index(drop=True)   # 확대 학습표본(≥10) 이상만 저장
    KF = kosis_feats(t0s, K50)
    CF.to_parquet(os.path.join(HERE, f'extB_cell_pub{suffix}.parquet')); KF.to_parquet(os.path.join(HERE, f'extB_ksic_pub{suffix}.parquet'))
    # 연계율: 적격 셀(5개국 t0-2..t0 ≥50)
    el = CF.merge(pd.DataFrame(dict(cell=np.tile(cells, len(t0s)), t0=np.repeat(t0s, len(cells)), tot_W=tw)), on=['cell', 't0'])
    el = el[el.tot_W >= 50]; el['ksic'] = el.cell.str[:5]
    lk = el.groupby('t0').agg(n_elig=('cell', 'size'), ros_linked=('ros_linked', 'mean'), oa_cov=('oa_cov', 'mean'),
                              oa_g_nonmiss=('oa_mg_g', lambda s: s.notna().mean()))
    kk = KF.groupby('t0')[['ki_gprod', 'ki_dinv', 'cu_cap_g1', 'ds_imp_g1']].agg(lambda s: s.notna().sum()).add_prefix('nksic_')
    lk = lk.join(kk); lk.to_csv(os.path.join(HERE, f'extB_link{suffix}.csv'), encoding='utf-8-sig')
    lks = el.groupby('ksic').agg(ros_linked=('ros_linked', 'mean'), oa_cov=('oa_cov', 'mean'))
    lks.to_csv(os.path.join(HERE, f'extB_link_ksic{suffix}.csv'), encoding='utf-8-sig')
    print(lk.round(3).to_string()); print('KSIC 연결: RoS', int((lks.ros_linked > 0).sum()), '/50, OpenAlex 셀', int((lks.oa_cov > 0).sum()), '/50')
    return CF, KF


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--t0max', type=int, default=2018); ap.add_argument('--suffix', default=''); a = ap.parse_args()
    os.environ.setdefault('T3R3_VAR', 'pub')
    import r3lib as L
    assert a.t0max + 3 < L.F_YEAR or os.environ.get('T3R3_FINAL_UNLOCK') == '1'
    build(L, list(range(2005, a.t0max + 1)), a.suffix)
