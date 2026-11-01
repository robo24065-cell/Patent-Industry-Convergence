# -*- coding: utf-8 -*-
"""R9 s04 (R8 NONCIRC_R8/s03_ctx.py 사본, 체계만 txt 로 교체; prov_check = 제공 재계산으로 R8 ctx_prov 와 같은지 대조): IPC 메인그룹(mg)마다 '산업 맥락(CTX)' 16피처를 산업 배정 체계별로 만든다. t0 년 말까지 공개된 문서만(창 t0-2..t0, 직전 t0-5..t0-3).
체계: prov  = 제공 KSIC 50 (R3/cache_pub/panel_ccy, 동결 캐시 읽기만)
      conc  = 연계표 argmax 27코드 (ESSENTIAL_R7/cache_conc_pub/panel_ccy, 읽기만)
      concm = 연계표 다중배정 (ESSENTIAL_R7/cache_concm_pub/panel_ccy, 읽기만) — 민감도
KOSIS 생산지수 3피처: prov = R3/ext_ksic_t0_pub.parquet, conc·concm = ESSENTIAL_R7/ext_pi_conc_pub.parquet (둘 다 동결 산출물).
출력: ctx_{sys}.parquet (mg, t0, c_*)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
import numpy as np, pandas as pd
Y0, Y1 = 2000, 2021; T0S = range(2005, 2019)
SYS = {'txt': (os.path.join(HERE, 'cache_txt_pub', 'panel_ccy.parquet'), os.path.join(T3, 'R3', 'ext_ksic_t0_pub.parquet')),
       'prov_check': (os.path.join(T3, 'R3', 'cache_pub', 'panel_ccy.parquet'), os.path.join(T3, 'R3', 'ext_ksic_t0_pub.parquet'))}
PI = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']


def arrays(fn, key_cols=('ksic', 'mg')):
    df = pd.read_parquet(fn, columns=list(key_cols) + ['cc', 'year', 'n'])
    assert df.year.max() <= Y1
    df['kr'] = np.where(df.cc == 'KR', df.n, 0.0); df['fx'] = np.where(df.cc != 'KR', df.n, 0.0)
    g = df.groupby(list(key_cols) + ['year'])[['n', 'kr', 'fx']].sum().reset_index()
    cells = g[list(key_cols)].drop_duplicates().reset_index(drop=True); cells['ci'] = np.arange(len(cells))
    g = g.merge(cells, on=list(key_cols))
    A = {c: np.zeros((len(cells), Y1 - Y0 + 1)) for c in ['n', 'kr', 'fx']}
    for c in A: A[c][g.ci.values, g.year.values - Y0] = g[c].values
    return cells, A


def ws(x, a, b):
    return x[:, a - Y0:b - Y0 + 1].sum(1)


mgc, mgA = arrays(os.path.join(HERE, 'cache_mg_pub', 'panel_ccy.parquet'))
mg_index = pd.Series(np.arange(len(mgc)), index=mgc.mg.values)
for sysname, (pfn, pifn) in SYS.items():
    t = time.time()
    cells, A = arrays(pfn)
    pi = pd.read_parquet(pifn)[['ksic', 't0'] + PI]
    out = []
    for t0 in T0S:
        d = cells[['ksic', 'mg']].copy()
        for nm, arr in (('tot', A['n']), ('kr', A['kr']), ('fx', A['fx'])):
            d[nm + '_W'] = ws(arr, t0 - 2, t0); d[nm + '_P'] = ws(arr, t0 - 5, t0 - 3)
        I = d.groupby('ksic')[['tot_W', 'tot_P', 'kr_W', 'kr_P', 'fx_W', 'fx_P']].transform('sum')
        lg = lambda a, b: np.log((a + 1) / (b + 1))
        d['peer_k'] = lg(I.kr_W - d.kr_W, I.kr_P - d.kr_P); d['peer_t'] = lg(I.tot_W - d.tot_W, I.tot_P - d.tot_P)
        d['peer_f'] = lg(I.fx_W - d.fx_W, I.fx_P - d.fx_P); d['peer_sh'] = lg(I.kr_W - d.kr_W, I.tot_W - d.tot_W)
        d['cell_k'] = lg(d.kr_W, d.kr_P); d['cell_f'] = lg(d.fx_W, d.fx_P)
        d = d.merge(pi[pi.t0 == t0].drop(columns='t0'), on='ksic', how='left')
        d = d[(d.tot_W > 0) | (d.tot_P > 0)]
        M = d.groupby('mg')[['tot_W', 'tot_P', 'kr_W']].transform('sum')
        d['w'] = d.tot_W / M.tot_W.replace(0, np.nan); d['wp'] = d.tot_P / M.tot_P.replace(0, np.nan)
        d['big'] = d.tot_W >= 10
        rows = {}
        gb = d.groupby('mg')
        wsum = lambda c: (d[c] * d.w).groupby(d.mg).sum(min_count=1)
        o = pd.DataFrame(index=gb.size().index)
        o['mg_tot_sys'] = gb.tot_W.sum(); o['mg_kr_sys'] = gb.kr_W.sum()
        for src, dst in (('peer_k', 'c_ind_kmom'), ('peer_t', 'c_ind_tmom'), ('peer_f', 'c_ind_fmom'), ('peer_sh', 'c_ind_krsh'), ('cell_k', 'c_cell_kwavg')):
            o[dst] = wsum(src)
        for c in PI:   # 결측 생산지수는 가중치 재정규화
            ok = d[c].notna(); num = (d[c].fillna(0) * d.w * ok).groupby(d.mg).sum(); den = (d.w * ok).groupby(d.mg).sum()
            o['c_' + c] = num / den.replace(0, np.nan)
        o['c_krsh_rel'] = np.log((o.mg_kr_sys + 1) / (o.mg_tot_sys + 1)) - o.c_ind_krsh
        o['c_n_ind'] = (d.w >= 0.05).groupby(d.mg).sum()
        o['c_entropy'] = (-(d.w * np.log(d.w.where(d.w > 0)))).groupby(d.mg).sum(min_count=1)
        db = d[d.big]
        o['c_cell_kmax'] = db.groupby('mg').cell_k.max(); o['c_cell_fmax'] = db.groupby('mg').cell_f.max()
        o['c_newind_sh'] = (d.tot_W * (d.tot_P == 0)).groupby(d.mg).sum() / o.mg_tot_sys.replace(0, np.nan)
        o['c_mix_shift'] = 0.5 * (d.w.fillna(0) - d.wp.fillna(0)).abs().groupby(d.mg).sum()
        o.loc[(gb.tot_P.sum() == 0) | (o.mg_tot_sys == 0), 'c_mix_shift'] = np.nan
        o.loc[o.mg_tot_sys == 0, [c for c in o.columns if c.startswith('c_') and c != 'c_mix_shift']] = np.nan
        mgt = ws(mgA['n'], t0 - 2, t0)[mg_index.reindex(o.index).fillna(-1).astype(int).clip(lower=0).values]
        o['c_cov'] = np.clip(o.mg_tot_sys / np.maximum(mgt, 1), 0, 1)
        o['t0'] = t0
        out.append(o.reset_index().rename(columns={'index': 'mg'}))
    O = pd.concat(out, ignore_index=True).drop(columns=['mg_tot_sys', 'mg_kr_sys'])
    O.to_parquet(os.path.join(HERE, f'ctx_{sysname}.parquet'))
    print(sysname, O.shape, round(time.time() - t), flush=True)
    print(O[O.t0 == 2015].describe().T[['count', 'mean', '50%', 'std']].round(3).to_string(), flush=True)
