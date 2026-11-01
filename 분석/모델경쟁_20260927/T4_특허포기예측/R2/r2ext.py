# -*- coding: utf-8 -*-
"""T4 2라운드 외부·추가 피처. 모두 기준일 L 시점에 알 수 있던 값만 쓴다.

접두사로 그룹을 구분한다.
 tm_  제공 융합데이터(5개국 KSIC별 특허 메타) 기술셀 출원 모멘텀 — '기술 퇴조' 신호. 출원연도 t <= (L월>=7 ? L연도-2 : L연도-3) (18개월 공개 완결분)
 tr_  무역(KSIC 5자리 ↔ HS6 연계표 + UN Comtrade 한국 수입, 분석/fusion/cn_import_yearly_ksic5.csv). 연도 t <= (L월>=7 ? L연도-1 : L연도-2)
 ks_  KOSIS 광업제조업조사 세세분류 5자리(출하·사업체·종사자·부가가치). 연도 t <= L연도-2 (t+1년 말 공표)
 io_  산업연관표(수요산업 구조) × KOSIS 광공업생산지수(수요산업 생산 증감). 생산지수는 L-2개월까지
 ka_  KSIC × 연차대 L 이전 불납률(산업 포기 곡선) — 제공 KSIC 라벨 × 등록원부
 bz_  권리목록(business_rights) 출원인 L 이전 상표·디자인 등록 활동(기업 활력)
"""
import os, sys, numpy as np, pandas as pd
import r2lib as R
T = R.T

BASE = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
TRADE_CSV = os.path.join(BASE, '분석', 'fusion', 'cn_import_yearly_ksic5.csv')
KOSIS_CSV = os.path.join(BASE, '공모전데이터', '작업데이터', 'kosis_광업제조업_50개산업.csv')
_C = {}


def _Lym(L):
    s = str(L)
    return int(s[:4]), int(s[5:7])


# ================================================================ 공통: pid → 메인그룹
def pid_mg():
    if 'mg' not in _C:
        pt = T.load()['pt']
        p = pd.read_parquet(os.path.join(R.T4, 'cache', 'patent.parquet'), columns=['rgst_no', 'ipcCd'])
        p = p.drop_duplicates('rgst_no').set_index('rgst_no').ipcCd
        ipc1 = p.reindex(pt.rgst_no.values).fillna('').str.split(',').str[0].str.replace(r'\(.*$', '', regex=True).str.strip()
        mg = ipc1.str.split('/').str[0].str.strip().str.replace(r'\s+', ' ', regex=True)
        _C['mg'] = mg.values
    return _C['mg']


# ================================================================ tm_: 기술셀 모멘텀(제공 데이터)
def _meta_tab(unit):
    k = 'meta_' + unit
    if k not in _C:
        d = pd.read_parquet(os.path.join(R.T.CACHE, f'meta_{unit}.parquet'))
        key = {'mg': 'mg', 'sub': 'sub', 'ksic': 'ksic'}[unit]
        d = d.rename(columns={key: 'key'})
        w = d.pivot_table(index='key', columns='yr', values='n', aggfunc='sum', fill_value=0)
        kr = d[d.cc == 'KR'].pivot_table(index='key', columns='yr', values='n', aggfunc='sum', fill_value=0).reindex(w.index, fill_value=0)
        nc = d[d.cc != 'CN'].pivot_table(index='key', columns='yr', values='n', aggfunc='sum', fill_value=0).reindex(w.index, fill_value=0)
        _C[k] = dict(w=w, kr=kr, nc=nc)
    return _C[k]


def _mom(M, t):
    """셀별 최근 2년 합 / 그 전 2년 합 (로그), 전체 대비 상대, 정점 대비 낙폭"""
    cols = [c for c in M.columns if c <= t]
    def s(a, b):
        cs = [c for c in cols if a <= c <= b]
        return M[cs].sum(1) if cs else pd.Series(0, index=M.index)
    a, b = s(t - 1, t), s(t - 3, t - 2)
    g = np.log((a + 1) / (b + 1))
    tot = np.log((a.sum() + 1) / (b.sum() + 1))
    roll = M[cols].T.rolling(2).sum().T if len(cols) >= 2 else M[cols]
    peak = roll.max(1)
    dd = (a / peak.replace(0, np.nan)).fillna(1.0)
    return g - tot, dd, np.log1p(a)


def meta_feats(d, cols):
    out = {c: np.full(len(d), np.nan, dtype='float32') for c in cols}
    pt = T.load()['pt']
    mg = pid_mg()
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        y, m = _Lym(L)
        t = y - 2 if m >= 7 else y - 3
        pids = d.pid.values[idx]
        for unit, keyv in [('mg', mg[pids]), ('sub', pt.ipc_sub.values[pids]), ('ksic', d.ksic.astype(str).values[idx])]:
            need = [c for c in cols if c.startswith(f'tm_{unit}_')]
            if not need: continue
            M = _meta_tab(unit)
            rg_w, dd_w, sz_w = _mom(M['w'], t)
            rg_kr, dd_kr, _ = _mom(M['kr'], t)
            rg_nc, _, _ = _mom(M['nc'], t)
            tab = dict(rg=rg_w, dd=dd_w, rgkr=rg_kr, ddkr=dd_kr, rgnc=rg_nc, sz=sz_w)
            for c in need:
                v = tab[c.split('_')[-1]]
                out[c][idx] = v.reindex(keyv).values
    for c in cols: d[c] = out[c]
    return d


# ================================================================ tr_: 무역
def _trade():
    if 'tr' not in _C:
        c = pd.read_csv(TRADE_CSV, dtype={'ksic': str})
        w = c.pivot_table(index='ksic', columns='year', values='wld_imp_usd')
        cn = c.pivot_table(index='ksic', columns='year', values='cn_imp_usd')
        _C['tr'] = (w, cn)
    return _C['tr']


def trade_at(L):
    w, cn = _trade()
    y, m = _Lym(L)
    t = y - 1 if m >= 7 else y - 2
    g1 = np.log(w[t] / w[t - 1]); g3 = np.log(w[t] / w[t - 3])
    sh = cn[t] / w[t]; sh3 = cn[t - 3] / w[t - 3]
    return pd.DataFrame({'tr_imp_g1': g1 - g1.mean(), 'tr_imp_rg3': g3 - g3.mean(), 'tr_cn_share': sh, 'tr_cn_dshare3': sh - sh3,
                         'tr_cn_rg3': np.log(cn[t] / cn[t - 3]) - np.log(cn[t] / cn[t - 3]).mean()})


# ================================================================ ks_: KOSIS 광업제조업조사 5자리
def _kosis():
    if 'ks' not in _C:
        k = pd.read_csv(KOSIS_CSV, dtype={'세세분류코드': str})
        k = k[k['기준'] == '10차']            # 개발 구간(t<=2016)은 10차만. 11차 연결은 final/future 에서 2020 중복연도 비율로 접속
        k11 = pd.read_csv(KOSIS_CSV, dtype={'세세분류코드': str})
        k11 = k11[k11['기준'] == '11차']
        piv = {}
        for code, nm in [('T04', 'ship'), ('T01', 'estab'), ('T02', 'emp'), ('T06', 'va')]:
            a = k[k['항목코드'] == code].pivot_table(index='세세분류코드', columns='연도', values='값')
            b = k11[k11['항목코드'] == code].pivot_table(index='세세분류코드', columns='연도', values='값')
            if 2020 in a.columns and 2020 in b.columns:     # 11차 계열을 2020 비율로 10차 수준에 접속
                r = (a[2020] / b[2020]).replace([np.inf, -np.inf], np.nan)
                for yy in [c for c in b.columns if c > 2020]:
                    a[yy] = b[yy] * r
            piv[nm] = a
        _C['ks'] = piv
    return _C['ks']


def _lastyr(cols, t):
    c = [x for x in cols if x <= t]
    return max(c) if c else None


def kosis_at(L):
    P = _kosis()
    y, _ = _Lym(L)
    t = y - 2
    out = {}
    for nm in ['ship', 'estab', 'emp', 'va']:
        M = P[nm]
        a = _lastyr(M.columns, t)
        b = _lastyr(M.columns, a - 3) if a is not None else None
        if a is None or b is None:
            out[nm] = pd.Series(np.nan, index=M.index); continue
        out[nm] = (np.log(M[a] / M[b]) / (a - b) * 3).replace([np.inf, -np.inf], np.nan)
    a = _lastyr(P['ship'].columns, t)
    va = (P['va'][a] / P['ship'][a]) if a is not None else pd.Series(np.nan, index=P['ship'].index)
    return pd.DataFrame({'ks_ship_rg3': out['ship'] - out['ship'].mean(), 'ks_estab_g3': out['estab'], 'ks_emp_g3': out['emp'], 'ks_va_ratio': va})


def ksic_level(d, cols, fn):
    out = {c: np.full(len(d), np.nan, dtype='float32') for c in cols}
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        tab = fn(L)
        ks = d.ksic.astype(str).values[idx]
        for c in cols:
            out[c][idx] = tab[c].reindex(ks).values
    for c in cols: d[c] = out[c]
    return d


# ================================================================ ka_: KSIC × 연차대 L 이전 불납률
KB = [0, 4, 6, 9, 12, 99]


def ka_table(L, key='ksic', yrs_back=3):
    k = ('ka', L, key, yrs_back)
    if k in _C: return _C[k]
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    Ld = np.datetime64(L); L0 = T.add_years(np.array([Ld]), -yrs_back)[0]
    p = pay[(pay.seq > 0) & (pay.pd <= Ld) & (pay.pd > L0)]
    l = lap[(lap.cd <= Ld) & (lap.cd > L0)]
    kv = pt[key].astype(str).values
    r = pd.DataFrame({'g': np.r_[kv[p.pid.values], kv[l.pid.values]], 'k': np.r_[p.s.values, l.lmax.values.astype(int) + 1],
                      'y': np.r_[np.zeros(len(p)), np.ones(len(l))]})
    r['kb'] = pd.cut(r.k, KB, labels=False)
    gk = r.groupby('kb').y.mean()
    t = r.groupby(['g', 'kb']).y.agg(['sum', 'size'])
    prior = gk.reindex(t.index.get_level_values(1)).values
    rate = (t['sum'] + 20 * prior) / (t['size'] + 20)
    rel = rate / prior                         # 같은 연차대 전체 대비 상대 위험(달력 수준 이동 제거)
    _C[k] = (rate, rel, gk)
    return _C[k]


def ka_feats(d, cols):
    pt = T.load()['pt']
    out = {c: np.full(len(d), np.nan, dtype='float32') for c in cols}
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        kb = pd.cut(pd.Series(d.next_annl.values[idx] if 'next_annl' in d else d.k.values[idx]), KB, labels=False).values
        for c in cols:
            key = 'ksic' if '_ksic' in c else 'ipc_sub'
            rate, rel, gk = ka_table(L, key)
            kv = (d.ksic.astype(str).values[idx] if key == 'ksic' else pt.ipc_sub.values[d.pid.values[idx]])
            keys = pd.MultiIndex.from_arrays([kv, kb])
            src = rel if c.endswith('_rel') else rate
            out[c][idx] = src.reindex(keys).values
    for c in cols: d[c] = out[c]
    return d


# ================================================================ 진입점
def add(d, cols):
    d = d.copy()
    groups = {}
    for c in cols:
        groups.setdefault(c.split('_')[0], []).append(c)
    for g, cs in groups.items():
        if g == 'tm': d = meta_feats(d, cs)
        elif g == 'tr': d = ksic_level(d, cs, trade_at)
        elif g == 'ks': d = ksic_level(d, cs, kosis_at)
        elif g == 'io':
            import r2io
            d = ksic_level(d, cs, r2io.io_at)
        elif g == 'ka': d = ka_feats(d, cs)
        elif g == 'bz':
            import r2biz
            d = r2biz.add(d, cs)
        else:
            raise KeyError(f'알 수 없는 피처 {cs}')
    return d
