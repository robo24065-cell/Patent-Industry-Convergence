# -*- coding: utf-8 -*-
"""산업연관표(한국은행 접속 산업연관표, 통합소분류 165부문, 국산거래표) × KOSIS 광공업생산지수 → KSIC 5자리별 '수요산업 경기' 피처.
연결: 특허 KSIC 5자리 → (IO-KSIC 대조표, KSIC 8자리 앞 5자리) → IO 기본부문 4자리 → 통합소분류 3자리(앞 3자리) = 공급 부문 i
      수요 부문 j(제조·광업 중간수요) → IO-KSIC 대조표 → KSIC 3자리(없으면 2자리) 광공업생산지수
피처(기준일 L, 생산지수는 L-2개월까지):
  io_dn_g  = Σ_j w_ij · (수요부문 j 생산지수 직전 12개월 평균 / 그 전 12개월 − 1),   w_ij = i 의 국산 중간수요 중 j 몫(제조·광업 j 로 정규화)
  io_dn_g3 = 같은 가중으로 3년 전 대비 증감
  io_dn_rel = io_dn_g − 자기 부문(i) 생산 증감 → 수요산업이 자기 산업보다 더 위축되는가
구조표: L연도 < 2019 → 2010년 표, 이후 → 2015년 표(접속표; 부문분류만 2020 기준으로 조정). 구조(가중)는 결과와 무관한 산업 간 거래 비중이다.
"""
import os, numpy as np, pandas as pd
import r2lib as R

IO_DIR = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\산업연관표'
_C = {}


def io_map():
    """KSIC 8/5자리 → IO 기본부문 4자리 → 통합소분류 3자리"""
    if 'map' not in _C:
        m = pd.read_excel(os.path.join(IO_DIR, 'IO코드와 KSIC 및 HS코드 대조표.xlsx'), sheet_name=0, header=None, skiprows=2, dtype=str)
        m.columns = ['io', 'nm', 'ksic']
        m[['io', 'nm']] = m[['io', 'nm']].ffill()
        m = m.dropna(subset=['ksic'])
        m = m[m.ksic.str.fullmatch(r'\d{5,8}')]
        m['k5'] = m.ksic.str[:5]
        m['k3'] = m.ksic.str[:3]
        m['sec'] = m.io.str[:3]
        _C['map'] = m
    return _C['map']


def io_table(year):
    k = ('tab', year)
    if k not in _C:
        sh = f'{str(year)[2:]}-A표_국산거래표(생산자)'
        d = pd.read_excel(os.path.join(IO_DIR, '2010-2015-2020_생산자가격(경상)_통합소분류_거래표 및 각종 계수표.xlsx'), sheet_name=sh, header=None)
        codes = d.iloc[4, 2:].astype(str).str.strip().values
        rows = d.iloc[6:, 0].astype(str).str.strip().values
        Z = d.iloc[6:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        Z = pd.DataFrame(Z, index=rows, columns=codes)
        Z = Z.loc[[r for r in Z.index if r.isdigit() and len(r) == 3], [c for c in Z.columns if c.isdigit() and len(c) == 3]]
        _C[k] = Z
        _C[('nm', year)] = pd.Series(d.iloc[6:, 1].astype(str).str.strip().values, index=rows)
    return _C[k]


def weights(year):
    """공급 부문 i(3자리) × 수요 부문 j(제조·광업) 가중"""
    k = ('w', year)
    if k not in _C:
        Z = io_table(year)
        m = io_map()
        mfg = sorted(set(m[m.k3.str[:2].astype(int).between(5, 33)].sec) & set(Z.columns))
        W = Z[mfg].copy()
        W = W.div(W.sum(1).replace(0, np.nan), axis=0)
        _C[k] = W
    return _C[k]


def sec_codes():
    """수요 부문 j → 생산지수 코드 목록(C+KSIC3, 없으면 C+KSIC2)"""
    if 'sc' not in _C:
        import feat2 as F2
        cols = set(F2._macro()['prod'].columns)
        m = io_map()
        out = {}
        for s, g in m.groupby('sec'):
            cs = []
            for k3 in sorted(set(g.k3)):
                c = 'C' + k3 if ('C' + k3) in cols else ('C' + k3[:2] if ('C' + k3[:2]) in cols else ('B' + k3[:2] if ('B' + k3[:2]) in cols else None))
                if c: cs.append(c)
            out[s] = sorted(set(cs))
        _C['sc'] = out
    return _C['sc']


def ksic_sec():
    """50개 KSIC → 공급 부문(3자리) — 대조표 행 수가 가장 많은 부문"""
    m = io_map()
    return m.groupby('k5').sec.agg(lambda s: s.value_counts().index[0])


def io_at(L):
    import feat2 as F2
    t = F2.macro_at(L)                    # index: 산업코드, cols prod_yoy, prod_3y ... (L-2개월까지)
    y = int(str(L)[:4])
    W = weights(2010 if y < 2019 else 2015)
    sc = sec_codes()
    gj = pd.Series({s: t.prod_yoy.reindex(cs).mean() if cs else np.nan for s, cs in sc.items()})
    gj3 = pd.Series({s: t.prod_3y.reindex(cs).mean() if cs else np.nan for s, cs in sc.items()})

    def wavg(g):
        g = g.reindex(W.columns)
        ok = g.notna().values
        num = (W.values[:, ok] * g.values[ok]).sum(1)
        den = W.values[:, ok].sum(1)
        return pd.Series(num / np.where(den > 0, den, np.nan), index=W.index)
    dn, dn3 = wavg(gj), wavg(gj3)
    ks = ksic_sec()
    own = gj.reindex(ks.values).values
    out = pd.DataFrame({'io_dn_g': dn.reindex(ks.values).values, 'io_dn_g3': dn3.reindex(ks.values).values}, index=ks.index)
    out['io_dn_rel'] = out.io_dn_g - own
    return out


def link_table():
    """보고서용 연계표: KSIC 5자리 → IO 부문, 수요 상위 5개 부문과 비중"""
    ks = ksic_sec(); W = weights(2015); Z = io_table(2015)
    nm = _C[('nm', 2015)]
    rows = []
    for k, s in ks.items():
        if s not in W.index: continue
        top = W.loc[s].sort_values(ascending=False).head(5)
        tot = Z.loc[s].sum()
        rows.append(dict(ksic=k, io_sec=s, io_sec_name=nm.get(s, ''), mfg_intermediate_share=round(float(Z.loc[s, W.columns].sum() / tot), 3) if tot else np.nan,
                         top_demand=' | '.join(f'{j}:{nm.get(j, "")}:{v:.2f}' for j, v in top.items()), n_prodidx_codes=len(sec_codes().get(s, []))))
    return pd.DataFrame(rows)
