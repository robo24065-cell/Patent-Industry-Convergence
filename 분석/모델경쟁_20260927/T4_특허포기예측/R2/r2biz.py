# -*- coding: utf-8 -*-
"""권리목록(business_rights.sqlite → 1라운드 추출 집계 cache/biz_counts.parquet, 읽기 전용) — 출원인(법인) '사업 활력' 비율 피처.
기준일 L 의 전년도(L연도-1)까지 **등록된** 권리만 센다(스냅샷 누수 차단). 절대건수는 달력 드리프트라 쓰지 않고 비율·증가율만.
 bz_tm_act3   : 최근 3년 상표·디자인 등록이 1건 이상(0/1)  — 사업을 계속하는 기업인가
 bz_tm_share3 : 최근 3년 등록 중 상표·디자인 비중 — 기술보다 브랜드·제품 중심으로 이동?
 bz_pat_gr    : 특허 등록 최근 3년 / 그 전 3년 (로그) — 기술 투자 축소?
 bz_ext_share : 전체 특허 중 제공 50개 산업 밖 비중(L연도-1 누적) — 핵심 사업 밖 특허 비중
조회 성공(status OK) 출원인만. 나머지는 결측.
"""
import os, numpy as np, pandas as pd
import r2lib as R
_C = {}


def table():
    if 'b' not in _C:
        q = pd.read_parquet(os.path.join(R.T4, 'cache', 'biz_query.parquet'))
        ok = set(q[q.status == 'OK'].applicant_cd)
        r = pd.read_parquet(os.path.join(R.T4, 'cache', 'biz_counts.parquet'))
        r = r[r.applicant_cd.isin(ok)].copy()
        r['y'] = pd.to_numeric(r.y, errors='coerce')
        r['typ'] = np.select([r.t == '10', r.t == '30', r.t.str[0] == '4'], ['pat', 'des', 'tm'], 'oth')
        b = r.groupby(['applicant_cd', 'y', 'typ']).n.sum().unstack('typ', fill_value=0).reset_index()
        _C['b'] = (b, ok)
    return _C['b']


def add(d, cols):
    b, ok = table()
    pt = R.T.load()['pt']
    acd = d.acd.values if 'acd' in d else pt.acd.values[d.pid.values]
    Y = pd.to_datetime(pd.Series(d.L.astype(str).values)).dt.year.values
    port = None
    out = {c: np.full(len(d), np.nan, dtype='float32') for c in cols}
    for Yv in np.unique(Y):
        def win(a, z):
            return b[(b.y >= a) & (b.y <= z)].groupby('applicant_cd')[['pat', 'des', 'tm']].sum()
        r3, p3, cum = win(Yv - 3, Yv - 1), win(Yv - 6, Yv - 4), win(1900, Yv - 1)
        # 제공 범위(50개 KSIC) 안 등록특허 수: 등록원부 기준 L연도-1 까지
        reg = pd.DatetimeIndex(pt.rd.values).year.values <= Yv - 1
        inside = pd.Series(pt.acd.values[reg]).value_counts()
        t = pd.DataFrame(index=cum.index)
        t['bz_tm_act3'] = ((r3.tm + r3.des).reindex(t.index).fillna(0) > 0).astype(float)
        tot3 = (r3.tm + r3.des + r3.pat).reindex(t.index)
        t['bz_tm_share3'] = ((r3.tm + r3.des).reindex(t.index) / tot3.replace(0, np.nan))
        t['bz_pat_gr'] = np.log((r3.pat.reindex(t.index).fillna(0) + 1) / (p3.pat.reindex(t.index).fillna(0) + 1))
        t['bz_ext_share'] = 1 - (inside.reindex(t.index).fillna(0) / cum.pat.replace(0, np.nan)).clip(upper=1)
        idx = np.where(Y == Yv)[0]
        a = acd[idx]
        inok = np.array([x in ok for x in a])
        v = t.reindex(a)
        for c in cols:
            x = v[c].values.astype('float32')
            if c == 'bz_tm_act3': x = np.where(np.isnan(x), 0, x)
            x[~inok] = np.nan
            out[c][idx] = x
    for c in cols: d[c] = out[c]
    return d
