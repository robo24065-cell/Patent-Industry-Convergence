# -*- coding: utf-8 -*-
"""랜드마크 이산시간 위험(hazard) 모형.
학습 단위 = (특허, 기준일 L_t, 창 안의 납부기한 d_j). y=그 기한에서 불납 소멸(E==d_j).
- 결정은 실제 기록(납부 pd<=C, 소멸등록 cd<=C)에서만 만든다 -> 거짓 음성 없음.
- d_j + 450일 <= C 인 결정만(결과 확정).
예측: 창 안의 기한 d_1..d_n 에 대해 P = 1 - Π(1-h_j).
"""
import os, numpy as np, pandas as pd
import t4lib as T

LAG_OK = 450


def _dec_cols(f, d, L, j, k):
    """결정 행 추가 피처"""
    out = f.copy()
    Ld = np.datetime64(L)
    out['lead'] = T.yrs(d, np.full(len(d), Ld))
    out['j'] = j
    out['k'] = k
    out['age_d'] = out.age + out.lead
    out['tier_d'] = np.isin(k, [4, 7, 10, 13]).astype(int)
    out['exp_d'] = out.to_expiry - out.lead
    return out


def train_decisions(L, C):
    fn = os.path.join(T.CACHE, f'dec_{L.replace("-", "")}_{C.replace("-", "")}.parquet')
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    f = T.features(L)
    Ld, Cd = np.datetime64(L), np.datetime64(C)
    end = T.add_years(np.array([Ld]), T.H_YEARS)[0]
    lim = min(end, Cd - np.timedelta64(LAG_OK, 'D'))
    pids = f.pid.values
    py = pd.Series(f.paid_year.values, index=pids)
    # 납부 결정(y=0)
    p = pay[(pay.seq > 0) & pay.pid.isin(pids) & (pay.pd <= Cd) & (pay.due > Ld) & (pay.due <= lim)]
    p = p[p.s.values > py.reindex(p.pid.values).values]
    r0 = pd.DataFrame({'pid': p.pid.values, 'd': p.due.values, 'k': p.s.values, 'y': 0})
    # 소멸 결정(y=1)
    l = lap[lap.pid.isin(pids) & (lap.cd <= Cd) & (lap.E > Ld) & (lap.E <= lim)]
    r1 = pd.DataFrame({'pid': l.pid.values, 'd': l.E.values, 'k': l.lmax.values.astype(int) + 1, 'y': 1})
    r = pd.concat([r0, r1]).drop_duplicates(['pid', 'd']).sort_values(['pid', 'd'])
    r['j'] = r.groupby('pid').cumcount() + 1
    fi = f.set_index('pid').loc[r.pid.values].reset_index()
    out = _dec_cols(fi, r.d.values.astype('datetime64[D]'), L, r.j.values, r.k.values)
    out['y'] = r.y.values
    out['d'] = r.d.values
    out.to_parquet(fn, index=False)
    return out


def eval_decisions(f):
    """평가 행(기준일 L 하나의 피처 프레임)에 대해 창 안 기한 전개. 반환: 결정 프레임 + row index"""
    D = T.load(); pt = D['pt']
    L = f.L.iloc[0]
    Ld = np.datetime64(L)
    end = T.add_years(np.array([Ld]), T.H_YEARS)[0]
    rows = []
    rd = pt.rd.values[f.pid.values]
    exp = pt.exp.values[f.pid.values]
    for j in range(1, 5):
        k = f.paid_year.values + j
        d = T.add_years(rd, k.astype(int) - 1)
        ok = (d <= end) & (d < exp) & (d > Ld)
        idx = np.where(ok)[0]
        if len(idx) == 0: continue
        sub = _dec_cols(f.iloc[idx].reset_index(drop=True), d[idx], L, np.full(len(idx), j), k[idx])
        sub['row'] = idx
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def combine(dec, h, n):
    """P = 1 - Π(1-h)"""
    lg = pd.Series(np.log1p(-np.clip(h, 1e-6, 1 - 1e-6))).groupby(dec.row.values).sum()
    P = np.zeros(n)
    P[lg.index.values] = 1 - np.exp(lg.values)
    return P
