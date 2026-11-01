# -*- coding: utf-8 -*-
"""관계(이웃) 피처: 같은 발명자팀·같은 출원인×출원연도(형제 특허)·같은 출원인×KSIC의 L 이전 불납 이력.
모두 L 이전(납부 pd<=L, 소멸등록 cd<=L)에 결론 난 결정만 쓰고, 자기 특허 기여분은 뺀다."""
import os, numpy as np, pandas as pd
import t4lib as T

_INV = {}


def inv_pairs():
    if 'x' in _INV: return _INV['x']
    D = T.load(); pt = D['pt']
    d = pd.read_parquet(os.path.join(T.CACHE, 'inventor.parquet'))
    pid = pd.Series(pt.pid.values, index=pt.rgst_no.values)
    d['pid'] = d.rgst_no.map(pid)
    d = d[d.pid.notna()].copy(); d['pid'] = d.pid.astype(int)
    d['key'] = pt.acd.values[d.pid.values] + '|' + d.inventorName.str.replace(' ', '')
    d = d.drop_duplicates(['pid', 'key'])[['pid', 'key']]
    _INV['x'] = d
    return d


def _rate_by_group(gkey, ren, lap, ren2, lap2, alive, g, prior=5):
    """gkey: pid-> group key(array len n). 그룹 합에서 자기 몫을 뺀 평활 불납률"""
    df = pd.DataFrame({'k': gkey, 'ren': ren, 'lap': lap, 'ren2': ren2, 'lap2': lap2, 'alive': alive, 'one': 1})
    s = df.groupby('k')[['ren', 'lap', 'ren2', 'lap2', 'alive', 'one']].transform('sum')
    R = s.ren - df.ren; Lp = s.lap - df.lap; R2 = s.ren2 - df.ren2; L2 = s.lap2 - df.lap2
    rate = (Lp + prior * g) / (Lp + R + prior)
    rate2 = (L2 + prior * g) / (L2 + R2 + prior)
    n_oth = s.one - 1
    alive_r = (s.alive - df.alive) / n_oth.replace(0, np.nan)
    return rate.values, rate2.values, n_oth.values, alive_r.values, (Lp).values


def soc_features(L):
    T.guard_L(L)
    fn = os.path.join(T.CACHE, f'soc_{L.replace("-", "")}.parquet')
    if os.path.exists(fn): return pd.read_parquet(fn)
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    n = len(pt); Ld = np.datetime64(L); L1 = T.add_years(np.array([Ld]), -1)[0]
    reg = pt.rd.values <= Ld
    pr = pay[(pay.seq > 0) & (pay.pd <= Ld)]
    ren = np.bincount(pr.pid.values, minlength=n).astype(float)
    ren1 = np.bincount(pr[pr.pd > L1].pid.values, minlength=n).astype(float)
    lp = lap[lap.cd <= Ld]
    lpp = np.bincount(lp.pid.values, minlength=n).astype(float)
    lp1 = np.bincount(lp[lp.cd > L1].pid.values, minlength=n).astype(float)
    pids, _, _ = T.risk_set(L)
    alive = np.zeros(n); alive[pids] = 1
    g = lpp.sum() / max(lpp.sum() + ren.sum(), 1)
    out = pd.DataFrame({'pid': pids})
    # 형제: 출원인 x 출원연도
    ay = pd.DatetimeIndex(pt.ad.values).year.astype(str)
    for name, key in [('sib', pt.acd.values + '|' + ay), ('aks', pt.acd.values + '|' + pt.ksic.values),
                      ('aipc', pt.acd.values + '|' + pt.ipc_sub.values)]:
        key = np.where(reg, key, 'NOTREG_' + np.arange(n).astype(str))
        key = np.where(pt.acd.values == 'NA', 'NA_' + np.arange(n).astype(str), key)
        r, r1, no, ar, lcnt = _rate_by_group(key, ren, lpp, ren1, lp1, alive, g)
        out[name + '_rate'] = r[pids]; out[name + '_rate1y'] = r1[pids]; out[name + '_n'] = no[pids]; out[name + '_alive'] = ar[pids]
    # 발명자팀
    ip = inv_pairs()
    ip = ip[reg[ip.pid.values]]
    ip = ip.assign(ren=ren[ip.pid.values], lap=lpp[ip.pid.values], ren1=ren1[ip.pid.values], lap1=lp1[ip.pid.values], alive=alive[ip.pid.values], one=1)
    ks = ip.groupby('key')[['ren', 'lap', 'ren1', 'lap1', 'alive', 'one']].sum()
    ip = ip.join(ks, on='key', rsuffix='_k')
    for c in ['ren', 'lap', 'ren1', 'lap1', 'alive', 'one']:
        ip[c + '_o'] = ip[c + '_k'] - ip[c]
    pp = ip.groupby('pid')[['ren_o', 'lap_o', 'ren1_o', 'lap1_o', 'alive_o', 'one_o']].sum()
    pp['inv_rate'] = (pp.lap_o + 5 * g) / (pp.lap_o + pp.ren_o + 5)
    pp['inv_rate1y'] = (pp.lap1_o + 5 * g) / (pp.lap1_o + pp.ren1_o + 5)
    pp['inv_alive'] = pp.alive_o / pp.one_o.replace(0, np.nan)
    pp['inv_n'] = pp.one_o
    out = out.join(pp[['inv_rate', 'inv_rate1y', 'inv_alive', 'inv_n']], on='pid')
    # 출원인 최근 6개월 포기 수(정리 물결)
    L6 = Ld - np.timedelta64(182, 'D')
    lp6 = lap[(lap.cd > L6) & (lap.cd <= Ld)]
    a6 = pd.Series(pt.acd.values[lp6.pid.values]).value_counts()
    out['a_lap6m'] = a6.reindex(pt.acd.values[pids]).fillna(0).values
    out.loc[pt.acd.values[pids] == 'NA', 'a_lap6m'] = np.nan
    out.to_parquet(fn, index=False)
    return out


SOC = ['sib_rate', 'sib_rate1y', 'sib_n', 'sib_alive', 'aks_rate', 'aks_rate1y', 'aks_n', 'aks_alive', 'aipc_rate', 'aipc_rate1y',
       'aipc_n', 'aipc_alive', 'inv_rate', 'inv_rate1y', 'inv_alive', 'inv_n', 'a_lap6m']


def add_soc(d):
    parts = []
    for L in d.L.unique():
        s = soc_features(L).assign(L=L)
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    return d.merge(s, on=['pid', 'L'], how='left', validate='many_to_one') if len(d) else d


_AK = {}
KB = [0, 4, 6, 9, 12, 99]


def ak_table(L):
    """출원인 × 연차대 불납률(L 이전 결론 난 결정만). 연차대: 4 / 5-6 / 7-9 / 10-12 / 13+"""
    if L in _AK: return _AK[L]
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    Ld = np.datetime64(L)
    p = pay[(pay.seq > 0) & (pay.pd <= Ld)]
    l = lap[lap.cd <= Ld]
    r = pd.DataFrame({'acd': np.r_[pt.acd.values[p.pid.values], pt.acd.values[l.pid.values]],
                      'k': np.r_[p.s.values, l.lmax.values.astype(int) + 1],
                      'y': np.r_[np.zeros(len(p)), np.ones(len(l))]})
    r['kb'] = pd.cut(r.k, KB, labels=False)
    g = r.groupby('kb').y.mean()
    t = r.groupby(['acd', 'kb']).y.agg(['sum', 'size'])
    prior = g.reindex(t.index.get_level_values(1)).values
    rate = (t['sum'] + 5 * prior) / (t['size'] + 5)
    _AK[L] = (rate, t['size'], g)
    return _AK[L]


def add_ak(d, kcol):
    d = d.copy()
    acd = d.acd.values if 'acd' in d else T.load()['pt'].acd.values[d.pid.values]
    out = np.full((len(d), 2), np.nan)
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        rate, n, g = ak_table(L)
        kb = pd.cut(pd.Series(d[kcol].values[idx]), KB, labels=False).values
        keys = list(zip(acd[idx], kb))
        out[idx, 0] = np.where(np.isnan(rate.reindex(keys).values), g.reindex(kb).values, rate.reindex(keys).values)
        out[idx, 1] = n.reindex(keys).fillna(0).values
    out[acd == 'NA', 0] = np.nan
    d['ak_rate'], d['ak_n'] = out[:, 0], out[:, 1]
    return d
