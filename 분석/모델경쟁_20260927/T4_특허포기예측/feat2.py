# -*- coding: utf-8 -*-
"""2단계(감사·추가개선) 피처 모듈. 모두 기준일 L 이전 정보만 사용한다.
 - fix_dspst      : [감사 수정] 최종처분 스냅샷 중 등록 '후' 사건(이의신청후 등록유지 / 이의신청에 의한 등록취소)을 등록결정(일반)으로 되돌림
 - add_fam_dated  : [감사 수정] 해외 패밀리 토큰을 추정 공개연도 <= L연도-1 인 것만 집계 (FAM 이름 그대로 대체)
 - add_macro      : 외부 경기지표(KOSIS 광공업 생산·출하·재고지수, KSIC 3자리, L-2개월까지 공표분)
 - add_agent      : 출원대리인(사무소) 단위 L 이전 불납률(자기 몫 제외)
 - add_owner      : L 시점 권리자(이름) 단위 L 이전 불납률 — 결정(납부·불납)을 결정 시점의 권리자에게 귀속
"""
import os, numpy as np, pandas as pd
import t4lib as T

HERE = T.HERE
CACHE = T.CACHE
POSTGRANT = {'이의신청후 등록유지': '등록결정(일반)', '등록취소(이의신청에의한)': '등록결정(일반)'}
FAM = ['fam_n', 'fam_cc', 'fam_us', 'fam_ep', 'fam_jp', 'fam_cn', 'fam_wo']
MACRO = ['mac_prod_yoy', 'mac_ship_yoy', 'mac_inv_yoy', 'mac_prod_rel', 'mac_prod_3y']
AGENT = ['ag_rate', 'ag_rate1y', 'ag_alive']          # 절대규모 ag_n 은 달력 드리프트라 모형에서 제외(캐시에는 저장)
OWNER = ['ow_rate', 'ow_rate1y', 'ow_alive', 'ow_is_new']   # ow_n 동일 사유 제외
AGENT_ALL = AGENT + ['ag_n']
OWNER_ALL = OWNER + ['ow_n']
_C = {}


def fix_dspst(d):
    if 'dspst' in d:
        d = d.copy()
        d['dspst'] = d.dspst.astype(str).replace(POSTGRANT)
    return d


# ------------------------------------------------------------------ 시점 보정 패밀리
def _fam_tok():
    if 'fam' not in _C:
        fn = os.path.join(CACHE, 's2_fam_pid.parquet')     # 메모리 절약용 압축 캐시(pid int32, cc category, yr int16)
        if not os.path.exists(fn):
            pt = T.load()['pt']
            tk = pd.read_parquet(os.path.join(CACHE, 's2_fam_tokens.parquet'), columns=['applNo', 'cc', 'yr'])
            m = pd.Series(pt.pid.values, index=pt.applNo.values)
            m = m[~m.index.duplicated()]
            tk['pid'] = tk.applNo.map(m)
            tk = tk[tk.pid.notna()]
            out = pd.DataFrame({'pid': tk.pid.astype('int32').values, 'cc': tk.cc.astype('category').values, 'yr': tk.yr.astype('int16').values})
            out.to_parquet(fn, index=False)
            del tk
        _C['fam'] = pd.read_parquet(fn)
    return _C['fam']


def fam_dated(L, pids, mode='L'):
    """mode='L': 추정 공개연도 <= L연도-1 / mode='rd': 추가로 <= 등록연도(특허별 고정, 등록 시점에 보이던 패밀리)"""
    tk = _fam_tok()
    cut = int(str(L)[:4]) - 1
    s = tk[(tk.yr <= cut) & tk.pid.isin(pids)]
    if mode == 'rd':
        ry = pd.DatetimeIndex(T.load()['pt'].rd.values).year.values
        s = s[s.yr.values <= ry[s.pid.values]]
    out = pd.DataFrame(index=pd.Index(pids, name='pid'))
    out['fam_n'] = s.groupby('pid').size()
    out['fam_cc'] = s[~s.cc.isin(['WO', 'KR'])].groupby('pid').cc.nunique()
    for c in ['US', 'EP', 'JP', 'CN', 'WO']:
        out['fam_' + c.lower()] = s[s.cc == c].groupby('pid').size().gt(0).astype(float)
    return out.fillna(0.0)


def add_fam_dated(d, mode='L'):
    d = d.drop(columns=[c for c in FAM if c in d], errors='ignore').copy()
    M = np.zeros((len(d), len(FAM)), dtype='float32')
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        f = fam_dated(L, np.unique(d.pid.values[idx]), mode)
        M[idx] = f.reindex(d.pid.values[idx])[FAM].values
    for i, c in enumerate(FAM):
        d[c] = M[:, i]
    return d


# ------------------------------------------------------------------ 외부 경기지표
MAC_CSV = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv'


def _macro():
    if 'mac' not in _C:
        m = pd.read_csv(MAC_CSV, encoding='utf-8-sig', dtype={'산업코드': str, '연월': str})
        m['값'] = pd.to_numeric(m['값'], errors='coerce')
        m['kind'] = m['항목명'].str[:2].map({'생산': 'prod', '출하': 'ship', '재고': 'inv'})
        m['ym'] = pd.PeriodIndex(m['연월'].str[:4] + '-' + m['연월'].str[4:6], freq='M')
        _C['mac'] = {k: g.pivot_table(index='ym', columns='산업코드', values='값') for k, g in m.groupby('kind')}
    return _C['mac']


def _code(ksic, cols):
    for c in ('C' + ksic[:3], 'C' + ksic[:2]):
        if c in cols:
            return c
    return 'C'


def macro_at(L):
    """L 시점에 알 수 있던 값: 공표지연을 고려해 L-2개월까지. 12개월 평균 대비 직전 12개월 평균 증감."""
    mac = _macro()
    last = pd.Period(str(L)[:7], freq='M') - 2
    out = {}
    for k, P in mac.items():
        P = P[P.index <= last]
        a = P.iloc[-12:].mean() if len(P) >= 12 else P.iloc[:0].mean().reindex(P.columns)
        b = P.iloc[-24:-12].mean() if len(P) >= 24 else a * np.nan
        out[k + '_yoy'] = a / b - 1
        if k == 'prod':
            c3 = P.iloc[-48:-36].mean() if len(P) >= 48 else a * np.nan
            out['prod_3y'] = a / c3 - 1
    return pd.DataFrame(out)


def add_macro(d):
    d = d.copy()
    M = np.full((len(d), len(MACRO)), np.nan, dtype='float32')
    cols = _macro()['prod'].columns
    for L in d.L.unique():
        idx = np.where(d.L.values == L)[0]
        t = macro_at(L)
        ks = d.ksic.astype(str).values[idx]
        cmap = {k: _code(k, cols) for k in np.unique(ks)}
        codes = np.array([cmap[k] for k in ks])
        r = t.reindex(codes)
        tot = t.loc['C'].prod_yoy if 'C' in t.index else np.nan
        M[idx, 0] = r.prod_yoy.values
        M[idx, 1] = r.ship_yoy.values
        M[idx, 2] = r.inv_yoy.values
        M[idx, 3] = r.prod_yoy.values - tot
        M[idx, 4] = r.prod_3y.values
    for i, c in enumerate(MACRO):
        d[c] = M[:, i]
    return d


# ------------------------------------------------------------------ 결정 이력 공통
def _decisions(L):
    """L 이전(<=L)에 결론 난 결정: 연차료 납부(ren)·불납소멸(lap), 최근 1년분. pid별 개수"""
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    n = len(pt); Ld = np.datetime64(L); L1 = T.add_years(np.array([Ld]), -1)[0]
    pr = pay[(pay.seq > 0) & (pay.pd <= Ld)]
    lp = lap[lap.cd <= Ld]
    return dict(ren=np.bincount(pr.pid.values, minlength=n).astype(float),
                ren1=np.bincount(pr[pr.pd > L1].pid.values, minlength=n).astype(float),
                lap=np.bincount(lp.pid.values, minlength=n).astype(float),
                lap1=np.bincount(lp[lp.cd > L1].pid.values, minlength=n).astype(float), pr=pr, lp=lp)


def _grp_rate(key, dec, alive, valid, prior=5):
    """key: 길이 n 정수 그룹키(-1=없음). 그룹 합에서 자기 몫을 뺀 평활 불납률"""
    n = len(key)
    ren, lap, ren1, lap1 = dec['ren'], dec['lap'], dec['ren1'], dec['lap1']
    g = lap.sum() / max(lap.sum() + ren.sum(), 1)
    k = np.where(valid & (key >= 0), key, -1)
    ok = k >= 0
    K = int(k.max()) + 1 if ok.any() else 1
    S = {}
    one = np.ones(n)
    for nm, v in [('ren', ren), ('lap', lap), ('ren1', ren1), ('lap1', lap1), ('alive', alive), ('one', one)]:
        S[nm] = np.bincount(k[ok], weights=v[ok], minlength=K)

    def oth(nm, v):
        o = np.full(n, np.nan)
        o[ok] = S[nm][k[ok]] - v[ok]
        return o
    R, Lp, R1, L1 = oth('ren', ren), oth('lap', lap), oth('ren1', ren1), oth('lap1', lap1)
    no = oth('one', one); al = oth('alive', alive)
    rate = (Lp + prior * g) / (Lp + R + prior)
    rate1 = (L1 + prior * g) / (L1 + R1 + prior)
    alr = al / np.where(no > 0, no, np.nan)
    return rate, rate1, no, alr


# ------------------------------------------------------------------ 대리인
def _agent_key():
    if 'ag' not in _C:
        pt = T.load()['pt']
        a = pd.read_parquet(os.path.join(CACHE, 'agent.parquet')).drop_duplicates('rgst_no').set_index('rgst_no').agent1
        nm = a.reindex(pt.rgst_no.values).fillna('').str.replace(r'\s|\(유\)|유한회사|주식회사', '', regex=True).values
        codes, _ = pd.factorize(nm)
        _C['ag'] = np.where(nm == '', -1, codes).astype('int64')
    return _C['ag']


def agent_feats(L):
    fn = os.path.join(CACHE, f's2ag_{str(L).replace("-", "")}.parquet')
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    T.guard_L(L)
    D = T.load(); pt = D['pt']; n = len(pt)
    dec = _decisions(L)
    pids, _, _ = T.risk_set(L)
    alive = np.zeros(n); alive[pids] = 1
    reg = pt.rd.values <= np.datetime64(L)
    r, r1, no, al = _grp_rate(_agent_key(), dec, alive, reg)
    out = pd.DataFrame({'pid': pids, 'ag_rate': r[pids], 'ag_rate1y': r1[pids], 'ag_n': no[pids], 'ag_alive': al[pids]})
    out.to_parquet(fn, index=False)
    return out


# ------------------------------------------------------------------ L 시점 권리자
def _norm(s):
    return s.fillna('').str.replace(r'\s|주식회사|\(주\)|㈜|유한회사|\(유\)', '', regex=True)


def _owner_epochs():
    """(pid, od) 권리자 시기표: 같은 날 여러 행이면 seq 최소. 이름 코드"""
    if 'ow' not in _C:
        pt = T.load()['pt']
        o = pd.read_parquet(os.path.join(CACHE, 'owner.parquet'))
        pid = pd.Series(pt.pid.values, index=pt.rgst_no.values)
        o['pid'] = o.rgst_no.map(pid)
        o = o[o.pid.notna()].copy(); o['pid'] = o.pid.astype('int32')
        o['od'] = T.d64(o.ownerRgstCsDate)
        o = o[~pd.isna(o.od)]
        o['nm'] = _norm(o.ownerName)
        o = o.sort_values(['pid', 'od', 'seq']).drop_duplicates(['pid', 'od'])
        codes, _ = pd.factorize(o.nm.values)
        o['oc'] = np.where(o.nm.values == '', -1, codes).astype('int64')
        _C['ow'] = o[['pid', 'od', 'oc', 'seq']].reset_index(drop=True)
    return _C['ow']


def _owner_at(pids, dates, ep):
    """각 (pid, date)의 권리자 코드: od <= date 인 마지막 시기"""
    q = pd.DataFrame({'pid': np.asarray(pids).astype('int32'), 'dt': np.asarray(dates).astype('datetime64[ns]'),
                      'i': np.arange(len(pids))}).sort_values('dt')
    e = pd.DataFrame({'pid': ep.pid.values, 'dt': ep.od.values.astype('datetime64[ns]'), 'oc': ep.oc.values, 'sq': ep.seq.values}).sort_values('dt')
    m = pd.merge_asof(q, e, on='dt', by='pid', direction='backward')
    out = np.full(len(pids), -1, dtype='int64'); sq = np.zeros(len(pids))
    out[m.i.values] = m.oc.fillna(-1).astype('int64').values
    sq[m.i.values] = m.sq.fillna(0).values
    return out, sq


def owner_feats(L):
    fn = os.path.join(CACHE, f's2ow_{str(L).replace("-", "")}.parquet')
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    T.guard_L(L)
    D = T.load(); pt = D['pt']; n = len(pt); Ld = np.datetime64(L)
    ep = _owner_epochs()
    ep = ep[ep.od <= Ld]
    dec = _decisions(L)
    pr, lp = dec['pr'], dec['lp']
    oc_p, _ = _owner_at(pr.pid.values, pr.pd.values, ep)      # 결정을 그 시점 권리자에게 귀속
    oc_l, _ = _owner_at(lp.pid.values, lp.cd.values, ep)
    pids, _, _ = T.risk_set(L)
    reg = pt.rd.values <= Ld
    ocL, sqL = _owner_at(np.arange(n), np.full(n, Ld), ep)    # L 시점 권리자
    ocL = np.where(reg, ocL, -1)
    K = int(max(ocL.max(), oc_p.max() if len(oc_p) else 0, oc_l.max() if len(oc_l) else 0)) + 1
    okp, okl = oc_p >= 0, oc_l >= 0
    L1 = T.add_years(np.array([Ld]), -1)[0]
    r1m = pr.pd.values > L1; l1m = lp.cd.values > L1
    Sren = np.bincount(oc_p[okp], minlength=K).astype(float)
    Slap = np.bincount(oc_l[okl], minlength=K).astype(float)
    Sren1 = np.bincount(oc_p[okp & r1m], minlength=K).astype(float)
    Slap1 = np.bincount(oc_l[okl & l1m], minlength=K).astype(float)

    def own(oc, pid_arr, mask):
        return pd.Series(1.0, index=pd.MultiIndex.from_arrays([pid_arr[mask], oc[mask]])).groupby(level=[0, 1]).sum()
    sr = own(oc_p, pr.pid.values, okp); sl = own(oc_l, lp.pid.values, okl)
    sr1 = own(oc_p, pr.pid.values, okp & r1m); sl1 = own(oc_l, lp.pid.values, okl & l1m)
    c = ocL[pids]; okc = c >= 0; cc = np.where(okc, c, 0)
    key = pd.MultiIndex.from_arrays([pids, c])

    def g(s):
        return s.reindex(key).fillna(0).values
    R = Sren[cc] - g(sr); Lp = Slap[cc] - g(sl); R1 = Sren1[cc] - g(sr1); Lq = Slap1[cc] - g(sl1)
    gl = Slap.sum() / max(Slap.sum() + Sren.sum(), 1)
    Sal = np.bincount(c[okc], minlength=K).astype(float)
    regc = ocL[reg & (ocL >= 0)]
    Sreg = np.bincount(regc, minlength=K).astype(float)
    out = pd.DataFrame({'pid': pids})
    out['ow_rate'] = np.where(okc, (Lp + 5 * gl) / (Lp + R + 5), np.nan)
    out['ow_rate1y'] = np.where(okc, (Lq + 5 * gl) / (Lq + R1 + 5), np.nan)
    out['ow_n'] = np.where(okc, Sreg[cc] - 1, np.nan)
    out['ow_alive'] = np.where(okc & (Sreg[cc] > 1), (Sal[cc] - 1) / np.maximum(Sreg[cc] - 1, 1), np.nan)
    out['ow_is_new'] = (sqL[pids] > 0).astype(float)      # L 시점 권리자가 원권리자가 아님
    out.to_parquet(fn, index=False)
    return out


def _merge_perL(d, fn, cols):
    parts = [fn(L).assign(L=L) for L in d.L.unique()]
    s = pd.concat(parts, ignore_index=True)
    d = d.drop(columns=[c for c in cols if c in d], errors='ignore')
    return d.merge(s[['pid', 'L'] + cols], on=['pid', 'L'], how='left', validate='many_to_one')


def add_agent(d):
    return _merge_perL(d, agent_feats, AGENT_ALL)


def add_owner(d):
    return _merge_perL(d, owner_feats, OWNER_ALL)
