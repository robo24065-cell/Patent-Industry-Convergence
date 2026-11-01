# -*- coding: utf-8 -*-
"""T4 3라운드 새 피처. 모두 기준일 L 시점에 알 수 있던 값만 쓴다(원점 스냅샷·사후 재기준 금지).

접두사 = 그룹
 sep_ ETSI 표준필수특허(SEP) 선언 (`_shared/ext/trade_etsi_sep_link.parquet`, `trade_etsi_sep_patents.parquet`)
      공지일 kd = DB 반영일(first_refl). 단 2011년 이전 선언은 DB 이관 때 반영일이 2011-03 로 일괄 기록되어 있어
      kd = 선언 서명일(first_decl) + 90일로 둔다. kd <= L 인 선언만 사용. 선언 표준 수(n_std)는 사후 누적값이라 쓰지 않는다.
      sep_self: 그 KR 특허 자체가 선언됨 / sep_fam: 해외 패밀리(추정 공개연도 <= L연도-1 토큰)가 선언됨 / sep_a_share: 출원인 등록특허 중 SEP 비중
 np_  국민연금 가입 사업장 연말 스냅샷(`firm_nps_firm_panel.parquet`, 출원인코드 연결 `firm_nps_applicant_link.parquet`)
      스냅샷 게시월 < L 인 가장 최근 것만(첫 스냅샷 2015-12 → L >= 2016-01 에서만 값, 그 전은 NaN = '관측 전').
      사업자번호가 없는 출원인은 NaN, 사업자번호는 있는데 스냅샷에 없으면 np_pres=0.
 kp_  KPSS 특허 경제가치 xi (`quality_kpss_family_link.parquet`): KR 특허의 US 패밀리 중 등록일 <= L 인 것의 xi. 피인용수(cites)는 사후값이라 안 씀
 pr_  ECOS 생산자·수출 물가(`demand_ksic_price_panel_1996_2021.parquet`, KSIC 수작업 연결). 연평균은 이듬해 1월 공표 → L 이 1월이면 t=L연도-2, 7월이면 t=L연도-1
 ap_  (내부) 출원인 납부 행동: 최근 1년 연차료 납부 중 기한 후 납부 비중·다년 선납 비중·그 변화, 포기 가속도 — 권리자 단위 정리·자금 압박 신호
"""
import os, re, numpy as np, pandas as pd
import r3lib
R = r3lib.R
T = R.T
EXT = os.path.join(os.path.dirname(R.T4), '_shared', 'ext')
_C = {}


def _Ld(L):
    return np.datetime64(str(L)[:10])


def _Lym(L):
    s = str(L)
    return int(s[:4]), int(s[5:7])


def _per_L(d, cols, fn):
    """fn(L, idx) -> dict col->array(len idx)"""
    out = {c: np.full(len(d), np.nan, dtype='float32') for c in cols}
    Ls = d.L.astype(str).values
    for L in pd.unique(Ls):
        idx = np.where(Ls == L)[0]
        r = fn(L, idx)
        for c in cols:
            out[c][idx] = r[c]
    for c in cols:
        d[c] = out[c]
    return d


# ================================================================ sep_: ETSI SEP
def _kd(decl, refl):
    decl = pd.to_datetime(decl); refl = pd.to_datetime(refl)
    kd = np.where(decl >= pd.Timestamp('2011-01-01'), refl, decl + pd.Timedelta(days=90))
    return pd.to_datetime(kd).values.astype('datetime64[D]')


def _sep_tables():
    if 'sep' in _C:
        return _C['sep']
    cdir = os.path.join(r3lib.R3, 'cache'); os.makedirs(cdir, exist_ok=True)
    f1, f2 = os.path.join(cdir, 'sep_self_kd.npy'), os.path.join(cdir, 'sep_fam.parquet')
    if os.path.exists(f1) and os.path.exists(f2):            # 한 번 만든 연결표 재사용(메모리 절약)
        _C['sep'] = dict(self_kd=np.load(f1), fam=pd.read_parquet(f2))
        return _C['sep']
    pt = T.load()['pt']
    n = len(pt)
    # (1) KR 특허 자체 선언: 로컬 매칭 결과(trade판) — 출원번호 13자리
    s = pd.read_parquet(os.path.join(EXT, 'trade_etsi_sep_link.parquet'), columns=['cc', 'appl_no', 'first_decl', 'first_refl'])
    s = s[s.cc == 'KR'].copy()
    s['kd'] = _kd(s.first_decl, s.first_refl)
    s = s.groupby('appl_no').kd.min()
    m = pd.Series(pt.pid.values, index=pt.applNo.values)
    m = m[~m.index.duplicated()]
    self_kd = np.full(n, np.datetime64('NaT'), dtype='datetime64[D]')
    hit = s.index.intersection(m.index)
    self_kd[m.reindex(hit).values] = s.reindex(hit).values
    # (2) 해외 패밀리 선언: ETSI 공개번호 → 정규화 키(cc + 숫자, 앞 0 제거)
    e = pd.read_parquet(os.path.join(EXT, 'trade_etsi_sep_patents.parquet'), columns=['cc', 'PUBL_NUMBER', 'first_decl', 'first_refl'])
    e = e[e.cc.isin(['US', 'EP', 'JP', 'CN', 'WO', 'KR']) & e.PUBL_NUMBER.notna()].copy()
    e['kd'] = _kd(e.first_decl, e.first_refl)
    rows = []
    for cc, pub, kd in zip(e.cc.values, e.PUBL_NUMBER.values, e.kd.values):
        for p in str(pub).split('|'):
            p = re.sub(r'\s+[A-Z]\d?$', '', p.strip()).replace(' ', '')
            p = re.sub(r'[A-Z]\d?$', '', p) if re.search(r'\d[A-Z]\d?$', p) else p
            if p.startswith(cc): p = p[2:]
            dg = re.sub(r'[^0-9]', '', p)
            if len(dg) >= 6:
                rows.append((cc + str(int(dg)), kd))
    K = pd.DataFrame(rows, columns=['key', 'kd']).groupby('key').kd.min()
    tk = pd.read_parquet(os.path.join(R.T4, 'cache', 's2_fam_tokens.parquet'), columns=['applNo', 'cc', 'yr', 'tok'])
    tk = tk[tk.cc.isin(['US', 'EP', 'JP', 'CN', 'WO'])]
    dg = tk.tok.str[2:].str.replace(r'[^0-9]', '', regex=True)
    ok = dg.str.len() >= 6
    tk = tk[ok]; dg = dg[ok]
    tk = tk.assign(key=tk.cc.values + dg.astype('int64').astype(str).values)
    tk = tk[tk.key.isin(K.index)]
    tk = tk.assign(pid=tk.applNo.map(m), kd=K.reindex(tk.key.values).values)
    tk = tk[tk.pid.notna()]
    fam = pd.DataFrame({'pid': tk.pid.astype('int64').values, 'yr': tk.yr.values.astype('int16'), 'kd': tk.kd.values})
    np.save(f1, self_kd); fam.to_parquet(f2, index=False)
    _C['sep'] = dict(self_kd=self_kd, fam=fam)
    print('sep 표: 자체 선언 연결', int((~np.isnat(self_kd)).sum()), '패밀리 선언 연결 특허', fam.pid.nunique(), flush=True)
    return _C['sep']


def _sep_at(L):
    k = ('sepL', L)
    if k in _C:
        return _C[k]
    S = _sep_tables(); pt = T.load()['pt']; n = len(pt)
    Ld = _Ld(L); cut = int(str(L)[:4]) - 1
    sf = (~np.isnat(S['self_kd'])) & (S['self_kd'] <= Ld)
    f = S['fam']; f = f[(f.yr <= cut) & (f.kd <= Ld)]
    fa = np.zeros(n, bool); fa[f.pid.unique()] = True
    anyv = sf | fa
    reg = pt.rd.values <= Ld
    acd = pt.acd.values
    g = pd.DataFrame({'a': acd[reg], 's': anyv[reg].astype(float)}).groupby('a').s.agg(['sum', 'size'])
    share = ((g['sum'] + 0.0) / (g['size'] + 1.0))
    ash = share.reindex(acd).values.astype('float32')
    ash[acd == 'NA'] = np.nan
    _C[k] = (sf.astype('float32'), fa.astype('float32'), ash)
    return _C[k]


def sep_feats(d, cols):
    def fn(L, idx):
        sf, fa, ash = _sep_at(L)
        p = d.pid.values[idx]
        return {'sep_self': sf[p], 'sep_fam': fa[p], 'sep_any': np.maximum(sf[p], fa[p]), 'sep_a_share': ash[p]}
    return _per_L(d, cols, fn)


# ================================================================ np_: 국민연금 사업장
SNAPS = ['2015-12', '2016-12', '2017-12', '2018-12', '2019-12', '2020-12', '2021-12']   # 🔒 개발은 L<=2018-07 이라 2018-12 이후는 선택되지 않음


def _nps():
    if 'nps' not in _C:
        lk = pd.read_parquet(os.path.join(EXT, 'firm_nps_applicant_link.parquet'), columns=['applicantCd', 'bizno'])
        fp = pd.read_parquet(os.path.join(EXT, 'firm_nps_firm_panel.parquet'), columns=['applicantCd', 'snap', 'n_emp', 'n_loss', 'active'])
        fp = fp[fp.snap.isin(SNAPS)]
        _C['nps'] = (set(lk.applicantCd.values), {s: g.set_index('applicantCd') for s, g in fp.groupby('snap')})
    return _C['nps']


def _snap_for(L):
    y, m = _Lym(L)
    ok = [s for s in SNAPS if (int(s[:4]), int(s[5:7])) < (y, m)]     # 게시월 < L 월
    return ok[-1] if ok else None


def np_feats(d, cols):
    linked, P = _nps()
    acd_all = d.acd.astype(str).values

    def fn(L, idx):
        a = acd_all[idx]
        out = {c: np.full(len(idx), np.nan, dtype='float32') for c in cols}
        s = _snap_for(L)
        if s is None:
            return out
        lk = np.array([x in linked for x in a])
        cur = P[s]
        inn = pd.Index(a).isin(cur.index)
        emp = cur.n_emp.reindex(a).values.astype(float)
        act = cur.active.reindex(a).values
        loss = cur.n_loss.reindex(a).values.astype(float)
        out['np_pres'] = np.where(lk, inn.astype(float), np.nan)
        out['np_act'] = np.where(lk, np.where(inn, (act == True).astype(float), 0.0), np.nan)  # noqa: E712
        out['np_lemp'] = np.where(lk, np.log1p(np.nan_to_num(emp, nan=0.0)), np.nan)
        out['np_lossr'] = np.where(lk & inn, loss / np.maximum(np.nan_to_num(emp, nan=0.0), 1.0), np.nan)
        i = SNAPS.index(s)
        if i > 0:
            pv = P[SNAPS[i - 1]]
            inp = pd.Index(a).isin(pv.index)
            e0 = pv.n_emp.reindex(a).values.astype(float)
            out['np_gemp'] = np.where(lk & inp & inn, np.log1p(np.nan_to_num(emp, nan=0.0)) - np.log1p(np.nan_to_num(e0, nan=0.0)), np.nan)
            out['np_exit'] = np.where(lk, (inp & (~inn | (act != True))).astype(float), np.nan)  # noqa: E712
        return out
    return _per_L(d, cols, fn)


# ================================================================ kp_: KPSS 특허 가치
def _kpss():
    if 'kp' not in _C:
        pt = T.load()['pt']
        k = pd.read_parquet(os.path.join(EXT, 'quality_kpss_family_link.parquet'), columns=['cc', 'appl_no', 'us_pn', 'issue_date', 'xi_real'])
        k = k[k.cc == 'KR'].drop_duplicates(['appl_no', 'us_pn'])
        m = pd.Series(pt.pid.values, index=pt.applNo.values)
        m = m[~m.index.duplicated()]
        k = k.assign(pid=k.appl_no.map(m), isd=pd.to_datetime(k.issue_date.astype(str), errors='coerce').values.astype('datetime64[D]'))
        k = k[k.pid.notna() & k.isd.notna()]
        _C['kp'] = pd.DataFrame({'pid': k.pid.astype('int64').values, 'isd': k.isd.values, 'xi': k.xi_real.values.astype(float)})
    return _C['kp']


def kp_feats(d, cols):
    K = _kpss(); n = len(T.load()['pt'])

    def fn(L, idx):
        s = K[K.isd <= _Ld(L)]
        cnt = np.bincount(s.pid.values, minlength=n).astype('float32')
        mx = np.zeros(n, 'float32')
        g = s.groupby('pid').xi.max()
        mx[g.index.values] = np.log1p(g.values)
        p = d.pid.values[idx]
        return {'kp_n': np.log1p(cnt[p]), 'kp_lxi': mx[p], 'kp_has': (cnt[p] > 0).astype('float32')}
    return _per_L(d, cols, fn)


# ================================================================ pr_: ECOS 물가
def _price():
    if 'pr' not in _C:
        e = pd.read_parquet(os.path.join(EXT, 'demand_ksic_price_panel_1996_2021.parquet'))
        _C['pr'] = {s: g.pivot_table(index='ksic', columns='year', values='lidx') for s, g in e.groupby('series')}
    return _C['pr']


def price_at(L):
    P = _price(); y, m = _Lym(L)
    t = y - 1 if m >= 7 else y - 2
    out = {}
    for nm, ser in [('ppi', 'ppi_basic'), ('exp', 'exp_item')]:
        M = P[ser]
        for h in (1, 3):
            if t in M.columns and (t - h) in M.columns:
                g = M[t] - M[t - h]
                out[f'pr_{nm}_g{h}'] = g - g.mean()
            else:
                out[f'pr_{nm}_g{h}'] = pd.Series(np.nan, index=M.index)
    return pd.DataFrame(out)


def pr_feats(d, cols):
    ks = d.ksic.astype(str).values

    def fn(L, idx):
        tab = price_at(L)
        return {c: tab[c].reindex(ks[idx]).values for c in cols}
    return _per_L(d, cols, fn)


# ================================================================ ap_: 출원인 납부 행동(내부)
def _ap_at(L):
    k = ('ap', L)
    if k in _C:
        return _C[k]
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    Ld = _Ld(L)
    L1 = T.add_years(np.array([Ld]), -1)[0]; L3 = T.add_years(np.array([Ld]), -3)[0]
    acd = pt.acd.values
    p = pay[(pay.seq > 0) & (pay.pd <= Ld) & (pay.pd > L3)]
    a = acd[p.pid.values]
    rec = (p.pd.values > L1)
    late = (p.early.values < 0).astype(float)
    multi = ((p.l.values - p.s.values) >= 1).astype(float)
    df = pd.DataFrame({'a': a, 'rec': rec, 'late': late, 'multi': multi})
    g1 = df[df.rec].groupby('a').agg(n1=('late', 'size'), late1=('late', 'sum'), multi1=('multi', 'sum'))
    g0 = df[~df.rec].groupby('a').agg(n0=('late', 'size'), late0=('late', 'sum'))
    gl = late.mean() if len(late) else 0.1
    gm = multi[rec].mean() if rec.any() else 0.1
    g = g1.join(g0, how='outer').fillna(0)
    g['ap_late1y'] = (g.late1 + 5 * gl) / (g.n1 + 5)
    g['ap_late_d'] = g.ap_late1y - (g.late0 + 5 * gl) / (g.n0 + 5)
    g['ap_multi1y'] = (g.multi1 + 5 * gm) / (g.n1 + 5)
    g['ap_n1y'] = np.log1p(g.n1)
    l = lap[(lap.cd <= Ld) & (lap.cd > L3)]
    la = pd.DataFrame({'a': acd[l.pid.values], 'rec': l.cd.values > L1}).groupby('a').rec.agg(['sum', 'size'])
    acc = np.log((la['sum'] + 1) / ((la['size'] - la['sum']) / 2 + 1))
    g = g.join(acc.rename('ap_lapacc'), how='outer')
    g['ap_lapacc'] = g.ap_lapacc.fillna(0.0)
    g = g.drop(index='NA', errors='ignore')
    _C[k] = g[['ap_late1y', 'ap_late_d', 'ap_multi1y', 'ap_n1y', 'ap_lapacc']]
    return _C[k]


def ap_feats(d, cols):
    acd_all = d.acd.astype(str).values

    def fn(L, idx):
        g = _ap_at(L)
        return {c: g[c].reindex(acd_all[idx]).values for c in cols}
    return _per_L(d, cols, fn)


# ================================================================ 진입점
NEW = {'sep': sep_feats, 'np': np_feats, 'kp': kp_feats, 'pr': pr_feats, 'ap': ap_feats}


def add(d, cols):
    d = d.copy()
    groups = {}
    for c in cols:
        groups.setdefault(c.split('_')[0], []).append(c)
    old = []
    for g, cs in groups.items():
        if g in NEW:
            d = NEW[g](d, cs)
        else:
            old += cs
    if old:
        import r2ext
        d = r2ext.add(d, old)
    return d


# ================================================================ oa_: OpenAlex 기업 소속 저자 논문(KSIC 키워드) — '기술 퇴조' 과학 신호
def _oa():
    if 'oa' not in _C:
        e = pd.read_parquet(os.path.join(EXT, 'science_openalex_ksic_extra_year.parquet'))
        _C['oa'] = {v: g.pivot_table(index='ksic', columns='year', values='n_works', aggfunc='sum') for v, g in e.groupby('variant')}
    return _C['oa']


def oa_at(L):
    """출판연도 <= L연도-1 만(준-시점정합: 현재 스냅샷). 최근 2년 / 그 전 2년 로그비를 전체(TOTAL) 대비로"""
    P = _oa(); y, _ = _Lym(L); t = y - 1
    out = {}
    for nm, v in [('all', 'company_ALL'), ('kr', 'company_KR')]:
        M = P[v]
        a = M[[c for c in M.columns if t - 1 <= c <= t]].sum(1); b = M[[c for c in M.columns if t - 3 <= c <= t - 2]].sum(1)
        g = np.log((a + 1) / (b + 1))
        tot = g.get('TOTAL', g.mean())
        out[f'oa_{nm}_rg'] = (g - tot).drop(index='TOTAL', errors='ignore')
        if nm == 'all':
            sh = a / a.get('TOTAL', a.sum())
            out['oa_all_lsh'] = np.log(sh.drop(index='TOTAL', errors='ignore') + 1e-6)
    return pd.DataFrame(out)


def oa_feats(d, cols):
    ks = d.ksic.astype(str).values

    def fn(L, idx):
        tab = oa_at(L)
        return {c: tab[c].reindex(ks[idx]).values for c in cols}
    return _per_L(d, cols, fn)


NEW['oa'] = oa_feats


# ================================================================ (재시도) rs_: Reliance on Science — KR 특허의 US 패밀리 등록특허가 과학 논문을 인용하는가
# 연결: T4 s2_fam_tokens(해외 패밀리 토큰, 추정 공개연도 yr) 의 US 등록번호(7~8자리) → r3ros_patent_sci(us_pn, issue_date, n_cit).
# 시점: 토큰 yr <= L연도-1 (fam_dated 규칙과 동일) AND US 발행일 < L. US 특허의 논문 인용은 발행 때 확정되고 사후에 바뀌지 않는다.
def _rs():
    if 'rs' not in _C:
        pt = T.load()['pt']
        tk = pd.read_parquet(os.path.join(R.T4, 'cache', 's2_fam_tokens.parquet'), columns=['applNo', 'cc', 'yr', 'tok'])
        tk = tk[tk.cc == 'US']
        dg = tk.tok.str[2:].str.replace(r'[^0-9]', '', regex=True)
        ok = (dg.str.len() >= 6) & (dg.str.len() <= 8)
        tk = tk[ok].assign(us_pn=dg[ok].astype('int64').values)
        s = pd.read_parquet(os.path.join(EXT, 'r3ros_patent_sci.parquet'), columns=['us_pn', 'issue_date', 'n_cit'])
        s = s.assign(us_pn=s.us_pn.astype('int64'))
        tk = tk.merge(s, on='us_pn', how='inner')
        m = pd.Series(pt.pid.values, index=pt.applNo.values)
        m = m[~m.index.duplicated()]
        tk = tk.assign(pid=tk.applNo.map(m))
        tk = tk[tk.pid.notna()].drop_duplicates(['pid', 'us_pn'])
        _C['rs'] = pd.DataFrame({'pid': tk.pid.astype('int64').values, 'yr': tk.yr.values.astype('int16'),
                                 'isd': pd.to_datetime(tk.issue_date.astype(str)).values.astype('datetime64[D]'),
                                 'ncit': tk.n_cit.values.astype(float)})
        print('rs 연결: 특허', _C['rs'].pid.nunique(), 'US 등록', len(_C['rs']), flush=True)
    return _C['rs']


def rs_feats(d, cols):
    S = _rs(); n = len(T.load()['pt'])

    def fn(L, idx):
        s = S[(S.yr <= int(str(L)[:4]) - 1) & (S.isd < _Ld(L))]
        mx = np.zeros(n, 'float32'); g = s.groupby('pid').ncit.max(); mx[g.index.values] = g.values
        has = np.zeros(n, 'float32'); has[g.index.values] = 1
        p = d.pid.values[idx]
        return {'rs_sci': (mx[p] > 0).astype('float32'), 'rs_lncit': np.log1p(mx[p]), 'rs_has': has[p]}
    return _per_L(d, cols, fn)


NEW['rs'] = rs_feats


# ================================================================ (재시도) nt_: 국세청 휴·폐업 — 1순위 출원인(출원인코드→사업자번호)의 폐업일
# 폐업일(end_dt) < L 이면 L 시점 폐업. 폐업은 되돌려지지 않아 사후 수정 없음. 단 2013년 이전 폐업이 거의 관측되지 않는 좌측 절단이 있어
# (수집 기록 A3) L < 2016-01-01 에서는 값을 NaN(관측 전)으로 둔다. 사업자번호 여러 개인 출원인은 모두 폐업일 때만 폐업(날짜는 마지막 폐업일).
NT_FIRST = np.datetime64('2016-01-01')


def _nts():
    if 'nt' not in _C:
        a = pd.read_parquet(os.path.join(EXT, 'r3nts_applicant_status.parquet'), columns=['applicantCd', 'b_stt_cd', 'end_dt', 'registered'])
        a = a[a.registered == True]  # noqa: E712
        a = a.assign(cl=(a.b_stt_cd.astype(str) == '03'))
        g = a.groupby('applicantCd').agg(allcl=('cl', 'all'), end=('end_dt', 'max'))
        g['end'] = np.where(g.allcl, g.end.values.astype('datetime64[D]'), np.datetime64('NaT'))
        _C['nt'] = g
    return _C['nt']


def nt_feats(d, cols):
    g = _nts(); acd_all = d.acd.astype(str).values

    def fn(L, idx):
        Ld = _Ld(L)
        out = {c: np.full(len(idx), np.nan, dtype='float32') for c in cols}
        if Ld < NT_FIRST:
            return out
        a = acd_all[idx]
        lk = pd.Index(a).isin(g.index)
        end = g['end'].reindex(a).values.astype('datetime64[D]')
        closed = (~np.isnat(end)) & (end < Ld)
        L1 = T.add_years(np.array([Ld]), -1)[0]
        cl1 = closed & (end >= L1)
        yrs = np.where(closed, (Ld - end).astype('timedelta64[D]').astype(float) / 365.25, np.nan)
        out['nt_closed'] = np.where(lk, closed.astype(float), np.nan)
        out['nt_cl1y'] = np.where(lk, cl1.astype(float), np.nan)
        out['nt_yrs'] = np.where(lk, yrs, np.nan)
        return out
    return _per_L(d, cols, fn)


NEW['nt'] = nt_feats


# ================================================================ (재시도) at_: (내부) 출원인 × 연차료 구간 최근 2년 불납률 — 권리자의 '구간별 정리 정책'
# 결정 = L-2년 < 시점 <= L 인 연차료 납부(pd, 연차 s) 또는 소멸등록(cd, 연차 lmax+1). 연차 구간: 1-3/4-6/7-9/10-12/13+ (KR 연차료 단계).
# 특허의 다음 연차(next_annl) 구간에서 같은 출원인이 최근 2년간 얼마나 포기했는지. 자기 납부분은 뺀다. 구간별 전체율로 평활(prior 5).
def _tier(j):
    j = np.asarray(j, dtype=float)
    return np.select([j <= 3, j <= 6, j <= 9, j <= 12], [0, 1, 2, 3], 4).astype('int8')


def _at_at(L):
    k = ('at', L)
    if k in _C:
        return _C[k]
    D = T.load(); pt, pay, lap = D['pt'], D['pay'], D['lap']
    Ld = _Ld(L); L2 = T.add_years(np.array([Ld]), -2)[0]
    acd = pt.acd.values
    p = pay[(pay.seq > 0) & (pay.pd <= Ld) & (pay.pd > L2)]
    l = lap[(lap.cd <= Ld) & (lap.cd > L2)]
    ev = pd.DataFrame({'pid': np.r_[p.pid.values, l.pid.values], 't': np.r_[_tier(p.s.values), _tier(l.lmax.values + 1)],
                       'lap': np.r_[np.zeros(len(p)), np.ones(len(l))]})
    ev['a'] = acd[ev.pid.values]
    gt = ev.groupby('t').lap.mean()                                   # 구간별 전체 불납률(prior)
    g_all = float(ev.lap.mean()) if len(ev) else 0.1
    at = ev.groupby(['a', 't']).lap.agg(['sum', 'size'])
    aa = ev.groupby('a').lap.agg(['sum', 'size'])
    own = ev[ev.lap == 0].groupby(['pid', 't']).size()               # 자기 납부(자기 몫 제외용)
    _C[k] = dict(gt=gt, g_all=g_all, at=at, aa=aa, own=own)
    return _C[k]


def at_feats(d, cols):
    acd_all = d.acd.astype(str).values

    def fn(L, idx):
        S = _at_at(L)
        a = acd_all[idx]; pid = d.pid.values[idx]
        t = _tier(np.nan_to_num(d.next_annl.values[idx].astype(float), nan=4.0))
        mi = pd.MultiIndex.from_arrays([a, t])
        s_ = S['at']['sum'].reindex(mi).fillna(0).values; n_ = S['at']['size'].reindex(mi).fillna(0).values
        own = S['own'].reindex(pd.MultiIndex.from_arrays([pid, t])).fillna(0).values
        n_ = n_ - own
        g = S['gt'].reindex(t).fillna(S['g_all']).values
        rate = (s_ + 5 * g) / (n_ + 5)
        sa = S['aa']['sum'].reindex(a).fillna(0).values; na = S['aa']['size'].reindex(a).fillna(0).values
        owna = S['own'].reindex(pd.MultiIndex.from_arrays([pid, t])).fillna(0).values   # 자기 납부는 모두 같은 t 로 근사
        rate_a = (sa + 5 * S['g_all']) / (na - owna + 5)
        na_mask = (a == 'NA')
        out = {'at_rate': np.where(na_mask, np.nan, rate), 'at_n': np.where(na_mask, np.nan, np.log1p(np.maximum(n_, 0))),
               'at_rel': np.where(na_mask, np.nan, rate - rate_a)}
        return out
    return _per_L(d, cols, fn)


NEW['at'] = at_feats


# ================================================================ (B단계) kz_: KOSIS 광업제조업 — KSIC 5자리 연간 품목편 + 3자리 생산능력·가동률 + 국내공급(국산/수입)
# 공표 시점: 품목편 자료연도 Y 는 Y+2년 1~2월 공표 → L 이 1월이면 Y <= L연도-3, 7월이면 Y <= L연도-2.
#           월별 생산능력·국내공급의 연평균 Y 는 12월분이 Y+1년 1월 말 공표 → L 이 1월이면 Y <= L연도-2, 7월이면 Y <= L연도-1.
# 수준값(2020=100 재기준 지수)은 쓰지 않는다. 성장률·금액 비율(재고/출하)만, 성장률은 KSIC 단면 평균을 뺀다(경기 공통 성분 제거).
def _kz():
    if 'kz' not in _C:
        it = pd.read_parquet(os.path.join(EXT, 'r3kosis_ksic50_items_chain.parquet'), columns=['ksic', 'year', 'g_prod', 'g_ship', 'inv_ship1', 'd_inv_ship', 'chain_prod_idx'])
        cu = pd.read_parquet(os.path.join(EXT, 'r3kosis_ksic50_capacity_util_annual.parquet'), columns=['ksic', 'year', 'g1_util', 'g3_cap'])
        ds = pd.read_parquet(os.path.join(EXT, 'r3kosis_ksic50_domsupply_annual.parquet'), columns=['ksic', 'year', 'g3_import_minus_dom'])
        cu = cu.dropna(subset=['year']).assign(year=lambda x: x.year.astype(int))
        _C['kz'] = dict(it=it.assign(ksic=it.ksic.astype(str)), cu=cu.assign(ksic=cu.ksic.astype(str)), ds=ds.assign(ksic=ds.ksic.astype(str)))
    return _C['kz']


def kz_at(L):
    k = ('kzL', L)
    if k in _C:
        return _C[k]
    K = _kz(); y, m = _Lym(L)
    yi = y - 3 if m < 7 else y - 2                  # 품목편
    ym = y - 2 if m < 7 else y - 1                  # 월별 연평균
    it = K['it']; a = it[it.year == yi].set_index('ksic')
    b = it[it.year == yi - 3].set_index('ksic')
    out = pd.DataFrame(index=sorted(set(it.ksic) | set(K['cu'].ksic) | set(K['ds'].ksic)))
    g = a.g_prod.reindex(out.index); out['kz_gprod'] = g - g.mean()
    g3 = np.log(a.chain_prod_idx.reindex(out.index) / b.chain_prod_idx.reindex(out.index)); out['kz_gprod3'] = g3 - g3.mean()
    out['kz_invship'] = a.inv_ship1.reindex(out.index)
    out['kz_dinvship'] = a.d_inv_ship.reindex(out.index)
    cu = K['cu']; c = cu[cu.year == ym].set_index('ksic')
    u = c.g1_util.reindex(out.index); out['kz_util_g1'] = u - u.mean()
    cp = c.g3_cap.reindex(out.index); out['kz_cap_g3'] = cp - cp.mean()
    ds = K['ds']; s_ = ds[ds.year == ym].set_index('ksic')
    out['kz_impdom3'] = s_.g3_import_minus_dom.reindex(out.index)
    _C[k] = out
    return out


def kz_feats(d, cols):
    ks = d.ksic.astype(str).values

    def fn(L, idx):
        tab = kz_at(L)
        return {c: tab[c].reindex(ks[idx]).values for c in cols}
    return _per_L(d, cols, fn)


NEW['kz'] = kz_feats


# ================================================================ (B단계) dm_: (내부) 출원인 휴면 — 국세청 폐업일로 검증한 '폐업 대리변수'
# 국세청 폐업일은 L>=2016 에서만 쓸 수 있다(좌측 절단). 폐업한 법인은 모든 특허의 연차료 납부와 신규 등록을 멈추므로,
# 같은 출원인의 '다른' 특허에서 마지막 납부·등록 이후 경과연수를 L 시점 정보(pd<=L, rd<=L)로 만들면 모든 원점에서 쓸 수 있다.
#  dm_opay_yrs : 출원인의 다른 특허(자기 제외) 마지막 납부(등록료 포함) 이후 경과연수
#  dm_oreg_yrs : 출원인의 다른 특허 마지막 등록 이후 경과연수
#  dm_opay_n1y : 최근 1년 다른 특허 납부 건수(log1p)
# 출원인코드 없음(NA)·다른 특허 없음은 NaN.
def _top2(key, val):
    """key 별 val 최대값과 두 번째 최대값(서로 다른 행)"""
    df = pd.DataFrame({'k': key, 'v': val}).sort_values(['k', 'v'], ascending=[True, False]).reset_index(drop=True)
    r = df.groupby('k').cumcount().values
    f = df[r == 0].set_index('k').v
    s_ = df[r == 1].set_index('k').v
    return f, s_


def _dm_at(L):
    k = ('dm', L)
    if k in _C:
        return _C[k]
    D = T.load(); pt, pay = D['pt'], D['pay']
    Ld = _Ld(L); L1 = T.add_years(np.array([Ld]), -1)[0]
    n = len(pt); acd = pt.acd.values.astype(str)
    p = pay[pay.pd <= Ld]
    lastpay = np.full(n, np.datetime64('NaT'), dtype='datetime64[D]')
    g = p.groupby('pid').pd.max()
    lastpay[g.index.values] = g.values.astype('datetime64[D]')
    n1 = np.bincount(p[p.pd > L1].pid.values, minlength=n).astype(float)
    reg = pt.rd.values.astype('datetime64[D]') <= Ld
    ids = np.where(reg)[0]
    a = acd[ids]
    lp = lastpay[ids].astype('datetime64[D]').astype('int64').astype(float); lp[np.isnat(lastpay[ids])] = -1e9
    rd = pt.rd.values.astype('datetime64[D]')[ids].astype('int64').astype(float)
    f1, s1 = _top2(a, lp); f2, s2 = _top2(a, rd)
    na = pd.Series(n1[ids]).groupby(a).sum()
    cnt = pd.Series(1, index=a).groupby(level=0).size()
    _C[k] = dict(f1=f1, s1=s1, f2=f2, s2=s2, na=na, cnt=cnt, lastpay=lastpay, n1=n1, Lnum=float(Ld.astype('int64')))
    return _C[k]


def dm_feats(d, cols):
    acd_all = d.acd.astype(str).values
    pt = T.load()['pt']
    rd_all = pt.rd.values.astype('datetime64[D]').astype('int64').astype(float)

    def fn(L, idx):
        S = _dm_at(L)
        a = acd_all[idx]; pid = d.pid.values[idx]
        own_lp = S['lastpay'][pid].astype('int64').astype(float); own_lp[np.isnat(S['lastpay'][pid])] = -1e9
        f1 = S['f1'].reindex(a).values; s1 = S['s1'].reindex(a).values
        f2 = S['f2'].reindex(a).values; s2 = S['s2'].reindex(a).values
        other_lp = np.where(own_lp >= f1, s1, f1)            # 자기가 최대면 두 번째(동률이면 두 번째도 같은 값)
        other_rd = np.where(rd_all[pid] >= f2, s2, f2)
        cnt = S['cnt'].reindex(a).fillna(0).values
        none = (a == 'NA') | (cnt < 2)
        other_lp = np.where(other_lp < -1e8, np.nan, other_lp)
        out = {'dm_opay_yrs': np.where(none, np.nan, (S['Lnum'] - other_lp) / 365.25),
               'dm_oreg_yrs': np.where(none, np.nan, (S['Lnum'] - other_rd) / 365.25),
               'dm_opay_n1y': np.where(none, np.nan, np.log1p(np.maximum(S['na'].reindex(a).fillna(0).values - S['n1'][pid], 0)))}
        return out
    return _per_L(d, cols, fn)


NEW['dm'] = dm_feats
