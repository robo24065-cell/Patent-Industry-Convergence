# -*- coding: utf-8 -*-
"""T4 특허 포기(등록료불납) 예측 — 공용 라이브러리
단위: (특허, 기준일 L). 위험집합: L 시점에 '납부로 살아 있는' KR 등록특허.
결과: L 이후 3년 안에 실효일 E(=등록일 + 마지막 납부연차 년)가 오는 등록료불납 소멸.

홀드아웃 보호 장치
 - 소멸 이벤트는 E < HOLDOUT_START(2019-01-01) 인 것만 메모리에 올린다(1·2단계 전용 가드).
 - label_full()은 L+3y <= HOLDOUT_START 가 아니면 예외.
 - 피처는 L 이전(<=L) 정보만 쓴다.
"""
import os, re, pickle, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
HOLDOUT_START = np.datetime64('2019-01-01')
H_YEARS = 3
ALLOW_HOLDOUT = os.environ.get('T4_STAGE3_HOLDOUT') == '1'   # 3단계 평가자만 켠다

_D = {}


def d64(s):
    s = pd.Series(s, dtype='object').astype(str).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce').values.astype('datetime64[D]')


def add_years(dates, k):
    """datetime64[D] 배열 + 정수 년(기념일 정확 계산, 2/29 -> 2/28)"""
    di = pd.DatetimeIndex(dates)
    k = np.asarray(k)
    y = di.year.values + k
    m = di.month.values
    d = di.day.values
    d = np.where((m == 2) & (d == 29), 28, d)
    ok = ~np.isnan(y.astype(float)) & ~pd.isna(di)
    out = np.full(len(di), np.datetime64('NaT'), dtype='datetime64[D]')
    if ok.any():
        s = pd.to_datetime(pd.DataFrame({'year': y[ok].astype(int), 'month': m[ok], 'day': d[ok]}), errors='coerce')
        out[ok] = s.values.astype('datetime64[D]')
    return out


def yrs(a, b):
    return (a - b).astype('timedelta64[D]').astype(float) / 365.25


CORP_RE = re.compile(r'주식회사|\(주\)|（주）|㈜|유한|회사|법인|재단|협회|조합|연구원|연구소|대학|학교|공사|센터|코포레이션|인코포레이티드|가부시키|리미티드|게엠베하|'
                     r'엘티디|엘엘씨|에스\.?에이|에이\.?지|카이샤|컴퍼니|그룹|산업|전자|국방|공단|청$|부$|시$|INC|LTD|CORP|GMBH|LLC', re.I)


def load():
    if _D:
        return _D
    pk = os.path.join(CACHE, f'loaded_{"all" if ALLOW_HOLDOUT else "guard"}.pkl')
    if os.path.exists(pk):
        _D.update(pickle.load(open(pk, 'rb')))
        return _D
    C = CACHE + os.sep
    pt = pd.read_parquet(C + 'patent.parquet')
    pt['rd'] = d64(pt.rgstDate)
    pt['ad'] = d64(pt.applDate)
    pt['exp'] = d64(pt.cndrtExptnDate)
    pt['exd'] = d64(pt.rfoexDate)
    pt['opd'] = d64(pt.opnDate)
    pt = pt[~pd.isna(pt.rd) & (pt.rgst_no.str[:2] == '10')].reset_index(drop=True)   # 특허만(실용신안 20 제외)
    pt['pid'] = np.arange(len(pt))
    pid = pd.Series(pt.pid.values, index=pt.rgst_no.values)
    # 만료일 결측이면 출원일+20년
    m = pd.isna(pt.exp)
    pt.loc[m, 'exp'] = add_years(pt.loc[m, 'ad'].values, 20)
    # --- pay
    pay = pd.read_parquet(C + 'pay.parquet')
    pay['pid'] = pay.rgst_no.map(pid)
    pay = pay[pay.pid.notna()].copy()
    pay['pid'] = pay.pid.astype(int)
    pay['s'] = pd.to_numeric(pay.statAnnl, errors='coerce')
    pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
    pay['amt'] = pd.to_numeric(pay.payAmount, errors='coerce')
    pay['pd'] = d64(pay.payDate)
    pay = pay[pay.l.notna() & ~pd.isna(pay.pd)].copy()
    pay['s'] = pay.s.fillna(pay.l).astype(int)
    pay['l'] = pay.l.astype(int)
    rdv = pt.rd.values[pay.pid.values]
    pay['due'] = add_years(rdv, pay.s.values - 1)          # s년차 납부기한(시작 기념일)
    pay['early'] = (pay.due.values - pay.pd.values).astype('timedelta64[D]').astype(float)
    pay = pay.sort_values(['pid', 'pd', 'seq']).reset_index(drop=True)
    # --- right
    rt = pd.read_parquet(C + 'right.parquet')
    rt['pid'] = rt.rgst_no.map(pid)
    rt = rt[rt.pid.notna()].copy()
    rt['pid'] = rt.pid.astype(int)
    rt['cd'] = d64(rt.rgstCsDate)
    # --- 소멸(불납) 이벤트: E = 등록일 + (소멸등록일 전 최대 납부연차)년
    lap = rt[(rt.rgstCsName == '소멸등록') & (rt.rgstCsReason.isin(['등록료불납', '연차료불납'])) & ~pd.isna(rt.cd)][['pid', 'cd']].copy()
    pp = pay[['pid', 'pd', 'l']]
    mm = lap.reset_index().merge(pp, on='pid', how='left')
    mm = mm[mm.pd < mm.cd]
    lmax = mm.groupby('index').l.max()
    lap['lmax'] = lap.index.map(lmax)
    lap = lap[lap.lmax.notna()].copy()
    lap['E'] = add_years(pt.rd.values[lap.pid.values], lap.lmax.values.astype(int))
    if not ALLOW_HOLDOUT:
        lap = lap[lap.E < HOLDOUT_START]            # 🔒 홀드아웃 결과 차단
    lap = lap.sort_values(['pid', 'E']).reset_index(drop=True)
    # --- 기타 권리 이벤트
    ev = {}
    ev['recov'] = rt[rt.rgstCsName == '회복등록'][['pid', 'cd']]
    ev['malso'] = rt[rt.rgstCsName.isin(['권리의 말소등록', '말소등록', '직권말소등록'])][['pid', 'cd']]
    ev['partdel'] = rt[rt.rgstCsName == '권리의 일부말소등록'][['pid', 'cd']]
    ev['techeval'] = rt[rt.rgstCsName.str.contains('기술평가', na=False)][['pid', 'cd']]
    ev['trial'] = rt[rt.rgstCsName.str.contains('심판|심결|소송|상고|이의신청|취소신청', na=False)][['pid', 'cd']]
    ev['expire'] = rt[(rt.rgstCsName == '소멸등록') & (rt.rgstCsReason.isin(['존속기간만료', '기간만료']))][['pid', 'cd']]
    # --- owner (이전 이력)
    ow = pd.read_parquet(C + 'owner.parquet')
    ow['pid'] = ow.rgst_no.map(pid)
    ow = ow[ow.pid.notna()].copy()
    ow['pid'] = ow.pid.astype(int)
    ow['od'] = d64(ow.ownerRgstCsDate)
    ow['corp'] = ow.ownerName.fillna('').str.contains(CORP_RE)
    ow['foreign'] = ow.ownerNatl.notna() & (ow.ownerNatl != '대한민국')
    # --- applicant
    ap = pd.read_parquet(C + 'applicant.parquet')
    ap['pid'] = ap.rgst_no.map(pid)
    ap = ap[ap.pid.notna()].copy()
    ap['pid'] = ap.pid.astype(int)
    a1 = ap.sort_values(['pid', 'seq']).drop_duplicates('pid')
    pt['acd'] = pt.pid.map(a1.set_index('pid').applicantCd).fillna('NA')
    pt['atype'] = pt.acd.str[0].map({'1': 'corp', '2': 'univ', '3': 'inst', '4': 'indiv', '5': 'fcorp', '6': 'findiv'}).fillna('other')
    pt['aname'] = pt.pid.map(a1.set_index('pid').applicantName)
    pt['n_appl'] = pt.pid.map(ap.groupby('pid').size()).fillna(1)
    pt['n_appl_types'] = pt.pid.map(ap.assign(t=ap.applicantCd.str[0]).groupby('pid').t.nunique()).fillna(1)
    # --- 기타 정적
    inv = pd.read_parquet(C + 'inventor_agg.parquet')
    pt['n_inv'] = pt.rgst_no.map(inv.set_index('rgst_no').n_inv).fillna(0)
    pt['n_inv_f'] = pt.rgst_no.map(inv.set_index('rgst_no').n_inv_foreign).fillna(0)
    cf = pd.read_parquet(C + 'cofpr.parquet')
    pt['n_cofpr'] = pt.rgst_no.map(cf.groupby('rgst_no').size()).fillna(0)
    pt['cofpr_natl'] = pt.rgst_no.map(cf.dropna(subset=['cofprNatl']).drop_duplicates('rgst_no').set_index('rgst_no').cofprNatl).fillna('none')
    ag = pd.read_parquet(C + 'agent.parquet')
    pt['n_agent'] = pt.rgst_no.map(ag.set_index('rgst_no').n_agent).fillna(0)
    pt['claims'] = pd.to_numeric(pt.claimCount, errors='coerce')
    pt['pct'] = (pt.applTpcd == '국제출원').astype(int)
    pt['ipc1'] = pt.ipcCd.fillna('').str.split(',').str[0].str.strip()
    pt['ipc_sub'] = pt.ipc1.str[:4].replace('', 'NA')
    pt['ipc_cls'] = pt.ipc1.str[:3].replace('', 'NA')
    pt['ipc_sec'] = pt.ipc1.str[:1].replace('', 'NA')
    pt['n_ipc'] = pt.ipcCd.fillna('').str.count(r'\(') .clip(lower=1)
    pt['n_cpc'] = pt.cpcCd.fillna('').str.count(r'\(')
    pt['title_len'] = pt.title.fillna('').str.len()
    pt['dspst'] = pt.lastDspst.fillna('NA').str.strip()
    pt['contested'] = pt.dspst.str.contains('재심사|전치|환송|심판원', regex=True).astype(int)
    pt['grant_lag'] = yrs(pt.rd.values, pt.ad.values)
    pt['exam_lag'] = yrs(pt.exd.values, pt.ad.values)
    lic = pd.concat([pd.read_parquet(C + 'els.parquet').rename(columns={'elsLcnsStartDate': 'st'})[['rgst_no', 'st']].assign(kind='els'),
                     pd.read_parquet(C + 'noels.parquet').rename(columns={'noelsLcnsStartDate': 'st'})[['rgst_no', 'st']].assign(kind='noels')])
    lic['pid'] = lic.rgst_no.map(pid)
    lic = lic[lic.pid.notna()].copy()
    lic['pid'] = lic.pid.astype(int)
    lic['sd'] = d64(lic.st)
    # 첫 납부(설정등록료) 금액/청구항 -> 감면 추정
    p0 = pay[pay.seq == 0].drop_duplicates('pid').set_index('pid')
    pt['init_span'] = pt.pid.map(p0.l).fillna(3)
    pt['init_amt'] = pt.pid.map(p0.amt)
    # 메모리 절약: 파싱 끝난 문자열 열 제거
    pay = pay[['pid', 'seq', 's', 'l', 'amt', 'pd', 'due', 'early']].copy()
    pay['seq'] = pay.seq.astype('int32'); pay['s'] = pay.s.astype('int16'); pay['l'] = pay.l.astype('int16')
    pay['amt'] = pay.amt.astype('float32'); pay['early'] = pay.early.astype('float32'); pay['pid'] = pay.pid.astype('int32')
    rt = rt[['pid', 'cd', 'rgstCsName', 'rgstCsReason']].copy()
    ow = ow[['pid', 'seq', 'od', 'corp', 'foreign', 'ownerRgstCsName']].copy()
    pt = pt.drop(columns=['engTitle', 'cpcCd', 'ipcCd', 'claimCount', 'cndrtExptnDate', 'rfoexDate', 'opnDate', 'pubDate', 'applDate',
                          'title', 'aname', 'lastDspst', 'ipc1', 'rgstDate', 'applTpcd', 'rgstNo', 'opnNo', 'pubNo', 'rfoexYn'], errors='ignore')
    _D.update(dict(pt=pt, pay=pay, rt=rt, lap=lap, ev=ev, ow=ow, ap=None, lic=lic))
    try:
        pickle.dump(dict(_D), open(pk, 'wb'), protocol=4)
    except Exception as e:
        print('load cache save failed', e)
    return _D


# ------------------------------------------------------------------ 위험집합·레이블
def paid_through(L, asof=None):
    """asof 시점까지의 납부로 본 납부완료일(= rd + max lastAnnl 년). 반환: pid-indexed datetime64 배열"""
    D = load(); pt, pay = D['pt'], D['pay']
    asof = L if asof is None else asof
    p = pay[pay.pd <= asof]
    lm = p.groupby('pid').l.max()
    l = np.zeros(len(pt)); l[lm.index.values] = lm.values
    out = np.full(len(pt), np.datetime64('NaT'), dtype='datetime64[D]')
    ok = l > 0
    out[ok] = add_years(pt.rd.values[ok], l[ok].astype(int))
    return out, l


def risk_set(L):
    """L 시점 위험집합 pid 배열 + 납부완료일"""
    D = load(); pt = D['pt']
    L = np.datetime64(L)
    pth, l = paid_through(L)
    ok = (pt.rd.values < L) & (pth > L) & (pt.exp.values > L)
    ml = D['ev']['malso']; bad = ml[ml.cd <= L].pid.unique()
    ok[bad] = False
    return np.where(ok)[0], pth, l


def label_full(L, pids):
    """검증용 완전 레이블(전체 자료). E∈(L, L+3y]. 홀드아웃 L이면 예외."""
    L = np.datetime64(L)
    end = add_years(np.array([L]), H_YEARS)[0]
    if (end > HOLDOUT_START) and not ALLOW_HOLDOUT:
        raise RuntimeError('🔒 홀드아웃 결과 구간 — 1·2단계 금지')
    lap = load()['lap']
    e = lap[(lap.E > L) & (lap.E <= end)].pid.unique()
    return np.isin(pids, e).astype(int)


def label_asof(L, pids, C):
    """학습용 레이블: C 시점까지 공개된 정보만. 1=C 전 소멸등록 & E∈(L,L+3y], 0=C까지 납부로 L+3y 넘김 or 만료, NaN=미확정"""
    D = load(); pt, lap = D['pt'], D['lap']
    L = np.datetime64(L); C = np.datetime64(C)
    end = add_years(np.array([L]), H_YEARS)[0]
    e = lap[(lap.E > L) & (lap.E <= end) & (lap.cd <= C)].pid.unique()
    y = np.full(len(pids), np.nan)
    pthC, _ = paid_through(C, asof=C)
    zero = (pthC[pids] > end) | ((pt.exp.values[pids] <= end) & (pt.exp.values[pids] <= C) & (pthC[pids] >= pt.exp.values[pids]))
    ml = D['ev']['malso']
    mz = ml[(ml.cd > L) & (ml.cd <= min(end, C))].pid.unique()
    zero = zero | np.isin(pids, mz)          # 창 안 말소(포기·무효)는 불납 아님 -> 0
    y[zero] = 0
    y[np.isin(pids, e)] = 1
    return y


# ------------------------------------------------------------------ 피처
def _cnt_before(df, dcol, L, n, pids=None):
    c = df[df[dcol] <= L].groupby('pid').size()
    a = np.zeros(n); a[c.index.values] = c.values
    return a


def _last_before(df, dcol, L, n):
    c = df[df[dcol] <= L].groupby('pid')[dcol].max()
    a = np.full(n, np.nan); a[c.index.values] = yrs(np.full(len(c), L), c.values)
    return a


def guard_L(L):
    if np.datetime64(L) >= HOLDOUT_START and not ALLOW_HOLDOUT:
        raise RuntimeError('🔒 홀드아웃 기준일 피처는 T4_STAGE3_HOLDOUT=1 (3단계)에서만 계산')


def features(L, use_cache=True):
    guard_L(L)
    L = np.datetime64(L)
    fn = os.path.join(CACHE, f'feat_{str(L).replace("-", "")}.parquet')
    if use_cache and os.path.exists(fn):
        return pd.read_parquet(fn)
    D = load(); pt, pay, lap, ev, ow, lic = D['pt'], D['pay'], D['lap'], D['ev'], D['ow'], D['lic']
    n = len(pt)
    pids, pth, lpaid = risk_set(L)
    end = add_years(np.array([L]), H_YEARS)[0]
    f = pd.DataFrame({'pid': pids})
    P = pt.iloc[pids]
    # ---- 기준선 공변량
    f['age'] = yrs(np.full(len(pids), L), P.rd.values)
    f['atype'] = P.atype.values
    f['ksic'] = P.ksic.values
    f['claims'] = P.claims.values
    f['rgst_year'] = pd.DatetimeIndex(P.rd.values).year.values
    # ---- 납부 구조
    f['paid_ahead'] = yrs(pth[pids], np.full(len(pids), L))           # 다음 납부기한까지 년
    f['paid_year'] = lpaid[pids]                                        # 납부 완료 연차
    f['next_annl'] = lpaid[pids] + 1
    f['n_due_win'] = np.clip(np.ceil(yrs(np.full(len(pids), end), pth[pids])), 0, None)  # 창 안의 납부기한 수(근사)
    f['to_expiry'] = yrs(P.exp.values, np.full(len(pids), L))
    f['exp_in_win'] = (P.exp.values <= end).astype(int)
    f['tier_jump'] = np.isin(f.next_annl.values, [4, 7, 10, 13]).astype(int)
    f['tier_in_win'] = 0
    for k in [4, 7, 10, 13]:
        f['tier_in_win'] += ((f.next_annl.values <= k) & (f.next_annl.values + f.n_due_win.values - 1 >= k)).astype(int)
    f['age_total'] = yrs(np.full(len(pids), L), P.ad.values)
    # ---- L 이전 납부행동
    p = pay[(pay.pd <= L) & (pay.pid.isin(pids))]
    pr = p[p.seq > 0].assign(late=lambda d: (d.early < 0).astype(int), span=lambda d: d.l - d.s + 1)
    agg = pr.groupby('pid').agg(n_renew=('seq', 'size'), n_late=('late', 'sum'), mean_early=('early', 'mean'),
                                min_early=('early', 'min'), max_span=('span', 'max'))
    last = pr.groupby('pid').tail(1).set_index('pid')
    agg['last_early'] = last.early
    agg['last_span'] = last.l - last.s + 1
    agg['last_amt_per_yr'] = last.amt / (last.l - last.s + 1)
    agg['yrs_since_pay'] = yrs(np.full(len(last), L), last.pd.values)
    f = f.join(agg, on='pid')
    f['n_renew'] = f.n_renew.fillna(0); f['n_late'] = f.n_late.fillna(0)
    f['late_share'] = f.n_late / f.n_renew.replace(0, np.nan)
    # 감면 추정: 같은 (납부연도, 연차, 청구항) 중위 금액 대비
    pr2 = pay[(pay.pd <= L) & (pay.seq > 0) & (pay.l == pay.s)].copy()
    pr2['py'] = pd.DatetimeIndex(pr2.pd.values).year
    pr2['cl'] = pt.claims.values[pr2.pid.values]
    med = pr2.groupby(['py', 's', 'cl']).amt.transform('median')
    pr2['rel'] = pr2.amt / med.replace(0, np.nan)
    lr = pr2.groupby('pid').tail(1).set_index('pid').rel
    f['fee_rel'] = f.pid.map(lr)
    p0 = pay[(pay.seq == 0) & (pay.pd <= L)].copy()
    p0['py'] = pd.DatetimeIndex(p0.pd.values).year
    p0['cl'] = pt.claims.values[p0.pid.values]
    p0['sp'] = p0.l - p0.s
    med0 = p0.groupby(['py', 'sp', 'cl']).amt.transform('median')
    p0['rel0'] = p0.amt / med0.replace(0, np.nan)
    f['init_fee_rel'] = f.pid.map(p0.drop_duplicates('pid').set_index('pid').rel0)
    f['init_span'] = P.init_span.values
    # ---- 권리 이벤트 이력(L 이전)
    for k in ['recov', 'partdel', 'techeval', 'trial']:
        f['n_' + k] = _cnt_before(ev[k], 'cd', L, n)[pids]
    f['n_prev_lapse'] = _cnt_before(lap.rename(columns={'cd': 'rcd'}), 'rcd', L, n)[pids]
    # ---- 권리이전(L 이전)
    o = ow[(ow.seq > 0) & (ow.od <= L)]
    f['n_transfer'] = _cnt_before(o, 'od', L, n)[pids]
    f['yrs_since_tr'] = _last_before(o, 'od', L, n)[pids]
    cur = ow[(ow.od <= L) | ow.od.isna()].sort_values(['pid', 'seq']).groupby('pid').tail(1).set_index('pid')
    f['own_corp'] = f.pid.map(cur.corp).astype(float)
    f['own_foreign'] = f.pid.map(cur.foreign).astype(float)
    trr = o.sort_values(['pid', 'seq']).groupby('pid').tail(1).set_index('pid').ownerRgstCsName.fillna('기타')
    f['tr_kind'] = f.pid.map(trr).fillna('none')
    f['n_owner_now'] = f.pid.map(ow[(ow.od <= L)].groupby('pid').size()).fillna(1)
    # ---- 실시권(L 이전)
    f['n_lic'] = _cnt_before(lic, 'sd', L, n)[pids]
    # ---- 정적 서지
    for c in ['n_appl', 'n_appl_types', 'n_inv', 'n_inv_f', 'n_cofpr', 'n_agent', 'pct', 'n_ipc', 'n_cpc', 'title_len',
              'contested', 'grant_lag', 'exam_lag']:
        f[c] = P[c].values
    f['cofpr_natl'] = P.cofpr_natl.values
    f['ipc_sub'] = P.ipc_sub.values
    f['ipc_cls'] = P.ipc_cls.values
    f['ipc_sec'] = P.ipc_sec.values
    f['dspst'] = P.dspst.values
    # ---- 출원인 포트폴리오 & 과거 포기율(L 이전만)
    acd = pt.acd.values
    alive = np.zeros(n, bool); alive[pids] = True
    reg = pt.rd.values <= L
    df_a = pd.DataFrame({'acd': acd, 'reg': reg, 'alive': alive, 'new3': (pt.rd.values > add_years(np.array([L]), -3)[0]) & reg,
                         'ksic': pt.ksic.values})
    ga = df_a[df_a.reg].groupby('acd')
    port = pd.DataFrame({'port_n': ga.size(), 'port_alive': ga.alive.sum(), 'port_new3': ga.new3.sum(), 'port_ksic_n': ga.ksic.nunique()})
    a_ren = pr.assign(acd=acd[pr.pid.values]).groupby('acd').size()   # 주의: pr은 위험집합 한정 -> 전체로 다시
    prall = pay[(pay.pd <= L) & (pay.seq > 0)]
    a_ren = pd.Series(acd[prall.pid.values]).value_counts()
    lp = lap[lap.cd <= L]
    a_lap = pd.Series(acd[lp.pid.values]).value_counts()
    L2 = add_years(np.array([L]), -2)[0]
    a_ren2 = pd.Series(acd[prall[prall.pd > L2].pid.values]).value_counts()
    a_lap2 = pd.Series(acd[lp[lp.cd > L2].pid.values]).value_counts()
    port['a_ren'] = a_ren; port['a_lap'] = a_lap; port['a_ren2'] = a_ren2; port['a_lap2'] = a_lap2
    port = port.fillna(0)
    glob_rate = len(lp) / max(len(lp) + len(prall), 1)
    port['a_lapse_rate'] = (port.a_lap + 5 * glob_rate) / (port.a_lap + port.a_ren + 5)
    port['a_lapse_rate2'] = (port.a_lap2 + 5 * glob_rate) / (port.a_lap2 + port.a_ren2 + 5)
    port['a_decisions'] = port.a_lap + port.a_ren
    ak = P.acd.values
    for c in ['port_n', 'port_alive', 'port_new3', 'port_ksic_n', 'a_lapse_rate', 'a_lapse_rate2', 'a_decisions']:
        f[c] = port[c].reindex(ak).values
    f.loc[P.acd.values == 'NA', ['a_lapse_rate', 'a_lapse_rate2']] = np.nan
    same_k = df_a[df_a.reg].groupby(['acd', 'ksic']).size()
    f['port_share_ksic'] = same_k.reindex(list(zip(ak, P.ksic.values))).values / f.port_n.values
    # ---- 같은 KSIC 최근 1년 포기율(L 이전)
    L1 = add_years(np.array([L]), -1)[0]
    k_l = pd.Series(pt.ksic.values[lp[lp.cd > L1].pid.values]).value_counts()
    k_r = pd.Series(pt.ksic.values[prall[prall.pd > L1].pid.values]).value_counts()
    kr = (k_l / (k_l + k_r)).fillna(0)
    f['ksic_lapse_1y'] = kr.reindex(f.ksic.values).values
    k_l3 = pd.Series(pt.ksic.values[lp[(lp.cd > add_years(np.array([L]), -3)[0]) & (lp.cd <= L1)].pid.values]).value_counts()
    k_r3 = pd.Series(pt.ksic.values[prall[(prall.pd > add_years(np.array([L]), -3)[0]) & (prall.pd <= L1)].pid.values]).value_counts()
    f['ksic_lapse_trend'] = f.ksic_lapse_1y - (k_l3 / (k_l3 + k_r3)).reindex(f.ksic.values).values
    # IPC 서브클래스 최근 2년 포기율
    s_l = pd.Series(pt.ipc_sub.values[lp[lp.cd > L2].pid.values]).value_counts()
    s_r = pd.Series(pt.ipc_sub.values[prall[prall.pd > L2].pid.values]).value_counts().reindex(s_l.index.union(pt.ipc_sub.unique())).fillna(0)
    s_l = s_l.reindex(s_r.index).fillna(0)
    f['ipc_lapse_2y'] = ((s_l + 10 * glob_rate) / (s_l + s_r + 10)).reindex(f.ipc_sub.values).values
    f['L'] = str(L)
    f.to_parquet(fn, index=False)
    return f
