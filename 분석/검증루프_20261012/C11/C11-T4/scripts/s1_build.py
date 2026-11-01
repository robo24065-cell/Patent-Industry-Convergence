# 후보 A/B/C 단위 표 생성. 사용: python -I s1_build.py CAND Y [Y ...]
# 출력 cache/u_{CAND}_{Y}.parquet : 활성 기업 × 50 KSIC 전체 행(위험·레이블 플래그와 세계별 피처)
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import scipy.sparse as sp
from lib11 import *

CAND = sys.argv[1]
YS = [int(x) for x in sys.argv[2:]]
CCS = ['KR'] if CAND == 'B' else ['CN', 'EP']
NK = 50
t_start = time.time()


def load(cc):
    d = pd.read_parquet(C + rf'\docs_{cc}.parquet')
    a = pd.read_parquet(C + rf'\app_{cc}.parquet')
    if cc == 'KR':
        a = a[a.corp]
    assert d.pub.max() < CUT
    m = a.merge(d, on='did', how='inner')
    keys = pd.Index(m.key.unique())
    m['aid'] = keys.get_indexer(m.key).astype(np.int64)
    return d, m, keys


KRDOC = pd.read_parquet(C + r'\docs_KR.parquet', columns=['pub', 'sc', 'mg', 'sg', 'kt'])


def mat(m, lo, hi, lab, ids):
    """[lo,hi) 공개 문서의 (기업, 라벨) 건수 행렬. lab 열 이름(정수 0..49, -1 = 미배정)."""
    s = m[(m.pub >= lo) & (m.pub < hi)]
    s = s[s[lab] >= 0]
    r = ids.get_indexer(s.aid)
    ok = r >= 0
    M = np.zeros((len(ids), NK), np.float32)
    np.add.at(M, (r[ok], s[lab].values[ok].astype(np.int64)), 1)
    return M


def tot(m, lo, hi, ids):
    s = m[(m.pub >= lo) & (m.pub < hi)].drop_duplicates(['aid', 'did'])
    r = ids.get_indexer(s.aid)
    return np.bincount(r[r >= 0], minlength=len(ids)).astype(np.float32)


def add_years(d, y):
    return d + y * 10000


def active_ids(m, t0):
    if CAND == 'B':
        n3 = m[(m.pub >= add_years(t0, -3)) & (m.pub < t0)].drop_duplicates(['aid', 'did']).groupby('aid').size()
        return pd.Index(np.sort(n3[n3 >= 3].index.values))
    s3 = m[(m.pub >= add_years(t0, -3)) & (m.pub < t0)].drop_duplicates(['aid', 'did'])
    n3 = s3.groupby('aid').size()
    n1 = s3[s3.pub >= add_years(t0, -1)].groupby('aid').size()
    ok = n3[n3 >= 10].index.intersection(n1[n1 >= 1].index)
    return pd.Index(np.sort(ok.values))


def risk_event(m, t0, lab, ids, wlo, whi):
    """사건 정의별 위험 행렬과 결과 행렬(창 [wlo,whi))."""
    if CAND == 'A':
        n5 = mat(m, add_years(t0, -5), t0, lab, ids)
        risk = n5 == 0
        nw = mat(m, wlo, whi, lab, ids)
        ev = nw >= 1
    elif CAND == 'B':
        past = mat(m, add_years(t0, -10), add_years(t0, -3), lab, ids)
        n3 = mat(m, add_years(t0, -3), t0, lab, ids)
        risk = (past > 0) & (n3 == 0)
        nw = mat(m, wlo, whi, lab, ids)
        ev = nw >= 1
    else:  # C
        n1 = mat(m, add_years(t0, -1), t0, lab, ids)
        risk = n1 >= 4
        nw = mat(m, wlo, whi, lab, ids)
        span = (whi - wlo) // 10000  # 0 → 6개월 창, 1 → 12개월 창
        if span >= 1:   # 12개월 창: 직전 12개월의 2배 이상, ≥ 8
            ev = (nw >= 2 * n1) & (nw >= 8)
        else:
            ev = (nw >= n1) & (nw >= 4)
    return risk, ev & risk


def world_feats(m, t0, lab, ids, cc, pre):
    """세계(lab)별 [K] 피처. 반환 dict name -> [n,50] 또는 [n] 배열."""
    F = {}
    n3 = mat(m, add_years(t0, -3), t0, lab, ids)
    n1 = mat(m, add_years(t0, -1), t0, lab, ids)
    H = (n3 > 0).astype(np.float32)
    F['nk3'] = np.repeat(H.sum(1, keepdims=True), NK, 1)
    Co = H.T @ H
    dg = np.diag(Co).copy(); dg[dg == 0] = 1
    P = Co / dg[None, :]               # P[k,j] = P(k | j)
    phi = np.minimum(P, P.T); np.fill_diagonal(phi, 0)
    den = phi.sum(1); den[den == 0] = 1
    F['dens'] = (H @ phi.T) / den[None, :]
    mx = np.zeros_like(H)
    for j in range(NK):
        mx = np.maximum(mx, H[:, [j]] * phi[:, j][None, :])
    F['maxphi'] = mx
    # IPC 겹침
    s = m[(m.pub >= add_years(t0, -3)) & (m.pub < t0) & m.sc.notna()]
    sl = s[s[lab] >= 0]
    scs = pd.Index(s.sc.unique())
    Q = np.zeros((len(scs), NK), np.float32)
    np.add.at(Q, (scs.get_indexer(sl.sc), sl[lab].values.astype(np.int64)), 1)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1)
    r = ids.get_indexer(s.aid); ok = r >= 0
    Fm = sp.csr_matrix((np.ones(ok.sum(), np.float32), (r[ok], scs.get_indexer(s.sc)[ok])), shape=(len(ids), len(scs)))
    nd = np.asarray(Fm.sum(1)).ravel(); nd[nd == 0] = 1
    F['meanq'] = np.asarray(Fm @ Q) / nd[:, None]
    F['shq10'] = np.asarray(Fm @ (Q >= 0.10).astype(np.float32)) / nd[:, None]
    # (c,k) 집계 — 나라 전체 문서
    dk3 = np.bincount(sl[lab].values.astype(np.int64), minlength=NK).astype(np.float32)
    s1 = m[(m.pub >= add_years(t0, -1)) & (m.pub < t0)]; s1 = s1[s1[lab] >= 0].drop_duplicates('did')
    dk1 = np.bincount(s1[lab].values.astype(np.int64), minlength=NK).astype(np.float32)
    dk3u = np.bincount(sl.drop_duplicates('did')[lab].values.astype(np.int64), minlength=NK).astype(np.float32)
    F['ck_log3'] = np.repeat(np.log1p(dk3u)[None, :], len(ids), 0)
    F['ck_gr'] = np.repeat((dk1 / np.maximum((dk3u - dk1) / 2, 1))[None, :], len(ids), 0)
    F['ck_nfirm'] = np.repeat(np.log1p(H.sum(0))[None, :], len(ids), 0)
    F['own_sh3'] = n3 / np.maximum(n3.sum(1, keepdims=True), 1)
    F['own_n1'] = n1
    return {pre + k: v for k, v in F.items()}


def persistence(m, t0, lab, keys_now):
    """직전 12개월 같은 사건 실현율: 원점 t0-1y, 창 [t0-1y, t0)."""
    tp = add_years(t0, -1)
    idp = active_ids(m, tp)
    risk, ev = risk_event(m, tp, lab, idp, tp, t0)
    rf = np.where(risk.sum(1) > 0, ev.sum(1) / np.maximum(risk.sum(1), 1), np.nan)
    rk = ev.sum(0) / np.maximum(risk.sum(0), 1)
    rfs = pd.Series(rf, index=idp)
    rf_now = rfs.reindex(keys_now).values.astype(np.float32)
    return rf_now, rk.astype(np.float32)


for Y in YS:
    t0 = ymd(Y); wlo, whi = win(Y); whi_ex = whi + 1
    assert whi_ex <= CUT, '잠금 창은 만들지 않는다'
    parts = []
    for cc in CCS:
        d, m, keys = load(cc)
        m['k_prov'] = m.k.astype(np.int64)
        # 공개 배정 라벨(원점 표)
        tabs = build_pub_tables(KRDOC, t0)
        K = json.load(open(C + r'\ksic_list.json'))
        K2I = {k: i for i, k in enumerate(K)}
        dd = d[['did', 'sc', 'mg', 'sg']].copy()
        arms = {'fi': 'H-sg-3', 'hs': 'H-sc-1'}
        for a, rule in arms.items():
            lab = apply_pub(dd, tabs, rule)
            dd['k_' + a] = lab.map(K2I).fillna(-1).astype(np.int64)
        if cc == 'KR':
            dd['k_tx'] = d.kt.map(K2I).fillna(-1).astype(np.int64)
            arms['tx'] = 'TXT-FREE'
        m = m.merge(dd[['did'] + ['k_' + a for a in arms]], on='did', how='left')
        ids = active_ids(m, t0)
        n = len(ids)
        U = {'cc': np.repeat(cc, n * NK), 'aid': np.repeat(ids.values, NK), 'k': np.tile(np.arange(NK), n)}
        # 활동(라벨 무관)
        n3t = tot(m, add_years(t0, -3), t0, ids); n1t = tot(m, add_years(t0, -1), t0, ids)
        U['f_log3'] = np.repeat(np.log1p(n3t), NK); U['f_log1'] = np.repeat(np.log1p(n1t), NK)
        U['f_gr'] = np.repeat(n1t / np.maximum((n3t - n1t) / 2, 1), NK)
        U['is_cn'] = np.repeat(np.float32(cc == 'CN'), n * NK)
        worlds = {'p': 'k_prov', **{a: 'k_' + a for a in arms}}
        for w, lab in worlds.items():
            risk, ev = risk_event(m, t0, lab, ids, wlo, whi_ex)
            U[f'risk_{w}'] = risk.ravel(); U[f'y_{w}'] = ev.ravel()
            F = world_feats(m, t0, lab, ids, cc, w + '_')
            for k2, v in F.items():
                U[k2] = v.ravel().astype(np.float32)
            rf, rk = persistence(m, t0, lab, ids)
            U[w + '_rfirm'] = np.repeat(rf, NK); U[w + '_rksic'] = np.tile(rk, n)
            if CAND == 'B':
                past = mat(m, add_years(t0, -10), add_years(t0, -3), lab, ids)
                U[w + '_pastn'] = np.log1p(past).ravel()
                U[w + '_pastsh'] = (past / np.maximum(past.sum(1, keepdims=True), 1)).ravel()
                s = m[(m.pub >= add_years(t0, -10)) & (m.pub < add_years(t0, -3))]
                s = s[s[lab] >= 0]
                lp = s.groupby(['aid', lab]).pub.max().reset_index()
                L = np.full((n, NK), np.nan, np.float32)
                r = ids.get_indexer(lp.aid); ok = r >= 0
                L[r[ok], lp[lab].values[ok]] = (t0 - lp.pub.values[ok]) / 10000.0
                U[w + '_since'] = L.ravel()
            if CAND == 'C':
                h1 = mat(m, ymd(Y - 1, 7, 1), t0, lab, ids)
                h0 = mat(m, ymd(Y - 2, 7, 1), ymd(Y - 1, 7, 1), lab, ids)
                U[w + '_sprev'] = (((h1 >= h0) & (h1 >= 4) & (h0 >= 4))).ravel().astype(np.float32)
                U[w + '_rprev'] = np.log((h1 + 1) / (h0 + 1)).ravel()
        parts.append(pd.DataFrame(U))
        out(Y, cc, 'active', n, 'risk_p', int(U['risk_p'].sum()), 'y_p', int(U['y_p'].sum()), round(time.time() - t_start, 1))
    df = pd.concat(parts, ignore_index=True)
    df['Y'] = Y
    df.to_parquet(C + rf'\u_{CAND}_{Y}.parquet', index=False)
out('done', round(time.time() - t_start, 1))
