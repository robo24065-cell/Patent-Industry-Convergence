# C14-T4 b2: U6×Eb(이종 산업 공동출원 관계 중단) 쌍 단위·결과·피처 (PREREG §2.2·2.4). 공개 < 2025만.
# 입력(읽기만): C13 cache app_/docs_{CN,EP}, C12 cache docs_KR(kt=TXT-FREE), agg5, ksic_list. 출력 cache/b2_u_{t0}.parquet
import sys, os, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
import scipy.sparse as sp
sys.path.insert(0, C13 + r'\scripts')
from lib13 import parse_ipc, build_pub_tables, apply_pub   # 같은 공개 배정 규칙(읽기만)
t = time.time()
T0S = list(range(2013, 2024))
SYSS = ['prov', 'free', 'hsc']

APP, DOC = {}, {}
for cc in CCS:
    guard_parquet_pub(C13C + rf'\app_{cc}.parquet'); guard_parquet_pub(C13C + rf'\docs_{cc}.parquet')
    a = pd.read_parquet(C13C + rf'\app_{cc}.parquet'); a['key'] = cc + ':' + a.key
    d = pd.read_parquet(C13C + rf'\docs_{cc}.parquet', columns=['appl_no', 'pub', 'ksic', 'main_ipc'])
    d['sc'], d['mg'], d['sg'] = parse_ipc(d.main_ipc); d = d.drop(columns='main_ipc')
    d = d[d.appl_no.isin(a.appl_no)].reset_index(drop=True)
    a['dk'] = cc + ':' + a.appl_no; d['dk'] = cc + ':' + d.appl_no
    APP[cc] = a[['dk', 'key', 'pub']]; DOC[cc] = d[['dk', 'pub', 'ksic', 'sc', 'mg', 'sg']]
A = pd.concat(APP.values(), ignore_index=True); D = pd.concat(DOC.values(), ignore_index=True)
del APP, DOC
assert A.pub.max() < CUT and D.pub.max() < CUT
kr = pd.read_parquet(C12C + r'\docs_KR.parquet', columns=['pub', 'kt', 'sc', 'mg', 'sg'])
assert kr.pub.max() < CUT
K = json.load(open(C12C + r'\ksic_list.json'))
ag = pd.read_parquet(C12C + r'\agg5.parquet'); assert ag.y.max() < 2025
ag['prov'] = ag.k.map(dict(enumerate(K)))
out('loaded', len(A), len(D), round(time.time() - t, 1))

# 쌍 공동 문서(순서 없음) + 순서쌍 첫 공동일(파트너 수)
nk = A.groupby('dk').key.nunique()
m = A[A.dk.map(nk) >= 2]
E = m.merge(m[['dk', 'key']], on='dk', suffixes=('', '_g'))
EO = E[E.key != E.key_g]
FIRSTO = EO.groupby(['key', 'key_g']).pub.min().rename('first').reset_index()
E = E[E.key < E.key_g][['key', 'key_g', 'pub']].reset_index(drop=True)
PFIRST = E.groupby(['key', 'key_g']).pub.agg(['min', 'max']).rename(columns={'min': 'pfirst', 'max': 'plast'})
del m, EO


def fyear(d):
    d = np.asarray(d, dtype=np.int64)
    return d // 10000 + ((d // 100) % 100 - 1) / 12.0 + (d % 100 - 1) / 365.0


def pub_labels(t0, sub):
    tabs = build_pub_tables(kr, (t0 + 1) * 10000 + 101)
    return {'free': apply_pub(sub, tabs, 'FREE-IPC'), 'hsc': apply_pub(sub, tabs, 'H-sc-1'), 'tabs': tabs}


def kg5_map(t0, ag_lab):
    Y = t0 + 1
    g1 = ag[(ag.y >= Y - 3) & (ag.y <= Y - 1)]; g0 = ag[(ag.y >= Y - 6) & (ag.y <= Y - 4)]
    l1 = ag_lab.loc[g1.index]; l0 = ag_lab.loc[g0.index]
    n1 = g1.n.groupby(l1.values).sum(); n0 = g0.n.groupby(l0.values).sum()
    N1, N0 = g1.n.sum(), g0.n.sum()
    idx = n1.index.union(n0.index)
    return (np.log((n1.reindex(idx).fillna(0) + 1) / (N1 + 50)) - np.log((n0.reindex(idx).fillna(0) + 1) / (N0 + 50)))


def csr_from(keys, cols, kidx, cidx):
    r = kidx.get_indexer(keys); c = cidx.get_indexer(cols)
    ok = (r >= 0) & (c >= 0)
    X = sp.csr_matrix((np.ones(ok.sum(), np.float32), (r[ok], c[ok])), shape=(len(kidx), len(cidx)))
    X.sum_duplicates()
    return X


def pair_cos(X, i, j):
    n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    dot = np.asarray(X[i].multiply(X[j]).sum(1)).ravel()
    den = n[i] * n[j]
    return np.where(den > 0, dot / np.where(den > 0, den, 1), 0.0)


HIST = {}   # t0 -> DataFrame(key,key_g, y, modal_f_L, modal_g_L ...) 산업쌍 실현율용
log = []
for t0 in T0S:
    end = t0 * 10000 + 1231
    a5 = A[(A.pub > end - 50000) & (A.pub <= end)]
    d5 = D[(D.pub > end - 50000) & (D.pub <= end)].copy()
    PL = pub_labels(t0, d5)
    d5['prov'] = d5.ksic.astype(str); d5['free'] = PL['free'].values; d5['hsc'] = PL['hsc'].values
    a5 = a5.merge(d5[['dk', 'prov', 'free', 'hsc', 'sc']], on='dk', how='left')
    modal, ent, Xl = {}, {}, {}
    keys5 = pd.Index(a5.key.unique())
    for L in SYSS:
        lc = a5[a5[L].notna()].groupby(['key', L]).size().rename('n').reset_index().sort_values(['key', 'n', L], ascending=[True, False, True])
        modal[L] = lc.drop_duplicates('key').set_index('key')[L]
        tot = lc.groupby('key').n.sum(); p = lc.n / lc.key.map(tot)
        ent[L] = (-(p * np.log(p))).groupby(lc.key).sum()
        Xl[L] = sp.csr_matrix((lc.n.values.astype(np.float32), (keys5.get_indexer(lc.key), pd.Index(sorted(lc[L].unique())).get_indexer(lc[L]))), shape=(len(keys5), lc[L].nunique()))
    scs = a5[a5.sc.notna()]
    sci = pd.Index(sorted(scs.sc.unique()))
    Xsc = sp.csr_matrix((np.ones(len(scs), np.float32), (keys5.get_indexer(scs.key), sci.get_indexer(scs.sc))), shape=(len(keys5), len(sci))); Xsc.sum_duplicates()
    # 위험 쌍: 활성 이종 산업 관계(제공 최빈)
    e3 = E[(E.pub > end - 30000) & (E.pub <= end)]
    n3 = e3.groupby(['key', 'key_g']).size().rename('n3'); n1 = e3[e3.pub > end - 10000].groupby(['key', 'key_g']).size().rename('n1')
    R = pd.concat([n3, n1], axis=1).fillna(0).reset_index()
    R = R[(R.n3 >= 2) & (R.n1 >= 1)].copy()
    R['mf_prov'] = R.key.map(modal['prov']); R['mg_prov'] = R.key_g.map(modal['prov'])
    R = R[R.mf_prov.notna() & R.mg_prov.notna() & (R.mf_prov != R.mg_prov)].reset_index(drop=True)
    lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
    ew = E[(E.pub >= lo) & (E.pub <= hi)].groupby(['key', 'key_g']).size().rename('nw')
    R = R.merge(ew, left_on=['key', 'key_g'], right_index=True, how='left'); R['nw'] = R.nw.fillna(0)
    R['y'] = (R.nw == 0).astype(np.int8)
    R['cc'] = R.key.str[:2]
    for L in ['free', 'hsc']:
        R[f'mf_{L}'] = R.key.map(modal[L]); R[f'mg_{L}'] = R.key_g.map(modal[L])
    HIST[t0] = R[['key', 'key_g', 'y'] + [f'm{s}_{L}' for L in SYSS for s in 'fg']].copy()
    if t0 < 2016:
        log.append({'t0': t0, 'n': len(R), 'pos': int(R.y.sum())}); continue
    # 기준 피처
    R['b_n3'] = np.log1p(R.n3); R['b_n1'] = np.log1p(R.n1)
    # 첫·마지막 공동일은 end 이전 문서로만 계산
    eb =E[E.pub <= end].groupby(['key', 'key_g']).pub.agg(['min', 'max']).reindex(pd.MultiIndex.from_frame(R[['key', 'key_g']]))
    R['b_age'] = fyear(end) - fyear(eb['min'].values); R['b_rec'] = fyear(end) - fyear(eb['max'].values)
    c3 = A[(A.pub > end - 30000) & (A.pub <= end)].groupby('key').dk.nunique()
    sf, sg_ = R.key.map(c3).fillna(0).values, R.key_g.map(c3).fillna(0).values
    R['b_szmin'] = np.log1p(np.minimum(sf, sg_)); R['b_szmax'] = np.log1p(np.maximum(sf, sg_))
    R['b_jshare'] = R.n3 / np.maximum(np.minimum(sf, sg_), 1)
    npart = FIRSTO[FIRSTO['first'] <= end].groupby('key').size()
    pa, pb = R.key.map(npart).fillna(0).values, R.key_g.map(npart).fillna(0).values
    R['b_npmin'] = np.log1p(np.minimum(pa, pb)); R['b_npmax'] = np.log1p(np.maximum(pa, pb))
    i = keys5.get_indexer(R.key); j = keys5.get_indexer(R.key_g)
    assert (i >= 0).all() and (j >= 0).all()
    R['b_sccos'] = pair_cos(Xsc, i, j)
    R['b_iscn'] = (R.cc == 'CN').astype(np.int8)
    # 체계 피처
    past = {L: [] for L in SYSS}
    for L in SYSS:
        R[f'{L}_cos'] = pair_cos(Xl[L], i, j)
        mf, mg = R[f'mf_{L}'], R[f'mg_{L}']
        R[f'{L}_same'] = ((mf == mg) & mf.notna()).astype(np.int8)
        ef, eg = R.key.map(ent[L]).fillna(0).values, R.key_g.map(ent[L]).fillna(0).values
        R[f'{L}_entmin'] = np.minimum(ef, eg); R[f'{L}_entmax'] = np.maximum(ef, eg)
        P = pd.concat([HIST[x] for x in (t0 - 3, t0 - 2, t0 - 1)])
        P = P[P[f'mf_{L}'].notna() & P[f'mg_{L}'].notna()]
        pk = np.where(P[f'mf_{L}'].astype(str) < P[f'mg_{L}'].astype(str), P[f'mf_{L}'].astype(str) + '|' + P[f'mg_{L}'].astype(str), P[f'mg_{L}'].astype(str) + '|' + P[f'mf_{L}'].astype(str))
        st = P.y.groupby(pk).agg(['sum', 'size'])
        rate = (st['sum'] + 1) / (st['size'] + 20)
        rk = np.where(mf.astype(str) < mg.astype(str), mf.astype(str) + '|' + mg.astype(str), mg.astype(str) + '|' + mf.astype(str))
        rr = pd.Series(rk).map(rate)
        R[f'{L}_ipr_na'] = rr.isna().values.astype(np.int8) | (mf.isna() | mg.isna()).values.astype(np.int8)
        R[f'{L}_ipr'] = rr.fillna(float(P.y.mean()) if len(P) else 0.5).values
        if L == 'prov':
            ag_lab = ag['prov']
        else:
            ag_lab = apply_pub(ag, PL['tabs'], 'FREE-IPC' if L == 'free' else 'H-sc-1')
        kg = kg5_map(t0, ag_lab)
        kf, kgg = mf.map(kg).astype(float).fillna(0).values, mg.map(kg).astype(float).fillna(0).values
        R[f'{L}_kgmin'] = np.minimum(kf, kgg); R[f'{L}_kgmax'] = np.maximum(kf, kgg)
    R['t0'] = t0
    keep = ['key', 'key_g', 'cc', 't0', 'y', 'nw', 'n3', 'n1', 'mf_prov', 'mg_prov', 'mf_free', 'mg_free', 'mf_hsc', 'mg_hsc'] + [c for c in R.columns if c.startswith(('b_', 'prov_', 'free_', 'hsc_'))]
    R[keep].to_parquet(C + rf'\b2_u_{t0}.parquet', index=False)
    log.append({'t0': t0, 'n': len(R), 'pos': int(R.y.sum()), 'modal_free_na': float(R.mf_free.isna().mean()), 'sec': round(time.time() - t, 1)})
    out(log[-1])
    assert time.time() - t < 520
json.dump(log, open(C + r'\b2_log.json', 'w'), indent=1)
out('done', round(time.time() - t, 1))
