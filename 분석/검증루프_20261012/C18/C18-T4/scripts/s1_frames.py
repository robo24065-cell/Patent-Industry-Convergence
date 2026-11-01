# C18-T4 s1: 후보별 표준 평가 프레임(저장된 개발 예측 + 군집 열). 다른 바퀴 캐시는 읽기만.
# 출력 cache/fr_{K}_{v}[_{learner}].parquet : f, k, (s, k2), y, w, base, prov, 공개 팔들
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
t = time.time()
log = {}

# ---------- K1: C12 예측 + 출원인 제공 최빈 KSIC(t0 이전 10년, 동률은 색인 오름차순) ----------
docs, keys_cc, mk = {}, {}, {}
for cc in ['CN', 'EP']:
    d = pd.read_parquet(C12C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k'])
    assert d.pub.max() < 20250101 and (d.did.values == np.arange(len(d))).all()
    a = pd.read_parquet(C12C + rf'\app_{cc}.parquet', columns=['did', 'key'])
    keys = pd.Index(a.key.unique())            # C12 feat12.Country와 같은 aid 색인
    aid = keys.get_indexer(a.key).astype(np.int64)
    docs[cc] = (aid, d.pub.values[a.did.values].astype(np.int64), d.k.values[a.did.values].astype(np.int64))
    keys_cc[cc] = keys
for v in [2020, 2021, 2022, 2023, 2024]:
    t0 = v * 10000 + 101
    P = None
    for g in ['base', 'pubarms', 'ref']:
        q = pd.read_parquet(C12C + rf'\pred_{v}_{g}.parquet')
        if P is None:
            P = q[['cc', 'aid', 's', 'y', 'w']].copy()
        else:
            assert (P.aid.values == q.aid.values).all() and (P.s.values == q.s.values).all() and (P.cc.values == q.cc.values).all()
        for m in q.columns:
            if m in ('pub', 'prov', 'free', 'hsc', 'scagg'):
                P[m] = q[m].values
    P = P.rename(columns={'pub': 'base'})
    kk = np.full(len(P), -1, np.int64)
    chk = {}
    for cc in ['CN', 'EP']:
        aid, pub, k = docs[cc]
        wl = (pub >= t0 - 100000) & (pub < t0) & (k >= 0)
        df = pd.DataFrame({'a': aid[wl], 'k': k[wl]})
        cnt = df.groupby(['a', 'k']).size().rename('n').reset_index().sort_values(['a', 'n', 'k'], ascending=[True, False, True])
        modal = cnt.drop_duplicates('a').set_index('a').k
        m = (P.cc == cc).values
        kk[m] = modal.reindex(P.aid.values[m]).fillna(-1).astype(np.int64).values
        # u 파일과 aid→key 대조(표본)
        u = pd.read_parquet(C12C + rf'\u_{cc}_{v}.parquet', columns=['aid', 'key']).drop_duplicates('aid')
        chk[cc] = float((keys_cc[cc][u.aid.values] == u.key.values).mean())
    SCL = json.load(open(C12C + r'\sc_list.json'))
    sidx = pd.Index(SCL)
    s_g = sidx.get_indexer(P.s.astype(str)).astype(np.int64)
    if (s_g < 0).any():   # 목록 밖 서브클래스는 뒤에 붙임(원점 공통 문자열 해시)
        s_g = np.where(s_g < 0, 100000 + (pd.util.hash_pandas_object(P.s.astype(str), index=False).values % 1000000).astype(np.int64), s_g)
    F = pd.DataFrame({'f': P.aid.values.astype(np.int64) + (P.cc == 'EP').values.astype(np.int64) * 10_000_000, 'k': kk, 's': s_g, 'cc': (P.cc == 'EP').astype(np.int8).values,
                      'y': P.y.values.astype(np.int8), 'w': P.w.values.astype(np.float64)})
    for m in ['base', 'prov', 'free', 'hsc', 'scagg']:
        F[m] = P[m].values
    log[f'K1_{v}'] = {'n': len(F), 'pos': int(F.y.sum()), 'k_missing': int((F.k < 0).sum()), 'k_missing_pos': int(((F.k < 0) & (F.y == 1)).sum()),
                      'n_f': int(F.f.nunique()), 'n_k': int(F.k[F.k >= 0].nunique()), 'n_s': int(F.s.nunique()), 'aid_key_match': chk}
    F.to_parquet(C + rf'\fr_K1_{v}.parquet', index=False)
    out('K1', v, json.dumps(log[f'K1_{v}']), round(time.time() - t, 1))
del docs

# ---------- K2: C13 X1' ----------
q = pd.read_parquet(C13C + r'\s5_pred_2023.parquet')
F = pd.DataFrame({'f': codes(q.u.astype(str)), 'k': codes(q.modal_prov.astype(str)), 'y': q.y.values.astype(np.int8), 'w': 1.0})
for m in ['base', 'prov', 'free', 'hsc']:
    F[m] = q['p_' + m].values
F.to_parquet(C + r'\fr_K2_2023.parquet', index=False)
log['K2_2023'] = {'n': len(F), 'pos': int(F.y.sum()), 'n_f': int(F.f.nunique()), 'n_k': int(F.k.nunique()), 'modal_null': int(q.modal_prov.isna().sum())}
out('K2', json.dumps(log['K2_2023']))

# ---------- K3: C14 U6×Eb ----------
q = pd.read_parquet(C14C + r'\b3_pred_2023.parquet')
allk = pd.Index(sorted(set(q.mf.astype(str)) | set(q.mg.astype(str))))
F = pd.DataFrame({'f': codes(q.key.astype(str) + '|' + q.key_g.astype(str)), 'k': allk.get_indexer(q.mf.astype(str)).astype(np.int64),
                  'k2': allk.get_indexer(q.mg.astype(str)).astype(np.int64), 'y': q.y.values.astype(np.int8), 'w': 1.0})
for m in ['base', 'prov', 'free', 'hsc']:
    F[m] = q['p_' + m].values
F.to_parquet(C + r'\fr_K3_2023.parquet', index=False)
log['K3_2023'] = {'n': len(F), 'pos': int(F.y.sum()), 'n_f': int(F.f.nunique()), 'n_k': int(F.k.nunique()), 'n_k2': int(F.k2.nunique()),
                  'same_k': int((F.k == F.k2).sum())}
out('K3', json.dumps(log['K3_2023']))

# ---------- K4: C17 US 유지료 ----------
for v in [2021, 2022, 2023]:
    for L in ['xgb', 'lr']:
        q = pd.read_parquet(C17C + rf'\pred_{v}_{L}.parquet')
        hsh = lambda x: pd.util.hash_pandas_object(x.astype(str), index=False).values.astype(np.int64)
        F = pd.DataFrame({'f': hsh(q.firm), 'k': q.k_prov.astype(np.int64).values, 'st': q.s.values.astype(np.int8),
                          'y': q.y.values.astype(np.int8), 'w': 1.0})
        F['c'] = hsh(q.firm.astype(str) + '|' + q.k_prov.astype(str))
        for m in ['base', 'prov', 'free', 'hsc', 'off']:
            F[m] = q[m].values
        F.to_parquet(C + rf'\fr_K4_{v}_{L}.parquet', index=False)
        log[f'K4_{v}_{L}'] = {'n': len(F), 'pos': int(F.y.sum()), 'n_f': int(F.f.nunique()), 'n_k': int(F.k.nunique()), 'n_c': int(F.c.nunique()),
                              'firm_null': int(q.firm.isna().sum())}
        out('K4', v, L, json.dumps(log[f'K4_{v}_{L}']))
log['sec'] = round(time.time() - t, 1)
json.dump(log, open(C + r'\s1_log.json', 'w'), indent=1, ensure_ascii=False)
out('done', log['sec'])
