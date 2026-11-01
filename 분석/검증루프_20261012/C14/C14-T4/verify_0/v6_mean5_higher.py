# C14-T4 verify_0 v6 (v5 복제, 5원점 평균 D(Δskill min) 상위 군집 SE 추가) (반박 재시도): 기업 ⊂ 제공 KSIC, 서브클래스 ⊂ IPC 클래스/섹션 상위 군집에서도 '교차·CGM 충분'이 유지되는지.
# 영향함수(선형화) 기반. C12 캐시 읽기만. 날짜 가드: 문서 pub < 20250101 assert, 원점 v는 pub < (v+1)0101만 사용.
import sys, os, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import norm
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
B0 = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012'
C12C = B0 + r'\C12\C12-T4\cache'
OUT = B0 + r'\C14\C14-T4\verify_0\v6_out.json'
YS = [2020, 2021, 2022, 2023, 2024]
E_LOCK = 15961.345174791417
t0 = time.time()
SCL = np.array(json.load(open(C12C + r'\sc_list.json', encoding='utf-8')))
CLS = pd.factorize(np.array([s[:3] for s in SCL]))[0]
SEC = pd.factorize(np.array([s[0] for s in SCL]))[0]


def wauc_if(score, y, w):
    y = y.astype(bool)
    u, g = np.unique(score, return_inverse=True)
    ng = len(u)
    wp = np.bincount(g[y], weights=w[y], minlength=ng)
    wn = np.bincount(g[~y], weights=w[~y], minlength=ng)
    P, N = wp.sum(), wn.sum()
    below_n = np.cumsum(wn) - wn
    above_p = P - np.cumsum(wp)
    A = float((wp * (below_n + 0.5 * wn)).sum() / (P * N))
    ui = np.where(y, w * ((below_n + 0.5 * wn)[g] / N - A) / P, w * ((above_p + 0.5 * wp)[g] / P - A) / N)
    return A, ui


# 나라별 문서·출원인
CC = {}
for cc in ('CN', 'EP'):
    d = pd.read_parquet(C12C + rf'\docs_{cc}.parquet', columns=['did', 'pub', 'k'])
    assert d.pub.max() < 20250101 and (d.did.values == np.arange(len(d))).all()
    a = pd.read_parquet(C12C + rf'\app_{cc}.parquet', columns=['did', 'key'])
    keys = pd.Index(a.key.unique())
    aid = keys.get_indexer(a.key).astype(np.int64)
    CC[cc] = pd.DataFrame({'aid': aid, 'pub': d.pub.values[a.did.values], 'k': d.k.values[a.did.values]})
    print(cc, 'loaded', len(CC[cc]), round(time.time() - t0, 1), flush=True)


def modal_k(cc, v):
    m = CC[cc]
    x = m[(m.pub < (v + 1) * 10000 + 101) & (m.pub >= (v - 4) * 10000 + 101) & (m.k >= 0)]
    c = x.groupby(['aid', 'k']).size().reset_index(name='n').sort_values(['aid', 'n', 'k'], ascending=[True, False, True])
    return c.drop_duplicates('aid').set_index('aid').k


res = {'single': {}, 'note': 'V_X = Σ_군집X (Σu)^2; CGM(A×B) = V_A + V_B − V_(A,B). K = 기업 as-of 최빈 제공 KSIC(5년), cls = IPC 클래스, sec = 섹션.'}
allrows = []
for v in YS:
    a = pd.read_parquet(C12C + rf'\pred_{v}_base.parquet', columns=['cc', 'aid', 's', 'y', 'w', 'prov'])
    b = pd.read_parquet(C12C + rf'\pred_{v}_pubarms.parquet', columns=['aid', 's', 'free', 'hsc'])
    assert (a.aid.values == b.aid.values).all() and (a.s.values == b.s.values).all()
    a['free'] = b.free.values; a['hsc'] = b.hsc.values
    K = np.full(len(a), -1, np.int64)
    for cc in ('CN', 'EP'):
        mk = modal_k(cc, v)
        sel = (a.cc == cc).values
        K[sel] = a.aid[sel].map(mk).fillna(-1).astype(np.int64).values
    a['K'] = K
    y = a.y.values; w = a.w.values.astype(np.float64)
    A = {}; U = {}
    for m in ('prov', 'free', 'hsc'):
        A[m], U[m] = wauc_if(a[m].values.astype(np.float64), y, w)
    best = 'free' if A['free'] >= A['hsc'] else 'hsc'
    a['u'] = U['prov'] - U[best]
    ds = {m: (A['prov'] - A[m]) / (1 - A[m]) for m in ('free', 'hsc')}; act = min(ds, key=ds.get)
    a['us'] = (U['prov'] / (1 - A[act]) - U[act] * (1 - A['prov']) / (1 - A[act]) ** 2) / 5.0
    a['fk'] = a.cc + ':' + a.aid.astype(str)
    a['Kk'] = a.K.astype(str)                       # KSIC 노드(나라 공통)
    a['cls'] = CLS[a.s.values]; a['sec'] = SEC[a.s.values]
    # 키 조합
    def V(cols):
        return float((a.groupby(cols).u.sum() ** 2).sum())
    Vf, Vs, Vc = V(['fk']), V(['s']), V(['fk', 's'])
    VK, Vcls, Vsec = V(['Kk']), V(['cls']), V(['sec'])
    VKs, VKcls, Vfcls, VKsec = V(['Kk', 's']), V(['Kk', 'cls']), V(['fk', 'cls']), V(['Kk', 'sec'])
    r = {'dAUC': A['prov'] - A[best], 'K_missing_row_frac': float((a.K < 0).mean()), 'nK': int(a.K.nunique()),
         'firm_in_one_K': True,
         'SE_f': Vf ** .5, 'SE_s': Vs ** .5, 'SE_K': VK ** .5, 'SE_cls': Vcls ** .5, 'SE_sec': Vsec ** .5,
         'SE_CGM_f_s': max(Vf + Vs - Vc, 0) ** .5,
         'SE_CGM_K_s': max(VK + Vs - VKs, 0) ** .5,
         'SE_CGM_f_cls': max(Vf + Vcls - Vfcls, 0) ** .5,
         'SE_CGM_K_cls': max(VK + Vcls - VKcls, 0) ** .5,
         'SE_CGM_K_sec': max(VK + Vsec - VKsec, 0) ** .5,
         'npos': int(y.sum())}
    sc = (r['npos'] / E_LOCK) ** .5
    for k in [x for x in r if x.startswith('SE_')]:
        r['pow_' + k[3:]] = float(norm.cdf(0.002 / (r[k] * sc) - 1.96)) if r[k] > 0 else None
    res['single'][v] = r
    allrows.append(a[['fk', 's', 'Kk', 'cls', 'sec', 'us']])
    print(v, json.dumps(r, ensure_ascii=False), round(time.time() - t0, 1), flush=True)
    assert time.time() - t0 < 500
M = pd.concat(allrows, ignore_index=True)
def VM(cols):
    return float((M.groupby(cols).us.sum() ** 2).sum())
Vf, Vs, Vc, VK, Vcls, VKs, VKcls = VM(['fk']), VM(['s']), VM(['fk', 's']), VM(['Kk']), VM(['cls']), VM(['Kk', 's']), VM(['Kk', 'cls'])
res['mean5_D'] = {'SE_f': Vf ** .5, 'SE_s': Vs ** .5, 'SE_K': VK ** .5, 'SE_cls': Vcls ** .5, 'SE_CGM_f_s': (Vf + Vs - Vc) ** .5, 'SE_prod_f_s_analytic': (Vf + Vs + Vc) ** .5, 'SE_CGM_K_s': (VK + Vs - VKs) ** .5, 'SE_CGM_K_cls': (VK + Vcls - VKcls) ** .5}
print('mean5_D', json.dumps(res['mean5_D']), flush=True)
res['secs'] = round(time.time() - t0, 1)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('done', res['secs'])

