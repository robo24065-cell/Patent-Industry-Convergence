# C14-T4 verify_0 v4 (반박 재시도): F1 '교차 구조·곱 가중 과대 = 칸 분산' 독립 재계산.
# 작업 스크립트(lib14·a0·a1·a2·v1) 미사용. C12 저장 예측 parquet만 읽기(pandas 직접).
# (1) 구조 진단 독립 재계산 (2) 가중 AUC 선형화(영향함수)로 군집 분산 V_f, V_s, V_c 해석적 계산
# (3) 곱 Poisson 가중 분산의 해석식 V_f+V_s+V_c 와 CGM V_f+V_s-V_c 비교, (4) 5원점 평균 D(Δskill min)도 동일.
# (5) 곱 가중 직접 몬테카를로(B=300, 새 시드)로 해석식 검증.
import sys, os, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
B0 = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012'
C12C = B0 + r'\C12\C12-T4\cache'
OUT = B0 + r'\C14\C14-T4\verify_0\v4_out.json'
YS = [2020, 2021, 2022, 2023, 2024]
t0 = time.time()


def load(v):
    a = pd.read_parquet(C12C + rf'\pred_{v}_base.parquet')
    b = pd.read_parquet(C12C + rf'\pred_{v}_pubarms.parquet')
    assert (a.aid.values == b.aid.values).all() and (a.s.values == b.s.values).all()
    d = a[['cc', 'aid', 's', 'y', 'w', 'prov']].copy()
    for m in ('free', 'hsc'):
        d[m] = b[m].values
    return d


def wauc_if(score, y, w):
    """가중 AUC와 행별 선형화 기여 u_i (A* - A ≈ Σ (W_i - 1) u_i)."""
    y = y.astype(bool)
    u, g = np.unique(score, return_inverse=True)
    ng = len(u)
    wp = np.bincount(g[y], weights=w[y], minlength=ng)
    wn = np.bincount(g[~y], weights=w[~y], minlength=ng)
    P, N = wp.sum(), wn.sum()
    below_n = np.cumsum(wn) - wn            # 음성 가중 중 점수 < 그룹
    above_p = wp.sum() - np.cumsum(wp)      # 양성 가중 중 점수 > 그룹
    A = float((wp * (below_n + 0.5 * wn)).sum() / (P * N))
    Fpos = (below_n + 0.5 * wn) / N          # 양성 i: 음성 중 아래 비율
    Gneg = (above_p + 0.5 * wp) / P          # 음성 j: 양성 중 위 비율
    ui = np.where(y, w * (Fpos[g] - A) / P, w * (Gneg[g] - A) / N)
    return A, ui


res = {'struct': {}, 'single': {}, 'note': 'u = 선형화 기여, V_k = Σ_군집 (Σ u)^2. 곱 Poisson 가중 분산(해석) = V_f+V_s+V_c, CGM = V_f+V_s-V_c.'}
fr = []
for v in YS:
    d = load(v)
    d['fk'] = d.cc.astype(str) + ':' + d.aid.astype(str)
    d['ck'] = d.fk + ':' + d.s.astype(str)
    # 구조
    nsc = d.groupby('fk').s.nunique(); nfi = d.groupby('s').fk.nunique()
    dup = int(d.duplicated(['fk', 's']).sum())
    pos = d[d.y == 1]
    # 양성이 있는 기업 중 양성 서브클래스 2개 이상인 기업 비율, 양성 서브클래스 중 양성 기업 2개 이상 비율
    pf = pos.groupby('fk').s.nunique(); ps = pos.groupby('s').fk.nunique()
    res['struct'][v] = {'n': len(d), 'npos': int(d.y.sum()), 'nfirm': int(d.fk.nunique()), 'nsc': int(d.s.nunique()), 'dup_cells': dup,
                        'q_f': float((d.fk.map(nsc) >= 2).mean()), 'q_s': float((d.s.map(nfi) >= 2).mean()),
                        'sc_per_firm_med': float(nsc.median()), 'firm_per_sc_med': float(nfi.median()),
                        'firm_per_sc_min': int(nfi.min()), 'sc_per_firm_min': int(nsc.min()),
                        'frac_firm_with_1sc': float((nsc == 1).mean()),
                        'pos_firm_multi_sc_frac': float((pf >= 2).mean()), 'pos_sc_multi_firm_frac': float((ps >= 2).mean()),
                        'w_unique_by_y': {str(k): int(x) for k, x in d.groupby('y').w.nunique().items()}}
    # 영향함수
    y = d.y.values; w = d.w.values.astype(np.float64)
    A = {}; U = {}
    for m in ('prov', 'free', 'hsc'):
        A[m], U[m] = wauc_if(d[m].values.astype(np.float64), y, w)
    best = 'free' if A['free'] >= A['hsc'] else 'hsc'
    u_d = U['prov'] - U[best]
    # Δskill min 의 활성 가지
    ds = {m: (A['prov'] - A[m]) / (1 - A[m]) for m in ('free', 'hsc')}
    act = min(ds, key=ds.get)
    # d/dA_prov = 1/(1-A_m), d/dA_m = -(1-A_prov)/(1-A_m)^2
    u_s = U['prov'] / (1 - A[act]) - U[act] * (1 - A['prov']) / (1 - A[act]) ** 2
    d['u_d'] = u_d; d['u_s'] = u_s
    out = {}
    for nm in ('u_d', 'u_s'):
        Vf = float((d.groupby('fk')[nm].sum() ** 2).sum())
        Vs = float((d.groupby('s')[nm].sum() ** 2).sum())
        Vc = float((d.groupby('ck')[nm].sum() ** 2).sum())
        Vi = float((d[nm] ** 2).sum())
        out[nm] = {'SE_f': Vf ** .5, 'SE_s': Vs ** .5, 'SE_c': Vc ** .5, 'SE_iid': Vi ** .5,
                   'SE_CGM': max(Vf + Vs - Vc, 0) ** .5, 'SE_prod_analytic': (Vf + Vs + Vc) ** .5,
                   'ratio_prod_CGM': ((Vf + Vs + Vc) / (Vf + Vs - Vc)) ** .5,
                   'excess_over_Vc': ((Vf + Vs + Vc) - (Vf + Vs - Vc)) / Vc,
                   'Vc_share_of_CGM': Vc / (Vf + Vs - Vc)}
    res['single'][v] = {'AUC': A, 'best': best, 'dAUC': A['prov'] - A[best], 'dskill_act': act, 'dskill': ds[act], **out}
    fr.append(d[['fk', 's', 'ck', 'u_s', 'u_d', 'y', 'w', 'prov', 'free', 'hsc']].assign(v=v))
    print(v, 'done', round(time.time() - t0, 1), json.dumps(res['single'][v]['u_d'], ensure_ascii=False), flush=True)

# 5원점 평균 D: 군집(기업·서브클래스·칸)이 원점 간 공유
allf = pd.concat(fr, ignore_index=True)
allf['u_m'] = allf.u_s / 5.0
allf['u_dm'] = allf.u_d / 5.0
m5 = {}
for nm in ('u_m', 'u_dm'):
    Vf = float((allf.groupby('fk')[nm].sum() ** 2).sum())
    Vs = float((allf.groupby('s')[nm].sum() ** 2).sum())
    Vc = float((allf.groupby('ck')[nm].sum() ** 2).sum())
    m5[nm] = {'SE_f': Vf ** .5, 'SE_s': Vs ** .5, 'SE_c': Vc ** .5, 'SE_CGM': max(Vf + Vs - Vc, 0) ** .5,
              'SE_prod_analytic': (Vf + Vs + Vc) ** .5, 'ratio_prod_CGM': ((Vf + Vs + Vc) / (Vf + Vs - Vc)) ** .5,
              'Vc_over_CGM': Vc / (Vf + Vs - Vc)}
res['mean5'] = {'D_point': float(np.mean([res['single'][v]['dskill'] for v in YS])), 'D': m5['u_m'], 'dAUC': m5['u_dm']}
print('mean5', json.dumps(res['mean5'], ensure_ascii=False), flush=True)

# (5) 곱 가중 몬테카를로로 해석식 검증 — 원점 2024 단일 ΔAUC, B=300, 새 시드
d = fr[-1]
fidx = pd.Index(d.fk.unique()); sidx = pd.Index(d.s.unique())
fi = fidx.get_indexer(d.fk); si = sidx.get_indexer(d.s)
rng = np.random.default_rng(99017)
y = d.y.values.astype(bool); w = d.w.values.astype(np.float64)


def auc_w(score, ww):
    u, g = np.unique(score, return_inverse=True)
    wp = np.bincount(g[y], weights=ww[y], minlength=len(u)); wn = np.bincount(g[~y], weights=ww[~y], minlength=len(u))
    return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


gs = {m: np.unique(d[m].values, return_inverse=True)[1] for m in ('prov', 'free', 'hsc')}


def auc_g(m, ww):
    g = gs[m]; ng = g.max() + 1
    wp = np.bincount(g[y], weights=ww[y], minlength=ng); wn = np.bincount(g[~y], weights=ww[~y], minlength=ng)
    return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


mc = {'prod': [], 'firm': [], 'sc': []}
for b in range(300):
    WF = rng.poisson(1.0, len(fidx)); WS = rng.poisson(1.0, len(sidx))
    for k, g in (('prod', WF[fi] * WS[si]), ('firm', WF[fi]), ('sc', WS[si])):
        ww = w * g
        a = {m: auc_g(m, ww) for m in ('prov', 'free', 'hsc')}
        mc[k].append(a['prov'] - max(a['free'], a['hsc']))
    if time.time() - t0 > 500:
        break
res['mc_2024'] = {k: float(np.std(x, ddof=1)) for k, x in mc.items()}
res['mc_2024']['B'] = len(mc['prod'])
print('mc', res['mc_2024'], flush=True)
res['secs'] = round(time.time() - t0, 1)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('done', res['secs'])
