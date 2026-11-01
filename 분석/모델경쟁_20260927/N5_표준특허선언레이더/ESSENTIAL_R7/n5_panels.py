# -*- coding: utf-8 -*-
"""R7 N5: 개발 원점(2012–2020) 패널 생성. --pop orig | conc | concm | hyb_conc | hyb_concm
  orig      : 동결 n5lib.build_panel (26429) — 재현 기준. hybrid_panel 이 같은 값을 내는지 함께 검증(validate_hybrid.json)
  conc/concm: 연계표 C2640 모집단 전체를 동결 build_panel 로(단위·선언률·제목 풀·해외 패밀리 필터 모두 연계표 기준) = 과제 재정의
  hyb_*     : 단위·레이블 = 제공 26429 그대로, 모집단 통계(선언률·제목 풀)와 해외 패밀리 레코드 필터는 연계표 C2640 = 결과 고정 대체"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import n5_common as X
N = X.N
ap = argparse.ArgumentParser(); ap.add_argument('--pop', required=True); a = ap.parse_args()
T0S = range(2012, 2021); t = time.time()
k0 = N.load_kr('26429'); ex0 = N.load_family(k0, '26429'); E0 = N.load_ext('26429')
out = os.path.join(X.CACHE, f'panel_{a.pop}.parquet')
if a.pop == 'orig':
    D = pd.concat([N.build_panel(t0, '26429', 'dev', k0, ex0, E0) for t0 in T0S], ignore_index=True)
    FE = json.load(open(os.path.join(X.N5, 'final_spec_r4.json'), encoding='utf-8'))['champion_features']
    val = {}
    for t0 in (2014, 2020):
        H = X.hybrid_panel(t0, k0, k0, ex0, E0).set_index('appl_no'); B = D[D.t0 == t0].set_index('appl_no')
        assert set(H.index) == set(B.index) and (H.y.reindex(B.index) == B.y).all()
        val[t0] = {f: float(np.nanmax(np.abs(H[f].reindex(B.index).astype(float).values - B[f].astype(float).values))) if B[f].notna().any() else 0.0 for f in FE}
    mx = max(max(v.values()) for v in val.values())
    json.dump(dict(max_abs_diff=mx, by_t0=val), open(os.path.join(HERE, 'results', 'validate_hybrid.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('hybrid_panel 대조 최대차', mx, flush=True); assert mx < 1e-5
elif a.pop in ('conc', 'concm'):
    k = X.load_kr_pop(a.pop); ex = X.load_family_pop(k, a.pop, a.pop); E = X.load_ext_pop(k, E0, a.pop)
    print('모집단', a.pop, len(k), '| 26429 겹침', int(k.appl_no.isin(set(k0.appl_no)).sum()), '/', len(k0), flush=True)
    D = pd.concat([N.build_panel(t0, X.CODE, 'dev', k, ex, E) for t0 in T0S], ignore_index=True)
else:
    pop = a.pop.split('_')[1]
    ks = X.load_kr_pop(pop); exh = X.load_family_pop(k0, pop, f'orig_units_{pop}')
    D = pd.concat([X.hybrid_panel(t0, k0, ks, exh, E0) for t0 in T0S], ignore_index=True)
D.to_parquet(out, index=False)
print(a.pop, D.groupby('t0').agg(n=('y', 'size'), pos=('y', 'sum')).T.to_string(), round(time.time() - t), flush=True)
