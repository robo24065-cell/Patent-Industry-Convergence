# verify_1: C12 캐시 u_{cc}_{Y}에서 SCAGG 맥락 열이 제공 열과 같은지(=제공 KSIC 의존) 확인. 읽기만
import sys, json, numpy as np, pandas as pd
C12C = sys.argv[1]; OUT = sys.argv[2]
CTX = ['dot', 'cos', 'adopt', 'entry', 'qmain', 'mom_c', 'sind_c', 'mom_kr', 'sind_kr', 'mom_5', 'sind_5']
R = {}
for cc in ['CN', 'EP']:
    for Y in [2020, 2024]:
        cols = [f'{a}_{x}' for a in ['prov', 'scagg', 'free', 'hsc'] for x in CTX]
        u = pd.read_parquet(C12C + rf'\u_{cc}_{Y}.parquet', columns=cols)
        r = {}
        for x in CTX:
            p = u[f'prov_{x}'].values.astype(float); s = u[f'scagg_{x}'].values.astype(float); f = u[f'free_{x}'].values.astype(float)
            ok = ~(np.isnan(p) | np.isnan(s))
            r[x] = {'identical_share': float(np.mean(np.isclose(p[ok], s[ok], rtol=1e-5, atol=1e-7))),
                    'corr_prov_scagg': float(np.corrcoef(p[ok], s[ok])[0, 1]) if p[ok].std() > 0 and s[ok].std() > 0 else None,
                    'corr_prov_free': float(np.corrcoef(p[ok & ~np.isnan(f)], f[ok & ~np.isnan(f)])[0, 1]) if np.nanstd(f) > 0 else None}
        R[f'{cc}_{Y}'] = r
        print(cc, Y, {x: (round(v['identical_share'], 3), None if v['corr_prov_scagg'] is None else round(v['corr_prov_scagg'], 3)) for x, v in r.items()}, flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
