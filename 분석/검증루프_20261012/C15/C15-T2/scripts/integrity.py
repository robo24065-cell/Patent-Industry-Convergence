# -*- coding: utf-8 -*-
"""9분 시간 초과(rc=124)로 재시작된 학습의 산출 무결성: 묶음별 px 시드 파일 크기·o{c}.px = 시드 평균·o{c}.pl = pl 파일, 행 키 동일. → cache/t1/integrity.json"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard15 as G
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from t1common import seeds, rd, CT1
out = {}
ref = None
for b in range(10):
    for c in ([2019] + ([2008, 2009, 2010, 2011, 2012] if b in (0, 1, 2, 3) else [])):
        if b == 0 and c != 2019 and c != 2008: continue
        RD = rd(b, c); o = pd.read_parquet(RD/f'o{c}.parquet')
        px = [np.load(RD/f'o{c}_px{s}.npy') for s in seeds(b)['xgb']]; pl = np.load(RD/f'o{c}_pl.npy')
        r = dict(n=len(o), px_files_ok=all(len(p) == len(o) for p in px), px_eq_mean=bool((o.px.values == np.mean(px, 0)).all()), pl_eq=bool((o.pl.values == pl).all()),
                 nan=int(np.isnan(o.px.values).sum() + np.isnan(o.pl.values).sum()), corr_px_seeds=float(np.corrcoef(px[0], px[1])[0, 1]))
        if c == 2019:
            if ref is None: ref = o[['firm', 'k']]
            r['keys_eq_b0'] = bool((o.firm.values == ref.firm.values).all() and (o.k.values == ref.k.values).all())
        out[f'b{b}_o{c}'] = r
out['all_ok'] = all(v['px_files_ok'] and v['px_eq_mean'] and v['pl_eq'] and v['nan'] == 0 and v.get('keys_eq_b0', True) for v in out.values() if isinstance(v, dict))
json.dump(out, open(CT1/'integrity.json', 'w', encoding='utf-8'), indent=1); print(out['all_ok']); print({k: v for k, v in out.items() if isinstance(v, dict) and not (v['px_eq_mean'] and v['pl_eq'])})
