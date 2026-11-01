"""T3 3라운드 최종시험 전 감사(D단계) — 절단 캐시 대조.
공개연도 ≤2016 문서만으로 새로 만든 절단 캐시(audit_trunc2016)의 t0=2012..2016 셀 피처(feats)·feat2 가
개발 캐시(cache_pub, 공개연도 ≤2021)의 같은 원점 값과 같은지 본다.
같으면 → t0 피처가 t0+1..2021 공개분에 전혀 의존하지 않는다(원점 이후 공개분 혼입 없음, '최초 등장' 계열 포함).
(앞선 audit_prefinal_r3.py 는 2021 vs 2024 절단만 대조 → 원점 직후 1–3년 혼입은 이 스크립트로 추가 확인)
레이블 열(kr_fut, y)은 대조에서 뺀다(t0=2012·2013은 절단 캐시에서도 결과창 ≤2016이라 계산되지만 비교 대상 아님).
출력: audit_trunc_r3d.json
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.update(T3R3_VAR='pub', T3R3_CACHE='audit_trunc2016', T3R3_MAXYEAR='2016', T3R3_LABEL_MAX='2016')
os.environ.pop('T3R3_FINAL_UNLOCK', None)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import r3lib as L
P = L.load(); out = {}
LAB = {'kr_fut', 'y'}
for t0 in range(2012, 2017):
    a = pd.read_parquet(os.path.join(HERE, 'cache_pub', f'feat_{t0}.parquet'))
    b = L.feats(P, t0, 10); b = b[[c for c in b.columns if c not in LAB]]
    cols = [c for c in a.columns if c not in LAB]
    assert set(cols) == set(b.columns), set(cols) ^ set(b.columns)
    m = a[cols].merge(b, on=['cell', 'ksic', 'mg', 't0'], how='outer', suffixes=('_d', '_t'), indicator=True)
    mb = m[m._merge == 'both']; num = [c for c in cols if c not in ('cell', 'ksic', 'mg', 't0')]
    diffs = {c: float(np.nan_to_num(np.nanmax(np.r_[0.0, np.abs(mb[c + '_d'].astype(float).values - mb[c + '_t'].astype(float).values)]))) for c in num}
    nanmis = int(sum(((mb[c + '_d'].isna()) != (mb[c + '_t'].isna())).sum() for c in num))
    out[f'feats_{t0}'] = dict(n_dev=len(a), n_trunc=len(b), merge={k: int(v) for k, v in m._merge.value_counts().items()},
                              maxdiff=max(diffs.values()), worst=max(diffs, key=diffs.get), nan_mismatch=nanmis)
    print(t0, out[f'feats_{t0}'], flush=True)
f2d = pd.read_parquet(os.path.join(HERE, 'cache_pub', 'feat2.parquet')); f2t = pd.read_parquet(os.path.join(HERE, 'audit_trunc2016', 'feat2.parquet'))
f2d = f2d[(f2d.t0 >= 2012) & (f2d.t0 <= 2016)]; f2t = f2t[(f2t.t0 >= 2012) & (f2t.t0 <= 2016)]
m = f2d.merge(f2t, on=['ksic', 'mg', 't0'], how='outer', suffixes=('_d', '_t'), indicator=True)
num = [c for c in f2d.columns if c not in ('ksic', 'mg', 't0', 'sub')]; mb = m[m._merge == 'both']
out['feat2_2012_2016'] = dict(n_dev=len(f2d), n_trunc=len(f2t), merge={k: int(v) for k, v in m._merge.value_counts().items()},
                              maxdiff=max(float(np.nanmax(np.abs(mb[c + '_d'] - mb[c + '_t']))) for c in num))
print('feat2', out['feat2_2012_2016'], flush=True)
ok = all(v['maxdiff'] < 1e-9 and v['nan_mismatch'] == 0 and v['merge'].get('left_only', 0) == 0 and v['merge'].get('right_only', 0) == 0
         for k, v in out.items() if k.startswith('feats_')) and out['feat2_2012_2016']['maxdiff'] < 1e-9 \
     and out['feat2_2012_2016']['merge'].get('left_only', 0) == 0 and out['feat2_2012_2016']['merge'].get('right_only', 0) == 0
out['all_identical'] = bool(ok)
json.dump(out, open(os.path.join(HERE, 'audit_trunc_r3d.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ALL IDENTICAL:', ok)
