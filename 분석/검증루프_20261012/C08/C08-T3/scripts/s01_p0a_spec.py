# -*- coding: utf-8 -*-
# P0(a): 공식 챔피언(F22)과 동결 미래(FUT) spec 대조, 예측 파일 구조 확인
import json, hashlib, os, sys
import pandas as pd, numpy as np
R2 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T4_특허포기예측\R2"
def sha(p):
    h = hashlib.sha256(); h.update(open(p,'rb').read()); return h.hexdigest()
def flat(d, pre=''):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items(): o.update(flat(v, f'{pre}.{k}' if pre else k))
    else: o[pre] = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return o
out = {}
S = {n: json.load(open(os.path.join(R2, 'specs', f'FROZEN_{n}.json'), encoding='utf-8')) for n in ('dev', 'final', 'future')}
FS = json.load(open(os.path.join(R2, 'final_spec_r2.json'), encoding='utf-8'))
fs = flat(FS['spec'])
for n, s in S.items():
    f = flat(s['spec'])
    diff = {k: (fs.get(k), f.get(k)) for k in set(fs) | set(f) if fs.get(k) != f.get(k)}
    out[f'diff_final_spec_vs_FROZEN_{n}'] = diff
    out[f'FROZEN_{n}_meta'] = {k: v for k, v in s.items() if k != 'spec'}
out['final_spec_r2_meta'] = {k: v for k, v in FS.items() if k != 'spec'}
# spec 안 C/날짜 표기 찾기
def find(d, pre=''):
    r = []
    if isinstance(d, dict):
        for k, v in d.items(): r += find(v, f'{pre}.{k}')
    elif isinstance(d, list):
        for i, v in enumerate(d): r += find(v, f'{pre}[{i}]')
    elif isinstance(d, str) and ('2025' in d or '2026' in d or '2022' in d): r.append((pre, d))
    return r
out['date_strings_final_spec'] = find(FS)
out['date_strings_FROZEN_future'] = find(S['future'])
# 모형 파일 sha 비교(F22 vs FUT: 다른 학습자료라 달라야 정상, 키 이름은 같아야)
F22 = os.path.join(R2, 'runs_final', 'F22', 'models'); FUT = os.path.join(R2, 'runs_future', 'FUT', 'models')
out['model_keys_F22'] = sorted(os.listdir(F22)); out['model_keys_FUT'] = sorted(os.listdir(FUT))
out['model_keys_same'] = sorted(os.listdir(F22)) == sorted(os.listdir(FUT))
p = os.path.join(R2, 'runs_future', 'FUT', 'pred_20250701.parquet')
d = pd.read_parquet(p)
out['pred_20250701'] = dict(sha256=sha(p), mtime=str(pd.Timestamp(os.path.getmtime(p), unit='s')), rows=len(d), cols=list(d.columns),
                            dtypes={c: str(t) for c, t in d.dtypes.items()}, L_unique=list(map(str, d.L.unique())), pid_unique=int(d.pid.nunique()),
                            p_mean=float(d.p.mean()), y_realized_nonnull=int(d['y_realized'].notna().sum()) if 'y_realized' in d else None)
ev = pd.read_parquet(os.path.join(R2, 'runs_future', 'FUT', 'ev.parquet'), columns=['pid', 'L'])
out['ev_FUT'] = dict(rows=len(ev), L=list(map(str, ev.L.unique())), same_pid_order=bool((ev.pid.values == d.pid.values).all()) if len(ev) == len(d) else False)
for f in sorted(os.listdir(os.path.join(R2, 'runs_future', 'FUT', 'pred'))):
    a = np.load(os.path.join(R2, 'runs_future', 'FUT', 'pred', f)); out.setdefault('FUT_pred_npy', {})[f] = len(a)
json.dump(out, open(sys.argv[1], 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if not k.startswith('model_keys_F') }, ensure_ascii=False, indent=1, default=str)[:7000])
