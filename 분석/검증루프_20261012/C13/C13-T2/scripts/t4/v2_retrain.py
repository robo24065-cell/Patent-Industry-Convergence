# -*- coding: utf-8 -*-
"""R5 레드팀(T4) 2단계: 동결 스펙(R2-E03) 구성요소를 학습 프레임(동결 runs_final 사본)에서 다시 학습해 동결 예측 npy와 대조.
r2lib 함수는 그대로 쓰되 출력 경로(odir)만 VERIFY_R5/F22로 돌린다(모형 저장 안 함). 사용: python v2_retrain.py base|logit|hgb|cls0|haz0|...
(전체 파이프라인 1회 실행은 962초·메모리 3.95GB라 9분 규칙 안에서 구성요소별로 나눠 돈다. 프레임 생성은 재실행하지 않는다.)"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ['T4R2_MODE'] = 'final'
os.environ.pop('T4R2_SAVE', None)
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TR, 'R2'))
import numpy as np
import r2lib as R
VD = os.path.join(HERE, 'F22')
R.odir = lambda o: VD
R.SAVE = False
FS = json.load(open(os.path.join(TR, 'R2', 'final_spec_r2.json'), encoding='utf-8')); SPEC = FS['spec']
arg = sys.argv[1]
t = time.time()
if arg in ('base', 'logit', 'hgb'):
    p = R.fit_predict_sk('F22', SPEC, arg)
    key = R.comp_key(SPEC, arg) if arg != 'base' else [f for f in os.listdir(os.path.join(VD, 'pred')) if f.startswith('base_')][0][:-4]
else:
    comp, k = arg[:3], int(arg[3:])
    fn = R.fit_predict_xgb_bag('F22', SPEC, comp, k)
    key = f"{R.comp_key(SPEC, comp)}_b{k}"
    p = np.load(fn)
frz = np.load(os.path.join(TR, 'R2', 'runs_final', 'F22', 'pred', key + '.npy'))
res = dict(comp=arg, key=key, n=len(p), max_abs_diff=float(np.abs(p - frz).max()), corr=float(np.corrcoef(p, frz)[0, 1]), secs=round(time.time() - t), peak_gb=R.peak_mem_gb())
print(res, flush=True)
fo = os.path.join(HERE, 'v2_retrain.json')
allr = json.load(open(fo, encoding='utf-8')) if os.path.exists(fo) else {}
allr[arg] = res
json.dump(allr, open(fo, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
