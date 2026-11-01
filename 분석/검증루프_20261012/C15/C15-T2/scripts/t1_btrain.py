# -*- coding: utf-8 -*-
"""T1 묶음 2·3단계.
 2단계(로지스틱): 동결 train_origin()의 'fr.logit_fit(tr[cols+['y']], cols) → logit_pred(200,000행씩) → float32 저장'과 같은 수치 계산을
   중간 복사본 없이 한다(같은 식 tr[cols].fillna(0).values → StandardScaler → LogisticRegression(max_iter=3000)). 메모리 4GB 규칙 때문(동결 단독 3.85GB).
   값 동일성은 P0(FROZEN_final pl 차 0)로 확인.
 3단계(XGB 시드별 → o{c}.parquet): 동결 final_run_r3.train_origin() 그대로 호출. ec['seeds']만 묶음 값.
 python t1_btrain.py <c> <b>   (종료코드 3 = 단계 종료, run.sh가 재실행). 1단계 파일(s1_w)이 이미 있어야 한다(t1_bprep)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import gc
import numpy as np, pandas as pd
import final_run_r3 as F
from r2 import fr

c = int(sys.argv[1]); b = int(sys.argv[2]); S = seeds(b); RD = rd(b, c)
import time as _t
while (rd(b, c)/'LANE2').exists() and len(sys.argv) < 4: _t.sleep(15)   # 두 번째 차선이 같은 묶음을 처리 중이면 끝날 때까지 대기(중복 실행 방지)
if (RD/f'o{c}.parquet').exists():
    print('done', c, b); sys.exit(0)
assert (RD/f'o{c}_s1_w.npy').exists(), 'bprep 먼저'
ec = json.loads(json.dumps(F.EC)); ec['ext'] = list(ec['ext']); ec['seeds'] = list(S['xgb']); cols = list(ec['cols'])
print(f'[C15-T2] b={b} c={c} seeds={ec["seeds"]} RD={RD} rss={G.rss_now()}', flush=True)
fl = RD/f'o{c}_pl.npy'
if not fl.exists():
    assert ec.get('blend_w', 0.9) < 1.0
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = pd.read_parquet(RD/f'o{c}_s1_tr.parquet')
    lcols = [x for x in cols if x != 'ind2']                      # fr.logit_fit 과 같은 열 규칙
    X = tr[lcols].fillna(0).values; yv = tr.y.values; del tr; gc.collect()
    sc = StandardScaler().fit(X); Xs = sc.transform(X); del X; gc.collect()
    print('logit X', Xs.shape, Xs.dtype, G.rss_now(), flush=True)
    lr = LogisticRegression(max_iter=3000).fit(Xs, yv); del Xs; gc.collect()
    L = dict(sc=sc, lr=lr, cols=lcols)
    te = pd.read_parquet(RD/f'o{c}_s1_te.parquet')
    np.save(fl, np.concatenate([fr.logit_pred(L, te.iloc[i:i + 200000]) for i in range(0, len(te), 200000)]).astype(np.float32))
    print('logit saved', G.rss_peak(), flush=True); sys.exit(3)
F.train_origin(c, fcp(c), PTR, RD, ec)
print('peak', G.rss_peak(), flush=True)
