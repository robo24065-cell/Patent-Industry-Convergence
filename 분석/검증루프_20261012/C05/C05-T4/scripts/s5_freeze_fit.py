# s5: 동결 학습(학습 원점 2015-12~2023-12 전부) → 잠금 S0=2024-12 위험집합 점수(레이블 없음). 사용법: python s5_freeze_fit.py <모형,...>
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
from libM import *
T0 = time.time()
models = sys.argv[1].split(',')
u = load_table(DEV + [LOCK])
tr = u[u.s0.isin(DEV)]; te = u[u.s0 == LOCK]
assert (tr.y >= 0).all() and (te.y == -1).all(), 'lock rows must carry no label'
te = te.drop(columns=['y']).assign(y=-1)
train, test = add_te(tr, te)
for mdl in models:
    p, pl, px, info = fit_predict(train, test, mdl)
    out = pd.DataFrame({'key': test.key.values, 's0': LOCK, 'p': p, 'p_lr': pl, 'p_xgb': px})
    out.to_parquet(os.path.join(C, f'lockscore_{mdl}.parquet'), index=False)
    json.dump(dict(model=mdl, train_origins=DEV, n_train=int(len(train)), pos_train=int(train.y.sum()), n_lock=int(len(test)), **info),
              open(os.path.join(C, f'freeze_fit_{mdl}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(mdl, 'n_lock', len(test), 'mean p', round(float(p.mean()), 6), round(time.time() - T0), flush=True)
