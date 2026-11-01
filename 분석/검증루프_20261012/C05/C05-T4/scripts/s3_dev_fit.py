# s3: 개발 전진 평가 한 칸. 사용법: python s3_dev_fit.py <평가원점 YYYYMM> <모형,모형...>
# 학습 = S0 ≤ v−1년 개발 원점. 평가 원점 레이블은 개발 레이블(공개일 ≤ 2024-06-30)뿐이다.
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
from libM import *
T0 = time.time()
v = sys.argv[1]; models = sys.argv[2].split(',')
assert v in EVAL
tr_or = [s for s in DEV if int(s[:4]) <= int(v[:4]) - 1]
u = load_table(tr_or + [v])
assert (u.y >= 0).all(), 'lock rows must not be here'
train, test = add_te(u[u.s0.isin(tr_or)], u[u.s0 == v])
for mdl in models:
    p, pl, px, info = fit_predict(train, test, mdl)
    out = pd.DataFrame({'key': test.key.values, 's0': v, 'y': test.y.values, 'p': p, 'p_lr': pl, 'p_xgb': px})
    out.to_parquet(os.path.join(C, f'pred_{v}_{mdl}.parquet'), index=False)
    a = auc(out.y, out.p)
    json.dump(dict(v=v, model=mdl, train_origins=tr_or, n_train=int(len(train)), pos_train=int(train.y.sum()), n_test=int(len(test)),
                   pos_test=int(test.y.sum()), auc=a, auc_lr=auc(out.y, pl), auc_xgb=auc(out.y, px), **info),
              open(os.path.join(C, f'fit_{v}_{mdl}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(v, mdl, 'AUC', round(a, 4), 'lr', round(auc(out.y, pl), 4), 'xgb', round(auc(out.y, px), 4), round(time.time() - T0), flush=True)
