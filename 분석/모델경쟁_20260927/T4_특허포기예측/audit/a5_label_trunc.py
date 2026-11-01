# 감사 5: 학습 레이블·위험모형 결정행이 C(=2014-01-01) 이후 자료 없이도 똑같이 만들어지는지(절단 불변성)
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import t4lib as T, hazard as HZ
from run import labels
C = '2014-01-01'; Cd = np.datetime64(C)
ref = {L: labels(L, T.features(L).pid.values, C) for L in ['2005-01-01', '2009-07-01']}
refd = HZ.train_decisions('2012-01-01', C)
D = T.load(); TR = dict(D)
TR['pay'] = D['pay'][D['pay'].pd <= Cd]; TR['lap'] = D['lap'][D['lap'].cd <= Cd]
TR['ev'] = {k: v[v.cd <= Cd] for k, v in D['ev'].items()}
T._D.clear(); T._D.update(TR)
for L, y in ref.items():
    y2 = T.label_asof(L, T.features(L).pid.values, C)
    print(L, '학습레이블 불일치', int((~np.isclose(y, y2, equal_nan=True)).sum()), '행', len(y), '양성', int(np.nansum(y)), '미확정', int(np.isnan(y).sum()))
T.CACHE = os.path.join(HERE, 'audit'); fn = os.path.join(T.CACHE, 'dec_20120101_20140101.parquet')
d2 = HZ.train_decisions('2012-01-01', C)
print('위험모형 결정행', len(refd), len(d2), '불일치 y', int((refd.y.values != d2.y.values).sum()) if len(refd) == len(d2) else 'len diff')
os.remove(fn)
