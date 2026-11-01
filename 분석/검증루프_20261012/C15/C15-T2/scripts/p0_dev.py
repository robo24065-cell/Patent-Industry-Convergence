# T1 개발 경로 P0: b0 o2008 재학습 예측 = FROZEN_dev o2008 (px·pl 최대 차)
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import numpy as np, pandas as pd
c = int(sys.argv[1]) if len(sys.argv) > 1 else 2008
a = pd.read_parquet(rd(0, c)/f'o{c}.parquet'); b = pd.read_parquet(R3/'runs'/'FROZEN_dev'/f'o{c}.parquet')
out = dict(c=c, rows_equal=bool(len(a) == len(b) and (a.firm.values == b.firm.values).all()), px_maxdiff=float(np.abs(a.px.values - b.px.values).max()), pl_maxdiff=float(np.abs(a.pl.values - b.pl.values).max()))
json.dump(out, open(CT1/f'p0_dev_o{c}.json', 'w', encoding='utf-8'), indent=1); print(out)
