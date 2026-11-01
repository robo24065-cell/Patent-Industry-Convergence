# 2단계 감사 1: v31 재현(리더보드 기록 없이) + 순위 피처 기준집단 점검
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from t6lib import *
from versions import VERSIONS, COMPACT, SECT
F = load_panel(with_patent=True)
print('panel', F.shape, 'years', F.year.min(), F.year.max(), 'ksic', F.ksic.nunique(), 'in50', F[F.in50==1].ksic.nunique())
V = VERSIONS['v31']
p = rolling_eval(F, V['fp'], train_filter=V['train_filter'])
old = pd.read_parquet('preds/v31.parquet')
m = p.merge(old, on=['ksic','year'], suffixes=('', '_old'))
print('재현 MAE', metrics(p.y, p.pred)['MAE'], 'old', metrics(old.y, old.pred)['MAE'], 'max|diff|', (m.pred-m.pred_old).abs().max())
b = pd.read_parquet('preds/baseline.parquet')
print('기준선 MAE', np.abs(b.y).mean(), 'n', len(b))
# 테스트 순위 기준집단: te는 50개(y 관측)만 → 학습(305개) 순위와 분포 불일치 여부
T = 2016
te = F[(F.year == T) & (F.in50 == 1) & F.y.notna()]
ref = SECT(F[F.year == T])
r_te = te[COMPACT].rank(pct=True)
r_ref = ref[COMPACT].rank(pct=True).loc[te.index]
print('2016 50개 내 순위 vs 305개 기준 순위 평균|차이|:', (r_te - r_ref).abs().mean().round(3).to_dict())
print('in50 산업의 SECT 소속 여부', F[F.in50==1].k2.isin(['15','20','21','23','24','25','26','27','28','29','30','31','33']).all())
