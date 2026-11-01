"""verify_0 준비: 최종 모드 S0 피처 행렬(D: 적격+t0 2005–2018·2021, W: 확대 10≤tot_W<50)을 파이프라인 순서 그대로 내보낸다.
(독립 검증 v0_indep.py 는 이 행렬의 비출원인 피처만 받아 쓰고, S1 피처 재계산·학습·채점은 모두 자체 코드로 한다.)"""
import os, sys, json
MY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(MY), 'scripts'))
from common import R3
os.environ.update(T3R3_VAR='pub', T3R3_CACHE='cache_pub_final', T3R3_MAXYEAR='2024', T3R3_LABEL_MAX='2024', T3R3_FINAL_UNLOCK='1',
                  T3R3_TRAIN_MAX_T0='2018', T3R3_EXT_SUFFIX='_final', T3R3_EXTB_SUFFIX='_final')
import numpy as np, pandas as pd
import c09lib as L, c09core as C
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8')); CFG = SPEC['cfg']
def prep(D):
    cf = pd.read_parquet(os.path.join(R3, 'extB_cell_pub_final.parquet')); kf = pd.read_parquet(os.path.join(R3, 'extB_ksic_pub_final.parquet'))
    n0 = len(D); D = D.merge(cf, on=['cell', 't0'], how='left').merge(kf, on=['ksic', 't0'], how='left'); assert len(D) == n0
    return D
need = sorted({f for c in CFG['components'] for f in c['features']})
keep = ['cell', 'ksic', 'mg', 't0', 'y', 'yb', 'kr_fut', 'kr_W', 'tot_W'] + need
D = prep(C.get_D(t0s=list(range(2005, 2019)) + [2021]))
W = C._gw(D, 10) if hasattr(C, '_gw') else C.get_wide(D, 10)
W = prep(W)
D[keep].to_parquet(os.path.join(MY, 'v0_D_final.parquet')); W[[c for c in keep if c in W]].to_parquet(os.path.join(MY, 'v0_W_final.parquet'))
print(len(D), len(W), D.t0.min(), D.t0.max(), W.t0.min(), W.t0.max())
