# -*- coding: utf-8 -*-
"""s6: facts C-T3-drop_ros-* 재현(저장 예측에서 독립 재채점, 9원점 평균 AUC와 짝 ΔAUC). 그리고 RoS 연결 수정판(V1·V2) 개발 AUC 와 full 의 차이."""
import json
from common import *
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
R7 = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/results"; C5 = MC + "/_shared_report/fusion_advantage/c5_R10/t3_results"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3/res/t3"
M = {"full": f"{R7}/pub_full_preds.parquet", "drop_ros": f"{R7}/pub_drop_ros_preds.parquet", "kr_rebuild": f"{R7}/pub_kr_rebuild_preds.parquet",
     "hyb_txt": f"{C5}/hyb_txt_preds.parquet", "hyb_free": f"{C5}/hyb_free_preds.parquet", "hyb_txt_reg": f"{C10}/hyb_txt_reg_preds.parquet", "hyb_free_reg": f"{C10}/hyb_free_reg_preds.parquet"}
P = {k: pd.read_parquet(v) for k, v in M.items()}
for V in ('S0', 'V1', 'V2', 'V3'):
    P['C11_' + V] = pd.concat([pd.read_parquet(f"{W}/cache/preds_{V}_dev_{s}.parquet") for s in ('2010-2011-2012-2013-2014', '2015-2016-2017-2018')])
def by_origin(df):
    return {int(t): roc_auc_score(g.yb, g.p) for t, g in df.groupby('t0')}
A = {k: by_origin(v) for k, v in P.items()}
R = {'mean_auc': {k: round(float(np.mean(list(v.values()))), 5) for k, v in A.items()}}
def pair(a, b):
    d = [A[a][t] - A[b][t] for t in sorted(A[a])]; return dict(dAUC=round(float(np.mean(d)), 5), wins=f"{sum(x > 0 for x in d)}/{len(d)}")
for a, b in (("drop_ros", "kr_rebuild"), ("drop_ros", "hyb_txt"), ("drop_ros", "hyb_free"), ("drop_ros", "hyb_txt_reg"), ("drop_ros", "hyb_free_reg"),
             ("full", "drop_ros"), ("C11_S0", "full"), ("C11_V2", "drop_ros"), ("C11_V1", "drop_ros"), ("C11_V3", "drop_ros"), ("C11_S0", "C11_V3")):
    R[f"{a}-{b}"] = pair(a, b)
# full 예측 = C11 S0 dev 예측 동일 여부
m = P['full'].merge(P['C11_S0'][['cell', 't0', 'p']], on=['cell', 't0'], suffixes=('', '_c11'))
R['full_vs_C11_S0_max_dp'] = float((m.p - m.p_c11).abs().max()) if len(m) else None; R['full_vs_C11_S0_n'] = len(m)
print(json.dumps(R, ensure_ascii=False, indent=1))
json.dump(R, open(W + "/res/s6_dropros_repro.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
