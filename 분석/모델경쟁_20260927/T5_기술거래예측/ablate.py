# v5 설정에서 피처 그룹 하나씩 제거(ablation) → 검증 AUC. python ablate.py OWN_HIST PAY ...
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import run
from evalkit import metrics
from common import *
tr = run.load(range(2006, TRAIN_MAX_L + 1)); va = run.load(VAL_L); tr['ytr'] = tr.y_MAIN
G = dict(OWN_HIST=run.OWN_HIST, PAY=run.PAY, STATIC=run.STATIC, EXTRA=run.EXTRA, OWN_STATUS=run.OWN_STATUS + ['cur_type'],
         KSIC=['ksic_c', 'ksic_rate3', 'ksic_rate1'], CORP=['own_corp3', 'prior_CORP'], OWNSALE=['own_out3', 'own_out3_days', 'own_out_all', 'own_out_rate', 'own_out1', 'own_out6m'],
         LAPSE=['own_lapse3', 'own_lapse1', 'own_lapse_rate', 'own_lapse_trend'], APPMOM=['own_app3', 'own_app_prev', 'own_app_mom', 'own_years'])
for gname in sys.argv[1:]:
    F = [f for f in run.FULL if f not in G[gname]]
    p, _ = run.fit_predict(dict(model='xgb', num=F, cat=[]), tr, va)
    mt = metrics(va.y_MAIN, p)
    print('drop', gname, len(F), {k: round(x, 4) for k, x in mt.items()},
          [round(metrics(va.y_MAIN[va.L == L], p[(va.L == L).values])['auc'], 4) for L in VAL_L], flush=True)
