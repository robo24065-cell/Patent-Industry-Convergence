# 3단계 보조: 검증(dryrun, c=2009)에서도 stage3_verify.py와 같은 설정의 XGB 기준선(같은 모형족)을 적합해, 홀드아웃과 같은 '최강 기준선' 정의로 검증 skill을 다시 잰다.
import sys, json, pathlib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding='utf-8')
HERE = pathlib.Path(__file__).resolve().parent; FC = HERE/'final_cache'/'dryrun'
BASE = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart']
e = pd.read_parquet(FC/'eval_scores.parquet'); r = json.load(open(HERE/'dryrun_result.json', encoding='utf-8'))
btr = pd.concat([pd.read_parquet(FC/f'base_app_c{c}.parquet') for c in r['config']['train_cuts']], ignore_index=True)
bte = pd.read_parquet(FC/f'base_app_c{r["config"]["eval_cut"]}.parquet'); assert (bte.y.values == e.y.values).all()
xb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
xb.fit(btr[BASE + ['ind2']].values.astype(np.float32), btr.y.values); s = xb.predict_proba(bte[BASE + ['ind2']].values.astype(np.float32))[:, 1]
y = e.y.values; nc = e.newc.values.astype(bool)
m = dict(auc_all=roc_auc_score(y, s), pr_all=average_precision_score(y, s), auc_new=roc_auc_score(y[nc], s[nc]), pr_new=average_precision_score(y[nc], s[nc]))
fx = r['baseline_fixed_strengthened']; mm = r['model']
best = {k: max(m[k], fx[k]) for k in ('auc_all', 'auc_new')}
out = dict(val_xgb_baseline=m, val_logit_fixed={k: fx[k] for k in m}, val_strongest=best,
           val_skill_all_vs_strongest=(mm['auc_all'] - best['auc_all']) / (1 - best['auc_all']), val_skill_new_vs_strongest=(mm['auc_new'] - best['auc_new']) / (1 - best['auc_new']))
json.dump(out, open(HERE/'stage3_val_xgbbase.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1))
