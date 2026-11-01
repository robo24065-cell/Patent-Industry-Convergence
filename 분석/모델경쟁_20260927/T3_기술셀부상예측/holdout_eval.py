"""[2단계 이후 대체됨] 이 스크립트 대신 final_run.py --mode holdout 을 사용한다 (동결 모형 v30, final_spec.json).
🔒 3단계 평가자 전용 — 1·2단계에서 실행 금지(실행하지 않았음).
동결 모형 v22 로 홀드아웃 t0=2018 (성장 2019–2021 vs 2016–2018) 을 단 한 번 평가.
실행 순서 (PowerShell 예):
  $env:T3_HOLDOUT_UNLOCK='1'; $env:T3_CACHE='cache_holdout'; $env:T3_MAXYEAR='2021'; $env:T3_LABEL_MAX='2021'; $env:T3_T0MAX='2018'
  python build_base.py; python build_panel.py; python build_feat2.py; python holdout_eval.py
학습 레이블: t0 ∈ 2005..2015 (결과창 ≤2018, 홀드아웃 기준시점 이전 완전 관측). 적격: 5개국 t0-2..t0 합 ≥50.
"""
import os
assert os.environ.get('T3_HOLDOUT_UNLOCK')=='1' and os.environ.get('T3_CACHE')=='cache_holdout'
from models import *
from sklearn.metrics import roc_auc_score
H.TRAIN_START=2005
tr=enrich(L.add_labels(pd.concat([L.get_feats(t) for t in range(2005,2016)],ignore_index=True),H.MIN_TOTAL))
te=enrich(L.add_labels(L.get_feats(2018),H.MIN_TOTAL))
assert tr.t0.max()+3<=2018
p=best_fp()(tr,te)
def base_fp(tr,te):
    m=LogisticRegression(C=1e6,max_iter=1000).fit(tr[['kr_mom']].values,tr.yb.values); return m.predict_proba(te[['kr_mom']].values)[:,1]
p0=base_fp(tr,te)
m=L.evaluate(te.y.values,p,te.yb.values); m0=L.evaluate(te.y.values,p0,te.yb.values)
print('model',m); print('baseline',m0); print('skill',(m['auc']-m0['auc'])/(1-m0['auc']))
rng=np.random.default_rng(0); mg=te.mg.values; ums=np.unique(mg); gi={u:np.where(mg==u)[0] for u in ums}; bs=[]
for b in range(1000):
    ii=np.concatenate([gi[u] for u in rng.choice(ums,len(ums))]); y=te.yb.values[ii]
    a=roc_auc_score(y,p[ii]); a0=roc_auc_score(y,p0[ii]); bs.append((a,a0,(a-a0)/(1-a0)))
bs=np.array(bs); print('95% CI (IPC 메인그룹 군집 부트스트랩) AUC',np.percentile(bs[:,0],[2.5,97.5]),'base',np.percentile(bs[:,1],[2.5,97.5]),'skill',np.percentile(bs[:,2],[2.5,97.5]))
pd.DataFrame({'cell':te.cell,'p':p,'p_base':p0,'y':te.y,'yb':te.yb}).to_parquet('preds/HOLDOUT_t0_2018.parquet')
