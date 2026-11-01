# v1: 파일럿 텍스트 노출(분류기 50-class 평균확률·고신뢰 건수) 추가 — 절단·자사출원 보정 후 첫 측정
import pandas as pd, numpy as np, time
from lib import *
t0 = time.time()
CAP_TR = '20091231'
tr, ctr = build_panel(2004, 5, cap=CAP_TR); te, cte = build_panel(2009, 5)
tr = add_dart(tr, ctr); te = add_dart(te, cte)
P = train_clf(CAP_TR, 'c2009_base')     # 평가원점 2009년 말까지 등록된 범위안 특허로 학습
print('clf ready', time.time() - t0, flush=True)
tr, *_ = text_feats(tr, ctr, P); te, *_ = text_feats(te, cte, P)
base = json.load(open(T1/'baseline_v0.json'))['fixed']
BASE = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart', 'ind2']
F1 = BASE + ['t_mean', 't_hi']
m = fit_xgb(tr, F1); s = predict(m, te, F1); met = metrics(te, s)
print('v1 xgb', fmt(met))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
L1 = [c for c in F1 if c != 'ind2']
sc = StandardScaler().fit(tr[L1]); lr = LogisticRegression(max_iter=3000).fit(sc.transform(tr[L1]), tr.y)
sl = lr.predict_proba(sc.transform(te[L1]))[:, 1]; metl = metrics(te, sl)
print('v1 logit', fmt(metl))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sl=sl).to_parquet(CACHE/'v1_scores.parquet', index=False)
lb_add('v1', '+텍스트 노출(범위밖 자사출원 제목→50-class 평균확률, 확률>0.5 건수) XGB', '범위밖 특허의 산업 신호가 첫 진입을 예고한다', met, base, 'Y',
       f"logit 변형 auc_all={metl['auc_all']:.4f} auc_new={metl['auc_new']:.4f} pr_new={metl['pr_new']:.4f}")
print('time', time.time() - t0)
