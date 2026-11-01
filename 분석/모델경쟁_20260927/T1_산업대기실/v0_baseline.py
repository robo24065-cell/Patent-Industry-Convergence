# v0: 데이터 점검 + 고정 기준선(규모 + density + DART 업종 일치) — 검증 구간만
import pandas as pd, numpy as np, time, pickle
from lib import *
t0 = time.time()
CAP_TR = '20091231'
tr, ctr = build_panel(2004, 5, cap=CAP_TR)
te, cte = build_panel(2009, 5)
tr = add_dart(tr, ctr); te = add_dart(te, cte)
for nm, d, c in [('train c=2004 (cap 2009-12-31)', tr, ctr), ('valid c=2009', te, cte)]:
    print(f'[{nm}] firms={len(c["firms"])} pairs={len(d)} y=1 {d.y.sum()} rate={d.y.mean():.4f} | newcomer firms={d[d.newc].firm.nunique()} pairs={d.newc.sum()} y=1 {d[d.newc].y.sum()} rate={d[d.newc].y.mean():.4f}')
    pk = d.groupby('k').y.sum(); print('   events per KSIC min/median/max', pk.min(), pk.median(), pk.max(), ' KSIC with >=10 events', (pk >= 10).sum())
    print('   DART matched firms', d[d.has_dart == 1].firm.nunique(), 'pairs', int(d.has_dart.sum()), 'y', int(d[d.has_dart == 1].y.sum()))
    print('   y unique', np.unique(d.y), 'nfut>=3 share among y', (d[d.y == 1].nfut >= 3).mean().round(3))
    tp = load()['fm'].set_index('applicant_cd').tp
    print('   type', d.drop_duplicates('firm').firm.map(tp).value_counts().head(4).to_dict())
BASE = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart', 'ind2']
m = fit_xgb(tr, BASE); s = predict(m, te, BASE)
met = metrics(te, s)
print('v0 XGB baseline', fmt(met))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
BL = ['lsize', 'lin', 'lout', 'inshare', 'dens', 'kpop', 'dm2', 'dm3', 'dm4', 'has_dart']
sc = StandardScaler().fit(tr[BL]); lr = LogisticRegression(max_iter=2000).fit(sc.transform(tr[BL]), tr.y)
sl = lr.predict_proba(sc.transform(te[BL]))[:, 1]; metl = metrics(te, sl)
print('v0 Logit baseline', fmt(metl))
# 기준선 = 두 모형족 중 더 강한 것(전체 AUC 기준)을 고정
best = met if met['auc_all'] >= metl['auc_all'] else metl
json.dump(dict(xgb=met, logit=metl, fixed=best, fixed_name='xgb' if best is met else 'logit'), open(T1/'baseline_v0.json', 'w'), indent=1)
te[['firm', 'k', 'y', 'newc']].assign(s_base=s).to_parquet(CACHE/'v0_scores.parquet', index=False)
lb_add('v0', '고정 기준선: 규모(log 특허수·범위안/밖)+density+kpop+DART 업종일치(2/3/4자리)', '업종코드·포트폴리오만으로 진입 예측(텍스트 없음)', best, best, 'Y',
       f"기준선 고정({'xgb' if best is met else 'logit'}); logit auc_all={metl['auc_all']:.4f} auc_new={metl['auc_new']:.4f}; xgb auc_all={met['auc_all']:.4f} auc_new={met['auc_new']:.4f}")
print('time', time.time() - t0)
