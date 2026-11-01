# 2단 계층 모형 확장판. python owner2.py <ver> key=val ...
#  bag=N (시드·설정 배깅 수), y0=2006 (학습 첫 L), aug=0/1 (L2012 2년창·L2013 1년창 권리자 매각률 추가, h 피처),
#  within=<pred_ver|none> (권리자 내 상대비에 쓸 특허점수; none이면 owner_hat 단독), beta=1.0 (상대비 지수), fset=A|B
#  apply=2014,2015 (적용 L; 홀드아웃 평가자는 apply=2019,2020,2021 & 학습 레이블 컷오프 변경 필요)
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import xgboost as xgb
import run
from owner_model_lib import owner_table, feats, ksic_global, GLOB
from common import *
from evalkit import *
ver = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
bag = int(kw.get('bag', 1)); y0 = int(kw.get('y0', 2006)); aug = int(kw.get('aug', 0)); within = kw.get('within', 'v12')
beta = float(kw.get('beta', 1.0)); attr = int(kw.get('attr', 0)); ATTR = ['natl_grp', 'chaebol', 'holding', 'ip_entity'] if attr else []; fset = kw.get('fset', 'A'); glob = int(kw.get('glob', 0)); dart = int(kw.get('dart', 0)); DA = ['firm_age', 'dart_hit'] if dart else []; GM = GLOB if glob else []; pwin = kw.get('pw', 'sqrt')
tr = run.load(range(y0, TRAIN_MAX_L + 1)); tr['ytr'] = tr.y_MAIN
OA = pd.read_parquet('cache/owner_attr.parquet').rename(columns={'nm': 'cur'})
if attr: tr = tr.merge(OA, on='cur', how='left')
if glob: tr = tr.merge(ksic_global(range(y0, TRAIN_MAX_L + 1)), on=['ksic_c', 'L'], how='left')
OD = pd.read_parquet('cache/owner_dart.parquet').rename(columns={'nm': 'cur'})
def add_dart(d):
    d = d.merge(OD[['cur', 'est']], on='cur', how='left')
    Lt = pd.to_datetime(d.L.astype(str) + '-01-01')
    d['firm_age'] = ((Lt - d.est).dt.days / 365.25).where(d.est < Lt)   # L 이후 설립이면 결측
    d['dart_hit'] = d.firm_age.notna().astype(int)
    return d.drop(columns=['est'])
if dart: tr = add_dart(tr)
ot = owner_table(tr, extra_first=ATTR + DA, extra_mean=GM); ot['h'] = 3
if aug:
    ad = run.horizon_aug(tr.iloc[:0], None)            # L2012·2013 (레이블 = 2014-01-01 전 사건만)
    ot = pd.concat([ot, owner_table(ad).assign(h=lambda x: 2014 - x.L)], ignore_index=True)
va = run.load(VAL_L); va['ytr'] = 0
if attr: va = va.merge(OA, on='cur', how='left')
if glob: va = va.merge(ksic_global(VAL_L), on=['ksic_c', 'L'], how='left')
if dart: va = add_dart(va)
ov = owner_table(va, with_y=False, extra_first=ATTR + DA, extra_mean=GM); ov['h'] = 3
F = feats(fset) + (['h'] if aug else []) + ATTR + DA + ['m_' + c for c in GM]
w = np.sqrt(ot.n) if pwin == 'sqrt' else ot.n ** float(pwin)
cfgs = [(5, 0.7), (4, 0.5), (6, 0.5), (5, 0.5), (4, 0.7), (6, 0.7)]
preds = []
for i in range(bag):
    depth, col = cfgs[i % len(cfgs)]
    m = xgb.XGBRegressor(objective='reg:logistic', n_estimators=int(kw.get('ntree', 500)), max_depth=depth, learning_rate=0.03, subsample=float(kw.get('sub', 0.8)), colsample_bytree=col,
                         min_child_weight=int(kw.get('mcw', 5)), reg_lambda=5.0, nthread=2, tree_method='hist', random_state=i)
    m.fit(ot[F], ot.rate, sample_weight=w); preds.append(m.predict(ov[F]))
ov['owner_hat'] = np.mean(preds, axis=0)
lin = float(kw.get('lin', 0))
if lin > 0:   # 권리자 수준 선형모형(로그변환·표준화·결측지시, 분수 레이블은 가중 이진 복제로 적합) 결합
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    def LX(o):
        X = o[F].astype(float).copy()
        for c in F:
            if X[c].min() >= 0 and X[c].max() > 20: X[c] = np.log1p(X[c])
        return X
    Xo = LX(ot); pos = ot.rate.values; ww = w.values if hasattr(w, 'values') else w
    X2 = pd.concat([Xo, Xo]); y2 = np.r_[np.ones(len(Xo)), np.zeros(len(Xo))]; w2 = np.r_[ww * pos, ww * (1 - pos)]
    lm = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    lm.fit(X2, y2, logisticregression__sample_weight=w2)
    ov['lin_hat'] = lm.predict_proba(LX(ov))[:, 1]
    ov['owner_hat'] = (1 - lin) * ov.owner_hat + lin * ov.lin_hat
    print('lin_hat only AUC (patent level)', round(metrics(va[['L','cur','y_MAIN']].merge(ov[['L','cur','lin_hat']], on=['L','cur']).pipe(lambda x: x.y_MAIN), va[['L','cur','y_MAIN']].merge(ov[['L','cur','lin_hat']], on=['L','cur']).lin_hat)['auc'], 4))
print('owners train', len(ot), 'val', len(ov), 'bag', bag)
d = va[['rgst_no', 'L', 'cur', 'y_MAIN']].merge(ov[['L', 'cur', 'owner_hat']], on=['L', 'cur'])
if within != 'none':
    d = d.merge(pd.read_parquet(f'cache/pred_{within}.parquet'), on=['rgst_no', 'L'])
    d['rel'] = d.p / d.groupby(['L', 'cur']).p.transform('mean')
    d['pf'] = d.owner_hat * d.rel ** beta
else:
    d['pf'] = d.owner_hat
print('owner_hat only', {k: round(x, 4) for k, x in metrics(d.y_MAIN, d.owner_hat).items()})
mt = metrics(d.y_MAIN, d.pf); ci = cluster_boot(d.y_MAIN.values, d.pf.values, d.cur.values)
for L in VAL_L:
    k = (d.L == L).values; print('  L', L, {kk: round(x, 4) for kk, x in metrics(d.y_MAIN[k], d.pf[k]).items()})
d[['rgst_no', 'L', 'pf']].rename(columns={'pf': 'p'}).to_parquet(f'cache/pred_{ver}.parquet', index=False)
ov[['L', 'cur', 'n', 'owner_hat']].to_parquet(f'cache/ownerhat_{ver}.parquet', index=False)
json.dump(dict(auc=mt['auc'], prauc=mt['prauc'], p1=mt['p1'], ci_auc=list(ci[0]), ci_pr=list(ci[1])), open(f'cache/metrics_{ver}.json', 'w'))
print('RESULT', ver, kw, round(mt['auc'], 4), round(mt['prauc'], 4), round(mt['p1'], 4), ci[0].round(3))
