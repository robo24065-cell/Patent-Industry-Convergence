# v15: 신규진입자 전용 모형(신규진입자 행만 학습) — 신규진입자 점수는 전용 모형, 기존진입자는 v14 모형. 확률 척도로 결합.
# v16: 단일 모형에서 신규진입자 행 가중 3배
import sys
from data8 import *
t0 = time.time(); v = sys.argv[1]
tr, te = load14(); base = json.load(open(T1/'baseline_v0.json'))['fixed']
P = json.load(open(T1/'v9_cv.json'))['g4']['params']
old = pd.read_parquet(CACHE/'v14_scores.parquet')
if v == 'v15':
    NC = [c for c in C14 if c not in ('lin', 'inshare', 'dens', 'ti_mean', 'ti_max', 'ti_n20', 'ti_lsum', 'sib4', 'sib3', 'sib2')]
    trn = tr[tr.newc]; print('newcomer train rows', len(trn), 'pos', int(trn.y.sum()))
    Pn = dict(P); Pn.update(min_child_weight=20, max_depth=3)
    sn = np.mean([predict(fit_xgb(trn, NC, seed=sd, **Pn), te[te.newc], NC) for sd in range(3)], 0)
    sx = old.sx.values.copy(); sx[te.newc.values] = sn
    sl = logit(tr, te, C14); s = 0.9 * rank(sx) + 0.1 * rank(sl)
    change = '신규진입자 전용 XGB(신규 행만 학습, depth3·mcw20, 범위안 전용 피처 제외)로 신규진입자 점수 교체'; hyp = '신규진입자는 범위안 이력이 없어 다른 결정구조를 가진다: 전용 모형이 낫다'
else:
    import xgboost as xgb
    w = np.where(tr.newc.values, 3.0, 1.0)
    ss = []
    for sd in range(3):
        p = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method='hist', nthread=2, random_state=sd); p.update(P)
        m = xgb.XGBClassifier(**p); m.fit(tr[C14].values.astype(np.float32), tr.y.values, sample_weight=w); ss.append(m.predict_proba(te[C14].values.astype(np.float32))[:, 1])
    sx = np.mean(ss, 0); sl = logit(tr, te, C14); s = 0.9 * rank(sx) + 0.1 * rank(sl)
    change = '단일 모형에서 신규진입자 행 가중 3배'; hyp = '신규진입자 사례가 적어 트리가 그쪽 분할을 덜 배운다: 가중으로 보완'
met = metrics(te, s); mx = metrics(te, sx)
print(v, 'xgb only', fmt(mx)); print(v, fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/f'{v}_scores.parquet', index=False)
lb_add(v, change, hyp, met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")
