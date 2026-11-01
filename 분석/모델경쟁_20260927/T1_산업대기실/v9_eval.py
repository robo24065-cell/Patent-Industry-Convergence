from data8 import *
t0 = time.time()
tr, te = load8(); base = json.load(open(T1/'baseline_v0.json'))['fixed']
cv = json.load(open(T1/'v9_cv.json')); g = max(cv, key=lambda x: cv[x]['cv_auc']); P = cv[g]['params']
ss = [predict(fit_xgb(tr, C8, seed=sd, **P), te, C8) for sd in range(3)]
s = np.mean(ss, 0); met = metrics(te, s)
print('v9', g, fmt(met), time.time() - t0)
te[['firm', 'k', 'y', 'newc']].assign(s=s).to_parquet(CACHE/'v9_scores.parquet', index=False)
lb_add('v9', f'XGB 하이퍼파라미터 내부 CV(법인 GroupKFold) 선택 {g}={P}, 3시드 평균확률', '얕고 규제 강한 트리가 희소 사건에서 일반화가 낫다', met, base, '?', f'CV auc {cv[g]["cv_auc"]:.4f} vs 기본 {cv["g0"]["cv_auc"]:.4f}')
