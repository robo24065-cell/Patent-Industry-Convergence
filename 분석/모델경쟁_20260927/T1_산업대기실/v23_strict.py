# v23: 엄격 정보집합 — 모든 패널(학습·검증)의 피처를 '기준시점 연말까지 등록·공개된 특허'만으로 계산(미래 등록 여부 사용 안 함).
#      위험집합·결과·법인 모집단은 v22와 동일(행 일치 확인). 배포(2025년 예측) 시점 정보집합과 같은 방식.
import os; assert os.environ.get('T1_STRICT') == '1'
from data8 import *
t0 = time.time(); cuts = (2001, 2002, 2003, 2004)
tr, te = load19(cuts); base = json.load(open(T1/'baseline_v0.json'))['fixed']
ref = pd.read_parquet(CACHE/'v22_scores.parquet'); assert (ref.firm.values == te.firm.values).all() and (ref.k.values == te.k.values).all() and (ref.y.values == te.y.values).all()
w = ipw_weights(tr)
s, sx = run_std(tr, te, C19, w=w); met = metrics(te, s); mx = metrics(te, sx)
print('v23 xgb only', fmt(mx)); print('v23', fmt(met), round(time.time() - t0))
# 엄격 모드 기준선도 같은 정보집합으로 재계산(공정 비교)
sb = logit(tr, te, BASE); mb = metrics(te, sb); print('strict baseline logit', fmt(mb))
json.dump(mb, open(T1/'baseline_strict.json', 'w'), indent=1)
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx, sb=sb).to_parquet(CACHE/'v23_scores.parquet', index=False)
lb_add('v23', 'v22를 엄격 정보집합으로: 모든 피처를 기준시점 연말까지 등록된 특허만으로(학습·검증 동일 규칙)', '출원 후 미래에 등록된 사실을 피처에 쓰지 않아도 성능이 유지되면, 누수 의심 없이 배포 시점(2025)에 그대로 쓸 수 있다', met, base, '?',
       f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}; 엄격모드 기준선 logit auc_all={mb['auc_all']:.4f} auc_new={mb['auc_new']:.4f} pr_new={mb['pr_new']:.4f}")
