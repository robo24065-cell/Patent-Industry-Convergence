# v10: XGB(v9) + 로지스틱(동일 피처, 표준화) 순위 평균 앙상블 — 가중치는 학습 패널 내부 CV로 선택
from data8 import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
t0 = time.time()
tr, te = load8(); base = json.load(open(T1/'baseline_v0.json'))['fixed']
cv = json.load(open(T1/'v9_cv.json')); P = cv['g4']['params']
# 내부 CV: fold별 xgb/logit 점수 → 가중치 w 선택
oof_x = np.zeros(len(tr)); oof_l = np.zeros(len(tr))
for a, b in GroupKFold(3).split(tr, groups=tr.firm):
    oof_x[b] = predict(fit_xgb(tr.iloc[a], C8, **P), tr.iloc[b], C8)
    oof_l[b] = logit(tr.iloc[a], tr.iloc[b], C8)
ws = {}
for w in [0, 0.1, 0.2, 0.3, 0.5]:
    s = (1 - w) * rank(oof_x) + w * rank(oof_l); ws[w] = roc_auc_score(tr.y, s)
wb = max(ws, key=ws.get); print('inner CV', ws, 'pick', wb, round(time.time() - t0), flush=True)
sx = pd.read_parquet(CACHE/'v9_scores.parquet').s.values; sl = logit(tr, te, C8)
print('logit alone', fmt(metrics(te, sl)))
s = (1 - wb) * rank(sx) + wb * rank(sl); met = metrics(te, s)
print('v10', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s).to_parquet(CACHE/'v10_scores.parquet', index=False)
lb_add('v10', f'XGB(v9)+로지스틱 순위평균 앙상블, 가중 w={wb}(내부CV)', '선형모형이 트리의 외삽 약점을 보완한다', met, base, '?', f'inner CV auc by w={ {k: round(v, 4) for k, v in ws.items()} }')
