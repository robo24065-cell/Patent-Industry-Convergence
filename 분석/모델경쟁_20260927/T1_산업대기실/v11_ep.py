# v11: 해외 선행 신호(제공 5개국 데이터 중 EP) — 법인의 EP 출원이 KSIC k에 먼저 나타나면 KR 진입이 뒤따른다
from data8 import *
t0 = time.time()
tr, te = load8(); tr = add_ep(tr); te = add_ep(te)
base = json.load(open(T1/'baseline_v0.json'))['fixed']
print('EP coverage valid firms', (te.drop_duplicates('firm').ep_n > 0).mean().round(3), 'newcomer firms', (te[te.newc].drop_duplicates('firm').ep_n > 0).mean().round(3),
      '| pairs ep_k>0', int((te.ep_k > 0).sum()), 'entry rate ep_k>0', te[te.ep_k > 0].y.mean().round(4), 'vs', te[te.ep_k == 0].y.mean().round(4))
P = json.load(open(T1/'v9_cv.json'))['g4']['params']; cols = C8 + EPF
sx = np.mean([predict(fit_xgb(tr, cols, seed=sd, **P), te, cols) for sd in range(3)], 0); sl = logit(tr, te, cols)
s = 0.9 * rank(sx) + 0.1 * rank(sl); met = metrics(te, s); mx = metrics(te, sx)
print('v11 xgb only', fmt(mx)); print('v11', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v11_scores.parquet', index=False)
lb_add('v11', '+해외 선행(EP 공개분≤기준시점: KSIC별 건수·총수·비중·최근3년·KR우선권) — 제공 5개국 데이터 융합', '해외에서 먼저 그 산업 특허를 낸 기업이 KR에 진입한다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")
