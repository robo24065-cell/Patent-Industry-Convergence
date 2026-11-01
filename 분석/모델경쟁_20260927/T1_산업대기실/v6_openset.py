# v6: open-set 51-class(OTHER=범위밖 자사출원, 법인 2-fold 교차적합) 확률로 노출 재계산
import sys
from exp import *
from feats import agg_feats, industry_mom
t0 = time.time()
tag = 'c2009_os51'
P, PO = train_clf51(CAP_TR, tag)
print('clf51', time.time() - t0, flush=True)
def pan(cut, cap):
    p = PC/f'panel_c{cut}_{cap}_{tag}_.parquet'
    d, c = build_panel(cut, 5, cap=cap)
    if p.exists(): return pd.read_parquet(p), c
    d = add_dart(d, c); d = agg_feats(d, c, P); d = industry_mom(d, c)
    # 법인의 범위밖 특허 중 '50산업 신호' 비율(1-P_other 평균)
    op = c['op']; rows = c['fi'][op.applicant_cd].values
    s50 = np.bincount(rows, weights=1 - PO[op.row.values], minlength=len(c['firms'])) / np.maximum(np.bincount(rows, minlength=len(c['firms'])), 1)
    d['in50'] = s50[d.fidx.values]; d['cut'] = cut
    d.to_parquet(p, index=False); return d, c
tr = pd.concat([pan(c, CAP_TR)[0] for c in (2002, 2003, 2004)], ignore_index=True)
te, cte = pan(2009, None)
base = json.load(open(T1/'baseline_v0.json'))['fixed']
cols = BASE + AGG + MOM + NBR + KMOM + ['in50']
s = predict(fit_xgb(tr, cols), te, cols); met = metrics(te, s)
print('v6', fmt(met), time.time() - t0)
te[['firm', 'k', 'y', 'newc']].assign(s=s).to_parquet(CACHE/'v6_scores.parquet', index=False)
lb_add('v6', '분류기를 open-set 51-class로 교체(범위밖 자사출원=OTHER, 법인 2-fold 교차적합) + 법인 50산업 신호비율', '50-class 강제 배분은 범위밖 특허에 가짜 산업 확률을 준다: OTHER를 두면 진짜 산업 신호만 남는다', met, base, '?', '')
