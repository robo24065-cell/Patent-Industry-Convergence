# 엄격/공개 정보집합 실험 러너: T1_STRICT=1 [T1_VIS=pub] python vstrict.py <version> <cuts comma> "<change>" "<hypothesis>"
import os, sys; assert os.environ.get('T1_STRICT') == '1'
from data8 import *
v = sys.argv[1]; cuts = tuple(int(x) for x in sys.argv[2].split(',')); change, hyp = sys.argv[3], sys.argv[4]
extra = sys.argv[5].split(',') if len(sys.argv) > 5 else []
t0 = time.time(); tr, te = load19(cuts); base = json.load(open(T1/'baseline_v0.json'))['fixed']
COLS = list(C19)
if 'pa' in extra: tr = add_pa(tr); te = add_pa(te); COLS += PAF
if 'rr' in extra: tr, te = add_rr(tr, te, cuts); COLS += RRF
if 'now' in extra: tr, te = add_now(tr, te, cuts); COLS += NOWF
if 'nopart' in extra: COLS = [c for c in COLS if c not in PART]
ref = pd.read_parquet(CACHE/'v22_scores.parquet'); assert (ref.firm.values == te.firm.values).all() and (ref.y.values == te.y.values).all()
w = ipw_weights(tr)
if 'dropempty' in extra:
    keep = tr.lsize.values > 0; print('drop empty-history train rows', int((~keep).sum()), 'of', len(tr)); tr = tr[keep].reset_index(drop=True); w = w[keep]
PP = None
for e in extra:
    if e.startswith('g') and e[1:].isdigit(): PP = json.load(open(T1/'v9_cv.json'))[e]['params'] or {'n_estimators': 400}; print('params', e, PP)
s, sx = run_std(tr, te, COLS, P=PP, w=w, key=v); met = metrics(te, s); mx = metrics(te, sx)
sb = logit(tr, te, BASE); mb = metrics(te, sb)
print(v, 'xgb only', fmt(mx)); print(v, fmt(met)); print(v, 'same-info baseline', fmt(mb), round(time.time() - t0))
json.dump(dict(model=met, baseline=mb, vis=VIS, cuts=cuts), open(T1/f'{v}_result.json', 'w'), indent=1)
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx, sb=sb).to_parquet(CACHE/f'{v}_scores.parquet', index=False)
lb_add(v, change, hyp, met, base, '?', f"[{VIS}모드] xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}; 동일정보 기준선 logit auc_all={mb['auc_all']:.4f} auc_new={mb['auc_new']:.4f} pr_new={mb['pr_new']:.4f}")
