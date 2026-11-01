# 진단3: 타이밍 피처 하나씩 추가(1시드) — 어떤 피처가 AUC를 떨어뜨리는가
import sys
from data8 import *
cuts = (2002, 2003, 2004); tr, te = load14(cuts)
xtr = pd.concat([tim_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = tim_panel(2009, 'None')
for c in TIMF: tr[c] = xtr[c].values; te[c] = xte[c].values
P = json.load(open(T1/'v9_cv.json'))['g4']['params']
for g in sys.argv[1:]:
    add = [] if g == 'none' else g.split('+')
    cols = C14 + add; s = predict(fit_xgb(tr, cols, seed=0, **P), te, cols); m = metrics(te, s)
    print(g, 'auc_all=%.4f auc_new=%.4f auc_inc=%.4f pr_all=%.4f pr_new=%.4f' % (m['auc_all'], m['auc_new'], m['auc_inc'], m['pr_all'], m['pr_new']), flush=True)
    if g != 'none':
        for c in add: print('   ', c, 'train mean/max %.2f/%.2f' % (tr[c].mean(), tr[c].max()), 'valid mean/max %.2f/%.2f' % (te[c].mean(), te[c].max()), ' newc train %.2f valid %.2f' % (tr[tr.newc][c].mean(), te[te.newc][c].mean()))
