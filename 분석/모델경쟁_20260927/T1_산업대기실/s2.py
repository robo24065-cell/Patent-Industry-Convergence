# 2단계 실험 러너(엄격 정보집합): python s2.py <version> "<change>" "<hypothesis>" opt1,opt2,...
# opts: pa2 | nodart | kprior | publab<w> | fany | g<N>(v9_cv 파라미터) | s2p<N>(s2_cv 파라미터) | seeds<N>
import os, sys; os.environ['T1_STRICT'] = '1'; os.environ.setdefault('T1_VIS', 'rgst')
from s2lib import *
v, change, hyp = sys.argv[1], sys.argv[2], sys.argv[3]
opts = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else []
t0 = time.time(); cuts = (2001, 2002, 2003, 2004)
tr, te = load_s2(cuts)
COLS = list(C32 if 'pa2' in opts else C31)
if 'pa2' not in opts:
    tr = add_pa(tr); te = add_pa(te)
if 'nodart' in opts: COLS = [c for c in COLS if c not in DART]
w = ipw_weights(tr)
if 'kprior' in opts: tr, te = k_prior(tr, te); COLS += ['k_prior', 'k_prior_new']
if 'fany' in opts: tr, te = firm_stage(tr, te, COLS); COLS += ['f_any']
if 'fanypub' in opts:
    trf = tr.assign(y=np.maximum(tr.y.values, pub_label(tr))); trf, te = firm_stage(trf, te, COLS); tr['f_any'] = trf.f_any.values; COLS += ['f_any']
P = None; seeds = 3
for o in opts:
    if o.startswith('g') and o[1:].isdigit(): P = json.load(open(T1/'v9_cv.json'))[o]['params'] or {'n_estimators': 400}
    if o.startswith('s2p'): P = json.load(open(T1/'s2_cv.json'))[o]['params']
    if o.startswith('seeds'): seeds = int(o[5:])
note = ''
for o in opts:
    if o.startswith('publab'):
        wp = float(o[6:] or 1); yp = pub_label(tr)
        add = (yp == 1) & (tr.y.values == 0)
        note += f'publab: 등록 양성 {int(tr.y.sum())} + 공개만 양성 {int(add.sum())}(가중 {wp}); '
        tr = tr.copy(); tr['y'] = np.maximum(tr.y.values, yp); w = np.where(add, wp, w)
print(v, 'opts', opts, 'ncols', len(COLS), 'train', len(tr), 'pos', int(tr.y.sum()), round(time.time() - t0), flush=True)
s, sx = run_std(tr, te, COLS, P=P, seeds=seeds, w=w, key=v)
met = metrics(te, s); mx = metrics(te, sx)
base = json.load(open(T1/'baseline_v0.json'))['fixed']
sb = pd.read_parquet(CACHE/'v31_scores.parquet').sb.values; mb = metrics(te, sb)
print(v, 'xgb only', fmt(mx)); print(v, fmt(met), round(time.time() - t0))
json.dump(dict(model=met, xgb=mx, baseline_strict=mb, cols=COLS, opts=opts, params=P), open(T1/f'{v}_result.json', 'w'), indent=1, ensure_ascii=False)
te[['firm', 'k', 'y', 'newc', 'lsize']].assign(s=s, sx=sx, sb=sb).to_parquet(CACHE/f'{v}_scores.parquet', index=False)
lb_add(v, change, hyp, met, base, '?', f"[2단계·엄격] {note}xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}; 동일정보 기준선 auc_all={mb['auc_all']:.4f} auc_new={mb['auc_new']:.4f}")
