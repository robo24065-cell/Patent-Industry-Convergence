# -*- coding: utf-8 -*-
"""2단계 실험 러너(검증 프로토콜 C=2014-01-01, 평가 L=2014-01..2015-07 반기 4개).
사용:
  python s2exp.py build            # 학습·평가 프레임 캐시(cache/s2_tr.parquet, s2_ev.parquet)
  python s2exp.py cls <tag> <set>  # XGB 3년 분류형(v30 하이퍼파라미터)으로 피처세트 비교 -> preds/pred_<tag>.parquet
피처세트: v30 | fix | fix_mac | fix_ag | fix_ow | fix_all ...
"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd, xgboost as xgb
import t4lib as T, soc as SO, feat2 as F2
from run import train_landmarks, get_frame, metrics, per_L, C_VAL, EVAL_L, labels
from versions import add_ratio, add_fam, F23N, F6C, lm_w

HERE = T.HERE
TR = os.path.join(T.CACHE, 's2_tr.parquet')
EV = os.path.join(T.CACHE, 's2_ev.parquet')
XGBP = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', nthread=2, max_depth=6, eta=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20, max_cat_to_onehot=8, reg_lambda=1.0, seed=0)
CLS_N = F23N + ['ak_rate', 'ak_n']
FD = ['fd_' + c for c in F2.FAM]


def enrich2(f):
    d = add_fam(SO.add_soc(add_ratio(f)))
    d = SO.add_ak(d, 'next_annl')
    fd = F2.add_fam_dated(d[['pid', 'L']].copy())
    for c in F2.FAM:
        d['fd_' + c] = fd[c].values
    d['dspst_fix'] = F2.fix_dspst(d[['dspst']]).dspst.values
    d = F2.add_macro(d)
    d = F2.add_agent(d)
    d = F2.add_owner(d)
    return d


def f32(d):
    fc = d.select_dtypes('float64').columns
    d = d.astype({c: 'float32' for c in fc})
    oc = [c for c in d.select_dtypes('object').columns if c != 'L']
    return d.astype({c: 'category' for c in oc})


def build():
    t = time.time()
    parts = []
    for L in train_landmarks(C_VAL):
        f = get_frame([L], C_VAL)
        parts.append(f32(enrich2(f)))
        print('tr', L, len(f), round(time.time() - t, 1), flush=True)
    tr = pd.concat(parts, ignore_index=True)
    tr = tr.astype({c: str for c in tr.select_dtypes('category').columns})
    tr.to_parquet(TR, index=False)
    parts = []
    for L in EVAL_L:
        f = get_frame([L], None)
        parts.append(f32(enrich2(f)))
        print('ev', L, len(f), round(time.time() - t, 1), flush=True)
    ev = pd.concat(parts, ignore_index=True)
    ev = ev.astype({c: str for c in ev.select_dtypes('category').columns})
    ev.to_parquet(EV, index=False)
    print('built', tr.shape, ev.shape, round(time.time() - t, 1))


def fsets():
    cats0 = F6C
    catsF = [c if c != 'dspst' else 'dspst_fix' for c in F6C]
    base = [c for c in CLS_N if c not in F2.FAM]
    S = {
        'v30': (CLS_N, cats0),
        'fix': (base + FD, catsF),
        'fix_nofam': (base, catsF),
        'fix_famrd': (base + ['fr_' + c for c in F2.FAM], catsF),
        'fix_famrd_mac': (base + ['fr_' + c for c in F2.FAM] + F2.MACRO, catsF),
        'fix_mac_nofam': (base + F2.MACRO, catsF),
        'all_famrd': (base + ['fr_' + c for c in F2.FAM] + F2.MACRO + F2.AGENT + F2.OWNER, catsF),
        'all_nofam': (base + F2.MACRO + F2.AGENT + F2.OWNER, catsF),
        'ow_ag_famrd': (base + ['fr_' + c for c in F2.FAM] + F2.AGENT + F2.OWNER, catsF),
        'fix_mac': (base + FD + F2.MACRO, catsF),
        'fix_ag': (base + FD + F2.AGENT, catsF),
        'fix_ow': (base + FD + F2.OWNER, catsF),
        'fix_ag_ow': (base + FD + F2.AGENT + F2.OWNER, catsF),
        'ow_nofam': (base + F2.OWNER, catsF),
        'ow_mac': (base + FD + F2.OWNER + F2.MACRO, catsF),
        'fix_all': (base + FD + F2.MACRO + F2.AGENT + F2.OWNER, catsF),
    }
    return S


def xm(d, num, cats, catmap):
    X = d[num].astype('float32').copy()
    for c in cats:
        X[c] = pd.Categorical(d[c].astype(str), categories=catmap[c])
    return X


def fit_cls(tr, ev, num, cats, params=None, rounds=600, weight=True):
    catmap = {c: sorted(tr[c].astype(str).unique()) for c in cats}
    P = dict(XGBP); P.update(params or {})
    dm = xgb.DMatrix(xm(tr, num, cats, catmap), tr.y.values, weight=lm_w(tr) if weight else None, enable_categorical=True)
    b = xgb.train(P, dm, rounds)
    return b.predict(xgb.DMatrix(xm(ev, num, cats, catmap), enable_categorical=True)), b


def run_cls(tag, setname, **kw):
    t = time.time()
    num, cats = fsets()[setname]
    if os.environ.get('COLPERM'):
        num = list(np.random.default_rng(int(os.environ['COLPERM'])).permutation(num))
    need = [c for c in num if not c.startswith('fr_')]
    tr = pd.read_parquet(TR, columns=list(set(need + cats + ['y', 'L', 'pid'])))
    ev = pd.read_parquet(EV, columns=list(set(need + cats + ['y', 'L', 'pid', 'ksic'])))
    if any(c.startswith('fr_') for c in num):
        for d in (tr, ev):
            fr = F2.add_fam_dated(d[['pid', 'L']].copy(), mode='rd')
            for c in F2.FAM: d['fr_' + c] = fr[c].values
    p, b = fit_cls(tr, ev, num, cats, **kw)
    m = metrics(ev.y.values, p); pl = per_L(ev, p)
    out = ev[['pid', 'L', 'ksic', 'y']].copy(); out['p'] = p
    out.to_parquet(os.path.join(HERE, 'preds', f'pred_{tag}.parquet'), index=False)
    imp = pd.Series(b.get_score(importance_type='total_gain')).sort_values(ascending=False)
    print(tag, setname, json.dumps({k: round(v, 5) for k, v in m.items()}), pl, f'n_feat={len(num) + len(cats)} {time.time() - t:.0f}s', flush=True)
    newf = [c for c in imp.index if c in F2.MACRO + F2.AGENT + F2.OWNER + FD or c.startswith('fr_')]
    print('gain rank of new feats:', {c: int(list(imp.index).index(c)) + 1 for c in newf}, 'of', len(imp))
    return m, pl


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    elif sys.argv[1] == 'cls':
        run_cls(sys.argv[2], sys.argv[3], params={'seed': int(os.environ.get('SEED', '0'))})
