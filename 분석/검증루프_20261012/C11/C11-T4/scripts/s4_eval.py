# 후보 A: P2(Δskill 대 PI·SB)와 P4b(AUC 제공 − 공개 피처), P4c(공개 세계) 군집 부트스트랩. 사용: python -I s4_eval.py P2|P4|P4c
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib11 import *
from evalkit import AUCPrep, skill

what = sys.argv[1]
t_start = time.time()
B = 200


def load(world):
    return {v: pd.read_parquet(C + rf'\pred_A_{world}_{v}.parquet') for v in EVAL_Y}


def boot(pairs, fn, twoway):
    """pairs: list of (P1prep, P0prep, fi, k). fn: 'skill' or 'diff'."""
    f = skill if fn == 'skill' else (lambda a, b: a - b)
    obs = float(np.mean([f(p1.auc(), p0.auc()) for p1, p0, fi, k in pairs]))
    nf = max(int(fi.max()) for _, _, fi, _ in pairs) + 1
    Wf = np.random.default_rng(SEED).poisson(1.0, size=(B, nf)).astype(float)
    Wk = np.random.default_rng(SEED + 7).poisson(1.0, size=(B, 50)).astype(float)
    bs = np.empty(B)
    for b in range(B):
        vals = []
        for p1, p0, fi, k in pairs:
            w = Wf[b, fi]
            if twoway:
                w = w * Wk[b, k]
            vals.append(f(p1.auc(w), p0.auc(w)))
        bs[b] = np.mean(vals)
    return dict(obs=obs, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), se=float(bs.std(ddof=1)))


res = {}
if what == 'P2':
    Pp = load('p')
    allf = pd.Index(np.unique(np.concatenate([P.fid.values for P in Pp.values()])))
    per = {}
    for v, P in Pp.items():
        per[v] = {c: AUCPrep(P[c].values, P.y.values).auc() for c in ['m', 'pi', 'sb', 'm_20261013', 'm_20261014', 'm_20261015']}
    res['auc_by_origin'] = per
    for base in ['pi', 'sb']:
        for mcol in ['m', 'm_20261013', 'm_20261014', 'm_20261015']:
            pairs = [(AUCPrep(P[mcol].values, P.y.values), AUCPrep(P[base].values, P.y.values), allf.get_indexer(P.fid.values), P.k.values) for P in Pp.values()]
            for tw in ([False, True] if mcol == 'm' else [False]):
                key = f'{mcol}_vs_{base}_{"twoway" if tw else "firm"}'
                res[key] = boot(pairs, 'skill', tw)
                out(key, res[key], round(time.time() - t_start, 1))
    res['pass'] = bool(res['m_vs_pi_firm']['lo'] > 0 and res['m_vs_sb_firm']['lo'] > 0)
elif what == 'P4':
    Pp = load('p')
    for arm in ['fi', 'hs']:
        Pa = load(arm)
        allf = pd.Index(np.unique(np.concatenate([P.fid.values for P in Pp.values()])))
        pairs = []
        for v in EVAL_Y:
            a, b = Pp[v], Pa[v]
            assert (a.fid.values == b.fid.values).all() and (a.k.values == b.k.values).all() and (a.y.values == b.y.values).all()
            pairs.append((AUCPrep(a.m.values, a.y.values), AUCPrep(b.m.values, b.y.values), allf.get_indexer(a.fid.values), a.k.values))
        res[arm] = dict(auc_pub={v: AUCPrep(Pa[v].m.values, Pa[v].y.values).auc() for v in EVAL_Y})
        for tw in [False, True]:
            res[arm]['twoway' if tw else 'firm'] = boot(pairs, 'diff', tw)
            out(arm, tw, res[arm]['twoway' if tw else 'firm'], round(time.time() - t_start, 1))
        # 시드별(보고)
        res[arm]['by_seed'] = {}
        for sd in ['m_20261013', 'm_20261014', 'm_20261015']:
            pr = [(AUCPrep(Pp[v][sd].values, Pp[v].y.values), AUCPrep(Pa[v][sd].values, Pa[v].y.values), allf.get_indexer(Pp[v].fid.values), Pp[v].k.values) for v in EVAL_Y]
            res[arm]['by_seed'][sd] = boot(pr, 'diff', False)
        res[arm]['pass_b'] = bool(res[arm]['firm']['lo'] > 0 and res[arm]['twoway']['lo'] > 0)
elif what == 'P4c':
    for arm in ['fi', 'hs']:
        Pa = load('pubw_' + arm)
        allf = pd.Index(np.unique(np.concatenate([P.fid.values for P in Pa.values()])))
        res[arm] = {}
        for base in ['pi', 'sb']:
            pairs = [(AUCPrep(P.m.values, P.y.values), AUCPrep(P[base].values, P.y.values), allf.get_indexer(P.fid.values), P.k.values) for P in Pa.values()]
            res[arm][base] = boot(pairs, 'skill', False)
        res[arm]['auc'] = {v: {c: AUCPrep(P[c].values, P.y.values).auc() for c in ['m', 'pi', 'sb']} for v, P in Pa.items()}
        res[arm]['pos'] = {v: int(P.y.sum()) for v, P in Pa.items()}
        out(arm, res[arm]['pi'], res[arm]['sb'])
json.dump(res, open(C + rf'\eval_{what}.json', 'w'), indent=1, default=float)
out('done', what, round(time.time() - t_start, 1))
