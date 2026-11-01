# C09-T4 s6: T4 채점 — 원판(트랙 공식 예측, 읽기 전용) vs 보정판(R1·R2·R3 재학습). 앙상블은 동결 spec 가중(로짓 가중평균).
#  F22: final_run_r2.cluster_boot 와 같은 가중 재표집(B=1000, seed 0, 1순위 출원인코드 군집)을 원판·보정판에 같은 가중으로 적용 → Δskill CI.
#  개발 원점: 원점별 skill·Δ, 같은 방식 B=200(spec bootstrap B) 짝 부트스트랩.
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
from s2_t4_feats import fdir
from s3_t4_train import SPEC, combine_haz, pdir
from sklearn.metrics import roc_auc_score
KEYS = {'cls': 'cls_99f270b3cd', 'haz': 'haz_0568d4297b', 'logit': 'logit_3008b8b195', 'hgb': 'hgb_b4bd0fa46a', 'base': 'base_46526c56d8'}
WS = SPEC['ensemble']['weights']
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def comp_preds(o, rule):
    if rule == 'official':
        d = os.path.join(fdir(o), 'pred')
        return {k: np.load(os.path.join(d, KEYS[k] + '.npy')) for k in WS}
    d = pdir(o, rule)
    ps = {}
    for k in ['cls', 'haz']:
        fs = [os.path.join(d, f'{k}_b{b}.npy') for b in range(3)]
        if not all(os.path.exists(f) for f in fs): return None
        z = np.mean([np.load(f) for f in fs], 0)
        ps[k] = 1 / (1 + np.exp(-z)) if k == 'cls' else combine_haz(o, z)
    for k in ['logit', 'hgb']:
        f = os.path.join(d, f'{k}.npy')
        if not os.path.exists(f): return None
        ps[k] = np.load(f)
    return ps


def ens(ps):
    z = sum(w * lg(ps[k]) for k, w in WS.items()) / sum(WS.values())
    return 1 / (1 + np.exp(-z))


def wauc_prep(y, s):
    o = np.argsort(s, kind='mergesort'); so = s[o]
    grp = np.r_[0, np.cumsum(so[1:] != so[:-1])]
    return o, y[o].astype(float), grp


def wauc(pr, w):
    o, yo, grp = pr
    wo = w[o]; G = grp[-1] + 1
    P = np.bincount(grp, wo * yo, G); N = np.bincount(grp, wo * (1 - yo), G)
    below = np.cumsum(N) - N
    return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


def score(o, B):
    ev = pd.read_parquet(os.path.join(fdir(o), 'ev.parquet'), columns=['pid', 'acd', 'y'])
    m = ev.y.notna().values; y = ev.y.values[m]
    pb = np.load(os.path.join(fdir(o), 'pred', KEYS['base'] + '.npy'))[m]
    P = {}
    for r in ['official', 'orig', 'R1', 'R2', 'R3']:
        ps = comp_preds(o, r) if r != 'orig' else (comp_preds(o, 'orig') if o == 'F22' else None)
        if ps is None: continue
        P[r] = ens(ps)[m]
        if r == 'official': comps_off = {k: float(roc_auc_score(y, v[m])) for k, v in ps.items()}
    ab = roc_auc_score(y, pb)
    res = dict(origin=o, n=int(m.sum()), base_rate=float(y.mean()), auc_base=float(ab))
    for r, p in P.items():
        a = roc_auc_score(y, p); res[f'auc_{r}'] = float(a); res[f'skill_{r}'] = float((a - ab) / (1 - ab))
    if 'orig' in P:
        res['orig_equals_official_maxabs'] = float(np.abs(P['orig'] - P['official']).max())
    # 짝 부트스트랩
    acd = ev.acd.values[m].astype(str); pid = ev.pid.values[m]
    cl = np.where(acd == 'NA', 'P' + pid.astype(str), acd)
    codes, uniq = pd.factorize(cl); K = len(uniq)
    prb = wauc_prep(y, pb); prs = {r: wauc_prep(y, p) for r, p in P.items()}
    rng = np.random.default_rng(0)
    S = {r: [] for r in P}
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        ab_ = wauc(prb, w)
        for r in P:
            S[r].append((wauc(prs[r], w) - ab_) / (1 - ab_))
    S = {r: np.array(v) for r, v in S.items()}
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    for r in P:
        res[f'skill_ci_{r}'] = q(S[r])
        if r not in ('official', 'orig'):
            res[f'delta_{r}'] = float(res[f'skill_{r}'] - res['skill_official'])
            res[f'delta_ci_{r}'] = q(S[r] - S['official'])
            res[f'delta_auc_{r}'] = float(res[f'auc_{r}'] - res['auc_official'])
    res['B'] = B; res['n_clusters'] = int(K); res['components_official'] = comps_off
    return res


if __name__ == '__main__':
    outs = []
    for o in sys.argv[1:]:
        r = score(o, 1000 if o == 'F22' else 200)
        outs.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False), flush=True)
        json.dump(r, open(os.path.join(C, f't4_score_{o}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
