# -*- coding: utf-8 -*-
r"""C19-T2 verify v2: (1) 짝 부트스트랩 다른 구현 — sklearn roc_auc_score(sample_weight), 군집 재표집 seed 12345, B=300 으로 F1·F1x Δskill CI 재계산
 (2) 상위 위험 1,000건: 공식 저장 예측(공식 F22/pred 배깅으로 다시 결합) vs 변형 — 적중 수, 교집합, 들고 난 단위의 적중률, 상위 1,000 중 fam_n 이 바뀐 단위 비율
 입력: 공식 F22(ev, pred), cache/pred_{var}.npy, cache/famn/*. 출력: verify/v2_boot.json"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from lib19 import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ev = pd.read_parquet(J(F22, 'ev.parquet'), columns=['pid', 'acd', 'y'])
y = ev.y.values.astype(int)
FROZ = J(F22, 'pred')
pb = np.load(J(FROZ, [f for f in os.listdir(FROZ) if f.startswith('base_')][0]))
p0 = np.load(J(C, 'pred_OFFICIAL.npy'))
cl = np.where(ev.acd.values == 'NA', 'P' + ev.pid.astype(str).values, ev.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)
raw = np.load(J(C, 'famn', 'P0__ev.npy'))
R = {}
for v in sys.argv[1].split(','):
    p = np.load(J(C, f'pred_{v}.npy'))
    rng = np.random.default_rng(12345); D = []
    for _ in range(300):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        ab = roc_auc_score(y, pb, sample_weight=w)
        D.append((roc_auc_score(y, p, sample_weight=w) - ab) / (1 - ab) - (roc_auc_score(y, p0, sample_weight=w) - ab) / (1 - ab))
    ab = roc_auc_score(y, pb)
    d = (roc_auc_score(y, p) - ab) / (1 - ab) - (roc_auc_score(y, p0) - ab) / (1 - ab)
    o0 = np.argsort(-p0, kind='stable')[:1000]; o1 = np.argsort(-p, kind='stable')[:1000]
    s0, s1 = set(o0), set(o1)
    out_ = np.array(sorted(s0 - s1)); in_ = np.array(sorted(s1 - s0))
    fv = np.load(J(C, 'famn', f'{v}__ev.npy')) if os.path.exists(J(C, 'famn', f'{v}__ev.npy')) else raw
    ch = fv != raw
    R[v] = dict(d_skill=float(d), d_skill_ci_sklearn_B300_seed12345=[float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))],
                top1000_hits_official=int(y[o0].sum()), top1000_hits_variant=int(y[o1].sum()), overlap=len(s0 & s1),
                dropped_n=len(out_), dropped_hit_rate=float(y[out_].mean()) if len(out_) else None, added_n=len(in_), added_hit_rate=float(y[in_].mean()) if len(in_) else None,
                top1000_official_famn_changed_share=float(ch[o0].mean()), all_units_famn_changed_share=float(ch.mean()),
                spearman_pred_official_variant=float(pd.Series(p0).corr(pd.Series(p), method='spearman')))
    out(v, R[v])
jdump(R, J(W, 'verify', 'v2_boot.json'))
