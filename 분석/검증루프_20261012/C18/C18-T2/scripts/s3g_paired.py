# -*- coding: utf-8 -*-
r"""C18-T2 s3g: 짝지은 군집 부트스트랩(B=1000, seed 0) — Δskill = skill(V) − skill(P0).
 T3: 군집 = IPC 메인그룹(mg). 같은 재표본(군집 중복 수 = 가중)을 P0·V 양쪽에 적용, 각자 자기 기준선(p_base)과의 skill. final(t0=2021) + dev(9원점 평균 skill).
 N5: 군집 = 출원인(app), skill = (AUC_champ − AUC_M0)/(1 − AUC_M0).
 가중 AUC는 정렬 1회 + 동률 묶음으로 O(n) 계산(가중치 1이면 sklearn roc_auc_score 와 같음을 먼저 확인).
 사용: python s3g_paired.py t3 <VAR> | n5 <VAR>  → cache/paired_<trk>_<VAR>.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
trk, VAR = sys.argv[1], sys.argv[2]; B = 1000; t = time.time()


class WA:
    def __init__(self, s, y, g):
        o = np.argsort(s, kind='mergesort'); ss = s[o]
        self.g = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]; self.y = y[o].astype(bool); self.c = g[o]; self.G = self.g[-1] + 1

    def auc(self, w_cl):
        w = w_cl[self.c]; wp = np.where(self.y, w, 0.0); wn = np.where(self.y, 0.0, w)
        gp = np.bincount(self.g, wp, self.G); gn = np.bincount(self.g, wn, self.G)
        den = gp.sum() * gn.sum()
        return float((gp * (np.cumsum(gn) - gn + 0.5 * gn)).sum() / den) if den > 0 else np.nan


def check(s, y):
    a = WA(s, y, np.zeros(len(s), int)).auc(np.ones(1)); b = roc_auc_score(y, s); assert abs(a - b) < 1e-12, (a, b)


res = {}
rng = np.random.default_rng(0)
if trk == 't3':
    T = os.path.join(C, 't3res')
    blocks = {}
    for mode in ('final', 'dev'):
        def load(v):
            if mode == 'final':
                return pd.read_parquet(os.path.join(T, f'preds_{v}_final.parquet'))
            return pd.concat([pd.read_parquet(os.path.join(T, f'preds_{v}_dev_{o}.parquet')) for o in ('2010-2011-2012-2013', '2014-2015-2016-2017-2018')])
        try:
            P, V = load('P0'), load(VAR)
        except FileNotFoundError as e:
            res[mode] = f'missing {e}'; continue
        mgs = np.array(sorted(set(P.mg) | set(V.mg))); mi = {m: i for i, m in enumerate(mgs)}
        bl = []
        for o in sorted(P.t0.unique()):
            pp, vv = P[P.t0 == o], V[V.t0 == o]
            for d in (pp, vv):
                check(d.p.values, d.yb.values)
            gp = pp.mg.map(mi).values; gv = vv.mg.map(mi).values
            bl.append([WA(pp.p.values, pp.yb.values, gp), WA(pp.p_base.values, pp.yb.values, gp), WA(vv.p.values, vv.yb.values, gv), WA(vv.p_base.values, vv.yb.values, gv)])
        sk = lambda a, b: (a - b) / (1 - b)
        one = np.ones(len(mgs))
        pt_ = [[sk(b[0].auc(one), b[1].auc(one)), sk(b[2].auc(one), b[3].auc(one))] for b in bl]
        pt_ = np.array(pt_).mean(0)
        D = []
        for r in range(B):
            w = np.bincount(rng.integers(0, len(mgs), len(mgs)), minlength=len(mgs)).astype(float)
            s0 = np.mean([sk(b[0].auc(w), b[1].auc(w)) for b in bl]); s1 = np.mean([sk(b[2].auc(w), b[3].auc(w)) for b in bl])
            D.append((s0, s1, s1 - s0))
        D = np.array(D); q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res[mode] = dict(skill_P0=float(pt_[0]), skill_V=float(pt_[1]), delta=float(pt_[1] - pt_[0]), delta_ci95=q(D[:, 2]), skill_P0_ci95=q(D[:, 0]),
                         n_cells_P0=int(len(P)), n_cells_V=int(len(V)), n_clusters=int(len(mgs)), origins=[int(x) for x in sorted(P.t0.unique())])
        out(trk, VAR, mode, json.dumps(res[mode]), round(time.time() - t))
else:
    N = os.path.join(C, 'n5res')
    P = pd.read_parquet(os.path.join(N, 'preds_P0.parquet')); V = pd.read_parquet(os.path.join(N, f'preds_{VAR}.parquet'))
    assert (P.appl_no.values == V.appl_no.values).all() and (P.y.values == V.y.values).all()
    apps = np.unique(P.app.astype(str).values, return_inverse=True)[1]; nA = apps.max() + 1
    y = P.y.values
    for s in (P.p_champ.values, V.p_champ.values, P.p_M0.values): check(s, y)
    W0, W1, WM = WA(P.p_champ.values, y, apps), WA(V.p_champ.values, y, apps), WA(P.p_M0.values, y, apps)
    sk = lambda a, b: (a - b) / (1 - b); one = np.ones(nA)
    m0 = WM.auc(one); s0, s1 = sk(W0.auc(one), m0), sk(W1.auc(one), m0)
    D = []
    for r in range(B):
        w = np.bincount(rng.integers(0, nA, nA), minlength=nA).astype(float); m = WM.auc(w)
        D.append(sk(W1.auc(w), m) - sk(W0.auc(w), m))
    D = np.array(D)
    res = dict(skill_P0=s0, skill_V=s1, delta=s1 - s0, delta_ci95=[float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))],
               n=int(len(y)), pos=int(y.sum()), n_clusters=int(nA), M0_same=bool(np.allclose(P.p_M0.values, V.p_M0.values)))
    out(json.dumps(res))
json.dump(res, open(os.path.join(C, f'paired_{trk}_{VAR}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
