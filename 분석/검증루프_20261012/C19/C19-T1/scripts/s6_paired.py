# -*- coding: utf-8 -*-
r"""C19-T1 s6: 짝지은 군집 부트스트랩(B=1000, seed 0) Δskill = skill(V) − skill(P0) (C18-T2 s3g_paired.py T3 부분 사본) + SB3 강화 기준선 재적합·CI(seed 11)
 + 노출 큰 셀(exposure_cells.csv hi) 순위 백분위 변화·상위 10% 적중 변화.
 사용: python s6_paired.py <VAR> [<VAR> ...] → cache/paired_<VAR>.json"""
import os, sys, json, time, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
B = 1000; t = time.time()
T = os.path.join(C, 't3res')


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


sk = lambda a, b: (a - b) / (1 - b)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load(v, mode):
    if mode == 'final':
        return pd.read_parquet(os.path.join(T, f'preds_{v}_final.parquet'))
    return pd.concat([pd.read_parquet(os.path.join(T, f'preds_{v}_dev_{o}.parquet')) for o in ('2010-2011-2012-2013', '2014-2015-2016-2017-2018')])


def hkey(cells): return np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in cells], dtype=np.int64)


def label_block(g):
    k = max(1, int(round(0.10 * len(g)))); o = np.lexsort((hkey(g.cell.values), -(g.kr_fut - g.kr_W).values, -g.y.values))
    yb = np.zeros(len(g), int); yb[o[:k]] = 1; return yb


SB3F = ['kr_mom', 'kr_mom_l1', 'kr_mom_l2', 'kr_mom5', 'kr_g1', 'kr_acc', 'kr_lvl', 'kr_last', 'kr_slope5', 'kr_zero_yrs', 'kr_share', 'kr_share_chg']


def sb3(var, P):
    cv = 'P0' if var in ('P0', 'VJ4a', 'VJ4b') else var
    cache = os.path.join(C, f't3_{cv}_fin')
    TR = []
    for tt in range(2005, 2019):
        g = pd.read_parquet(os.path.join(cache, f'feat_{tt}.parquet')); g = g[g.tot_W >= 50].reset_index(drop=True); g['yb'] = label_block(g); TR.append(g)
    TR = pd.concat(TR, ignore_index=True)
    TE = pd.read_parquet(os.path.join(cache, 'feat_2021.parquet')); TE = TE[TE.tot_W >= 50]
    TE = TE.merge(P[['cell', 't0', 'p', 'p_base', 'yb']], on=['cell', 't0'], how='inner'); assert len(TE) == len(P), (len(TE), len(P))
    bst = xgb.train(dict(objective='binary:logistic', max_depth=2, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, nthread=2, verbosity=0, seed=0),
                    xgb.DMatrix(TR[SB3F].values, label=TR.yb.values), 450)
    TE['sb3'] = bst.predict(xgb.DMatrix(TE[SB3F].values))
    am, a3 = roc_auc_score(TE.yb, TE.p), roc_auc_score(TE.yb, TE.sb3)
    # 공식 ci_sb_t3.py 방식(seed 11, mg 재표본, roc_auc_score)
    ums = np.unique(TE.mg.values); gi = pd.Series(np.arange(len(TE))).groupby(TE.mg.values).apply(lambda s: s.values).to_dict()
    rng = np.random.default_rng(11); R = []
    y = TE.yb.values; pm = TE.p.values; p3 = TE.sb3.values
    for _ in range(1000):
        smp = rng.choice(ums, len(ums)); ii = np.concatenate([gi[u] for u in smp])
        if y[ii].min() == y[ii].max(): continue
        a0 = roc_auc_score(y[ii], p3[ii]); R.append(sk(roc_auc_score(y[ii], pm[ii]), a0))
    return dict(auc_sb3=float(a3), skill_vs_sb3=float(sk(am, a3)), ci95=q(R), n_te=int(len(TE))), TE[['cell', 'sb3']]


res_all = {}
EXP = pd.read_csv(os.path.join(W, 'exposure_cells.csv'), encoding='utf-8-sig')[['cell', 'hi', 'e']]
for VAR in sys.argv[1:]:
    rng = np.random.default_rng(0); res = {}
    for mode in ('final', 'dev'):
        try:
            P, V = load('P0', mode), load(VAR, mode)
        except FileNotFoundError as e:
            res[mode] = f'missing {os.path.basename(str(e))[:80]}'; continue
        mgs = np.array(sorted(set(P.mg) | set(V.mg))); mi = {m: i for i, m in enumerate(mgs)}
        bl = []
        for o in sorted(P.t0.unique()):
            pp, vv = P[P.t0 == o], V[V.t0 == o]
            for d in (pp, vv): check(d.p.values, d.yb.values)
            gp = pp.mg.map(mi).values; gv = vv.mg.map(mi).values
            bl.append([WA(pp.p.values, pp.yb.values, gp), WA(pp.p_base.values, pp.yb.values, gp), WA(vv.p.values, vv.yb.values, gv), WA(vv.p_base.values, vv.yb.values, gv)])
        one = np.ones(len(mgs))
        pt_ = np.array([[sk(b[0].auc(one), b[1].auc(one)), sk(b[2].auc(one), b[3].auc(one))] for b in bl]).mean(0)
        D = []
        for r in range(B):
            w = np.bincount(rng.integers(0, len(mgs), len(mgs)), minlength=len(mgs)).astype(float)
            s0 = np.mean([sk(b[0].auc(w), b[1].auc(w)) for b in bl]); s1 = np.mean([sk(b[2].auc(w), b[3].auc(w)) for b in bl])
            D.append((s0, s1, s1 - s0))
        D = np.array(D)
        res[mode] = dict(skill_P0=float(pt_[0]), skill_V=float(pt_[1]), delta=float(pt_[1] - pt_[0]), delta_ci95=q(D[:, 2]), skill_P0_ci95=q(D[:, 0]), skill_V_ci95=q(D[:, 1]),
                         n_cells_P0=int(len(P)), n_cells_V=int(len(V)), n_evt_V=int(V.yb.sum()), n_clusters=int(len(mgs)), origins=[int(x) for x in sorted(P.t0.unique())])
        if mode == 'final':
            # 노출 큰 셀: 순위 백분위(전체 시험 셀 안) 변화, 상위 10% 적중
            m = P[['cell', 'p', 'yb']].merge(V[['cell', 'p', 'yb']], on='cell', suffixes=('_0', '_v'), how='inner').merge(EXP, on='cell', how='left')
            m['r0'] = m.p_0.rank(pct=True); m['rv'] = m.p_v.rank(pct=True)
            k0 = int(round(0.1 * len(P))); kv = int(round(0.1 * len(V)))
            top0 = set(P.nlargest(k0, 'p').cell); topv = set(V.nlargest(kv, 'p').cell)
            hi = m[m.hi == True]
            res['exposed'] = dict(n_common=int(len(m)), n_hi=int(len(hi)), med_abs_drank_hi=float((hi.rv - hi.r0).abs().median()), med_abs_drank_rest=float((m[m.hi != True].rv - m[m.hi != True].r0).abs().median()),
                                  mean_drank_hi=float((hi.rv - hi.r0).mean()),
                                  top10_hits_hi_P0=int(P[P.cell.isin(hi.cell) & P.cell.isin(top0)].yb.sum()), top10_hits_hi_V=int(V[V.cell.isin(hi.cell) & V.cell.isin(topv)].yb.sum()),
                                  top10_hits_all_P0=int(P[P.cell.isin(top0)].yb.sum()), top10_hits_all_V=int(V[V.cell.isin(topv)].yb.sum()),
                                  cells_only_P0=int(len(set(P.cell) - set(V.cell))), cells_only_V=int(len(set(V.cell) - set(P.cell))))
            res['sb3'], _ = sb3(VAR, V)
            if VAR == 'P0':
                pass
        out(VAR, mode, json.dumps(res[mode], ensure_ascii=False)[:600], round(time.time() - t))
    if 'exposed' in res: out(VAR, 'exposed', res['exposed'], 'sb3', res['sb3'])
    json.dump(res, open(os.path.join(C, f'paired_{VAR}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    res_all[VAR] = res
