# -*- coding: utf-8 -*-
"""R9 N9·N6 순환 없는 검증 — 공용 함수. 쓰기는 이 폴더만. 동결 트랙 파일은 읽기만(필요 모듈은 sys.path로 import, __pycache__ 생성 금지)."""
import os, sys, time, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
OUT = os.path.join(MC, r"_shared_report\noncirc_n9n6_R9")
CACHE = os.path.join(OUT, "cache"); os.makedirs(CACHE, exist_ok=True)
PRED = os.path.join(OUT, "preds"); os.makedirs(PRED, exist_ok=True)
N9 = os.path.join(MC, "N9_정부RnD과제_특허성과예측")
N9E = os.path.join(N9, "ESSENTIAL_R7")
N6 = os.path.join(MC, "N6_정부RnD기업_첫특허전환")
N6E = os.path.join(N6, "ESSENTIAL_R7")
R8 = os.path.join(MC, "_ideas_R8")
EXT = os.path.join(MC, r"_shared\ext")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
DEV_MAX_APPL = 2021
SECTIONS = ["G", "H", "C", "A", "B"]
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def wauc(p, y, w=None):
    """가중 Mann-Whitney AUC, 동점 0.5 처리."""
    p = np.asarray(p, float); y = np.asarray(y).astype(bool)
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    o = np.argsort(p, kind="mergesort"); ps = p[o]; ys = y[o]; ws = w[o]
    # 동점 묶음
    brk = np.r_[True, ps[1:] != ps[:-1]]
    gid = np.cumsum(brk) - 1
    wn = np.bincount(gid, weights=ws * (~ys)); wp = np.bincount(gid, weights=ws * ys)
    cn = np.cumsum(wn) - wn            # 아래쪽 음성 가중합
    num = (wp * (cn + 0.5 * wn)).sum(); den = wp.sum() * wn.sum()
    return num / den if den > 0 else np.nan


class FastWAUC:
    """같은 점수 벡터로 여러 가중치를 반복 평가(부트스트랩)할 때 정렬을 한 번만."""
    def __init__(self, p, y):
        p = np.asarray(p, float); y = np.asarray(y).astype(bool)
        o = np.argsort(p, kind="mergesort"); ps = p[o]
        brk = np.r_[True, ps[1:] != ps[:-1]]
        self.gid = np.cumsum(brk) - 1; self.o = o; self.ys = y[o]; self.ng = self.gid[-1] + 1

    def __call__(self, w):
        ws = np.asarray(w, float)[self.o]
        wn = np.bincount(self.gid, weights=ws * (~self.ys), minlength=self.ng)
        wp = np.bincount(self.gid, weights=ws * self.ys, minlength=self.ng)
        cn = np.cumsum(wn) - wn
        den = wp.sum() * wn.sum()
        return (wp * (cn + 0.5 * wn)).sum() / den if den > 0 else np.nan


def verdict(mean, lo, hi, npos, nneg, nor):
    need = int(np.ceil(0.6 * nor - 1e-9))
    if mean > 0 and lo > 0 and npos >= need:
        return "제공 우위"
    if mean < 0 and hi < 0 and nneg >= need:
        return "제공 열위"
    return "같다"


def cluster_boot(tabs, arms_pairs, cluster_col, n=200, seed=0):
    """tabs: {origin: DataFrame(y, cluster, p_<arm>...)}. 같은 군집 가중치를 모든 원점에 적용.
    반환: {(a,b): dict(mean, ci, per_origin, npos, nneg)} 와 팔별 평균 AUC."""
    rng = np.random.default_rng(seed)
    allc = pd.Index(sorted(set().union(*[set(t[cluster_col].astype(str)) for t in tabs.values()])))
    G = len(allc)
    idx = {v: allc.get_indexer(t[cluster_col].astype(str)) for v, t in tabs.items()}
    arms = sorted({c[2:] for t in tabs.values() for c in t.columns if c.startswith("p_")})
    F = {v: {a: FastWAUC(t[f"p_{a}"].values, t.y.values) for a in arms if f"p_{a}" in t} for v, t in tabs.items()}
    point = {v: {a: F[v][a](np.ones(len(t))) for a in F[v]} for v, t in tabs.items()}
    boots = {pr: [] for pr in arms_pairs}
    for _ in range(n):
        cnt = rng.multinomial(G, np.full(G, 1.0 / G)).astype(float)
        per = {pr: [] for pr in arms_pairs}
        for v, t in tabs.items():
            w = cnt[idx[v]]
            au = {a: F[v][a](w) for a in F[v]}
            for pr in arms_pairs:
                per[pr].append(au[pr[0]] - au[pr[1]])
        for pr in arms_pairs:
            boots[pr].append(np.nanmean(per[pr]))
    res = {}
    for pr in arms_pairs:
        d = [point[v][pr[0]] - point[v][pr[1]] for v in sorted(tabs)]
        lo, hi = np.percentile(boots[pr], [2.5, 97.5])
        m = float(np.mean(d)); npos = int(sum(x > 0 for x in d)); nneg = int(sum(x < 0 for x in d))
        res[f"{pr[0]}-{pr[1]}"] = dict(mean=m, ci=[float(lo), float(hi)], per_origin=[float(x) for x in d], npos=npos, nneg=nneg,
                                       n_origins=len(d), verdict=verdict(m, lo, hi, npos, nneg, len(d)))
    auc_mean = {a: float(np.mean([point[v][a] for v in sorted(tabs) if a in point[v]])) for a in arms}
    auc_by = {a: [float(point[v][a]) for v in sorted(tabs) if a in point[v]] for a in arms}
    return res, auc_mean, auc_by
