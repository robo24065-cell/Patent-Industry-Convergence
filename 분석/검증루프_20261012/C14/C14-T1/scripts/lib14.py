# -*- coding: utf-8 -*-
r"""C14-T1 공용. 입력은 읽기만(NONCIRC_R8, C10-T2). 쓰기는 C14\C14-T1 안에서만."""
import os, sys, time, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
NC8 = os.path.join(BASE, r"분석\모델경쟁_20260927\T1_산업대기실\NONCIRC_R8\cache")
C10 = os.path.join(BASE, r"분석\검증루프_20261012\C10\C10-T2")
C10C = os.path.join(C10, "cache")
OUT = os.path.join(BASE, r"분석\검증루프_20261012\C14\C14-T1")
CACHE = os.path.join(OUT, "cache"); RES = os.path.join(OUT, "res")
C_ORIG = [2008, 2009, 2010, 2011, 2012]
ARMS = ["P50", "FREE50", "TXT50", "FREEIPC50", "HSC1_50"]
PAIRS = [("P50", "FREE50"), ("P50", "FREEIPC50"), ("P50", "HSC1_50"), ("P50", "TXT50")]
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def safe_path(p):
    assert os.path.abspath(p).startswith(os.path.abspath(OUT)), p
    return p


def mode_map(df, key):
    g = df.groupby([key, "ksic"]).size().reset_index(name="n").sort_values([key, "n", "ksic"], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key).ksic


def build_data():
    """원점별 y, 점수(ARMS), 군집 인덱스(firm, sc, ksic_t, cell_fs, cell_fk). 인덱스는 5원점 공통."""
    f = safe_path(os.path.join(CACHE, "data14.npz"))
    if os.path.exists(f):
        z = np.load(f, allow_pickle=False)
        D = {}
        for c in C_ORIG:
            D[c] = {k: z[f"{k}_{c}"] for k in ["y"] + ARMS + ["firm", "sc", "ksic_t", "cell_fs", "cell_fk"]}
        G = {k: int(z[f"G_{k}"]) for k in ["firm", "sc", "ksic_t", "cell_fs", "cell_fk"]}
        return D, G
    P = pd.read_parquet(os.path.join(NC8, "patents.parquet"), columns=["applicantCd", "sc", "ksic", "ky"])
    P = P[P.ky <= 2012]
    KT = mode_map(P, "sc")
    rows = {}
    for c in C_ORIG:
        a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"])
        p1 = pd.read_parquet(os.path.join(C10C, f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0", "TXT50_p0"])
        p2 = pd.read_parquet(os.path.join(C10C, f"pred_c{c}_k0_FREEIPC50-HSC1_50-P50HSC1_p0.parquet"), columns=["FREEIPC50_p0", "HSC1_50_p0"])
        d = pd.concat([a, p1, p2], axis=1); d.columns = [x[:-3] if x.endswith("_p0") else x for x in d.columns]
        d["ksic_t"] = d.sc.map(KT)
        assert d.ksic_t.notna().all() and d[ARMS].notna().all().all()
        assert not d.duplicated(["applicantCd", "sc"]).any()
        rows[c] = d
    allr = pd.concat(rows.values())
    U = {"firm": {u: i for i, u in enumerate(sorted(allr.applicantCd.unique()))},
         "sc": {u: i for i, u in enumerate(sorted(allr.sc.unique()))},
         "ksic_t": {u: i for i, u in enumerate(sorted(allr.ksic_t.unique()))}}
    fs = allr.applicantCd.astype(str) + "|" + allr.sc
    fk = allr.applicantCd.astype(str) + "|" + allr.ksic_t.astype(str)
    U["cell_fs"] = {u: i for i, u in enumerate(pd.unique(fs))}
    U["cell_fk"] = {u: i for i, u in enumerate(pd.unique(fk))}
    D = {}; save = {}
    for c, d in rows.items():
        e = {"y": d.first_ky.notna().values.astype(np.int8)}
        for arm in ARMS:
            e[arm] = d[arm].values.astype(np.float64)
        e["firm"] = d.applicantCd.map(U["firm"]).values.astype(np.int64)
        e["sc"] = d.sc.map(U["sc"]).values.astype(np.int64)
        e["ksic_t"] = d.ksic_t.map(U["ksic_t"]).values.astype(np.int64)
        e["cell_fs"] = (d.applicantCd.astype(str) + "|" + d.sc).map(U["cell_fs"]).values.astype(np.int64)
        e["cell_fk"] = (d.applicantCd.astype(str) + "|" + d.ksic_t.astype(str)).map(U["cell_fk"]).values.astype(np.int64)
        D[c] = e
        for k, v in e.items():
            save[f"{k}_{c}"] = v
    G = {k: len(v) for k, v in U.items()}
    for k, v in G.items():
        save[f"G_{k}"] = np.array(v)
    np.savez(f, **save)
    return D, G


class AucPrep:
    """C10-T2 b_lib.AucPrep 복사(동일 수학). 가중 AUC = Σ_pos w_p (W_neg(아래)+0.5W_neg(동점)) / (Σw_p Σw_n)."""
    def __init__(self, y, s, cl):
        y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
        neg = ~y
        sn = s[neg]; o = np.argsort(sn, kind="mergesort"); sn = sn[o]
        sp_ = s[y]
        self.lo = np.searchsorted(sn, sp_, side="left"); self.hi = np.searchsorted(sn, sp_, side="right")
        self.neg_cl = [np.asarray(c)[neg][o].astype(np.int32) for c in cl]
        self.pos_cl = [np.asarray(c)[y].astype(np.int32) for c in cl]

    def auc(self, wcs):
        wn = wcs[0][self.neg_cl[0]]
        wp = wcs[0][self.pos_cl[0]]
        for k in range(1, len(wcs)):
            wn = wn * wcs[k][self.neg_cl[k]]; wp = wp * wcs[k][self.pos_cl[k]]
        cum = np.empty(len(wn) + 1); cum[0] = 0; np.cumsum(wn, out=cum[1:])
        below = cum[self.lo]; tied = cum[self.hi] - below
        return float((wp * (below + 0.5 * tied)).sum() / (wp.sum() * cum[-1]))

    def auc_rowweights(self, wn, wp):
        cum = np.empty(len(wn) + 1); cum[0] = 0; np.cumsum(wn, out=cum[1:])
        below = cum[self.lo]; tied = cum[self.hi] - below
        return float((wp * (below + 0.5 * tied)).sum() / (wp.sum() * cum[-1]))


def placement(y, s):
    """AUC와 행별 영향 함수 ψ (양성: (V10−A)/n1, 음성: (V01−A)/n0), 동점 0.5."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    sp = s[y]; sn = np.sort(s[~y]); n1 = len(sp); n0 = len(sn)
    v10 = (np.searchsorted(sn, sp, "left") + 0.5 * (np.searchsorted(sn, sp, "right") - np.searchsorted(sn, sp, "left"))) / n0
    sps = np.sort(sp); sneg = s[~y]
    # V01_j = P(pos > neg_j) + 0.5 P(pos = neg_j)
    gt = n1 - np.searchsorted(sps, sneg, "right"); eq = np.searchsorted(sps, sneg, "right") - np.searchsorted(sps, sneg, "left")
    v01 = (gt + 0.5 * eq) / n1
    A = v10.mean()
    psi = np.empty(len(s)); psi[y] = (v10 - A) / n1; psi[~y] = (v01 - A) / n0
    return A, psi


def vclust(psi, g, G):
    S = np.bincount(g, weights=psi, minlength=G)
    Gn = int((np.bincount(g, minlength=G) > 0).sum())
    return float(Gn / (Gn - 1) * (S ** 2).sum()), Gn
