# -*- coding: utf-8 -*-
r"""C10-T2 공용. 원본 TXT_R9 r9_common.py·c_eval.py에서 필요한 부분만 복사. 원본 폴더는 읽기만, 쓰기는 C10-T2 폴더만."""
import os, sys, time, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
SRC = os.path.join(MC, r"T1_산업대기실\TXT_R9")          # 읽기만
SRC_CACHE = os.path.join(SRC, "cache"); SRC_RES = os.path.join(SRC, "res")
NC8 = os.path.join(MC, r"T1_산업대기실\NONCIRC_R8")       # 읽기만
ALT_KR = os.path.join(MC, r"_shared_report\fusion_advantage\c5_R10\alt_assign\alt_KR.parquet")
REC = os.path.join(MC, r"T3_기술셀부상예측\ESSENTIAL_R7\rec_conc.parquet")
OUT = os.path.join(BASE, r"분석\검증루프_20261012\C10\C10-T2")
CACHE = os.path.join(OUT, "cache"); RES = os.path.join(OUT, "res")
for p in (CACHE, RES):
    os.makedirs(p, exist_ok=True)
assert os.path.abspath(CACHE).startswith(os.path.abspath(OUT))
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


BASEF = ["f_n", "f_rec3", "f_nsc", "f_age", "s_n", "s_grow", "s_nfirm", "s_newrate",
         "p_sec", "p_relmean", "p_relmax", "p_cls", "p_sect", "p_cf"]
CTX = ["dot", "cos", "adopt", "entry", "qmain", "mommain", "sindmom"]
cx = lambda a: [f"{a}_{x}" for x in CTX]
ARMS = {"NONE": BASEF, "P50": BASEF + cx("P50"), "TXT50": BASEF + cx("TXT50"), "FREE50": BASEF + cx("FREE50"),
        "FREEIPC50": BASEF + cx("FIPC"), "HSC1_50": BASEF + cx("HSC1"), "HMG3_50": BASEF + cx("HMG3"),
        "P50FIPC": BASEF + cx("P50") + cx("FIPC"), "P50HSC1": BASEF + cx("P50") + cx("HSC1"),
        "P50HMG3": BASEF + cx("P50") + cx("HMG3")}
PRM = dict(n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=20, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=5, tree_method="hist", nthread=2, eval_metric="auc")
NEG = 0.3
NEED = ["applicantCd", "sc", "t", "first_ky"] + BASEF + cx("P50")
C_ORIG = [2008, 2009, 2010, 2011, 2012]


def load(t, z=False):
    a = pd.read_parquet(os.path.join(NC8, "cache", f"A_o{t}.parquet"), columns=NEED)
    x = pd.read_parquet(os.path.join(SRC_CACHE, f"X_o{t}.parquet"))
    assert (a.applicantCd.values == x.applicantCd.values).all() and (a.sc.values == x.sc.values).all()
    parts = [a, x.drop(columns=["applicantCd", "sc"])]
    if z:
        zz = pd.read_parquet(os.path.join(CACHE, f"Z_o{t}.parquet"))
        assert (a.applicantCd.values == zz.applicantCd.values).all() and (a.sc.values == zz.sc.values).all()
        parts.append(zz.drop(columns=["applicantCd", "sc"]))
    return pd.concat(parts, axis=1)


def topk_hit(y, s, k):
    o = np.argsort(-np.asarray(s))[:k]
    return float(np.asarray(y)[o].mean())


def auc_w(y, s):
    y = np.asarray(y, dtype=np.int8); s = np.asarray(s, dtype=np.float64)
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    _, idx = np.unique(s, return_index=True)
    wp = np.add.reduceat((y == 1).astype(float), idx); wn = np.add.reduceat((y == 0).astype(float), idx)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def prep(y, s):
    o = np.argsort(s, kind="mergesort"); _, idx = np.unique(s[o], return_index=True)
    return o, idx, (y[o] == 1)


def fast(pr, w):
    o, idx, yp = pr; w = w[o]
    wp = np.add.reduceat(w * yp, idx); wn = np.add.reduceat(w * ~yp, idx)
    cn = np.cumsum(wn) - wn
    return (wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


def train_set(c, k, z=False):
    """공식 c_eval.run과 같은 학습 집합. k=0이면 음성 표집 seed 1000+t(공식), k=1 → 2000+t, k=2 → 3000+t."""
    tr = []
    for t in (c - 6, c - 5, c - 4):
        if t < 2002:
            continue
        d = load(t, z)
        d["y"] = (d.first_ky <= c).astype(np.int8)
        rng = np.random.default_rng((k + 1) * 1000 + t)
        d = d[(d.y == 1) | (rng.random(len(d)) < NEG)]
        d["w"] = np.where(d.y == 1, 1.0, 1 / NEG)
        tr.append(d)
    return pd.concat(tr, ignore_index=True)
