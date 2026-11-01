# -*- coding: utf-8 -*-
"""C05-T2 공용: 경로·읽기 전용 로더·가중 AUC·군집 부트스트랩. 쓰기는 C05-T2 폴더만."""
import os, re, sqlite3, time, json
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
N4 = os.path.join(MC, r"N4_특허행태_기업폐업조기경보")
N6 = os.path.join(MC, r"N6_정부RnD기업_첫특허전환")
N7 = os.path.join(MC, r"N7_특허기업_정부RnD첫수행")
T1 = os.path.join(MC, r"T1_산업대기실")
OUT = os.path.join(BASE, r"분석\검증루프_20261012\C05\C05-T2")
CACHE = os.path.join(OUT, "cache")
REG = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
os.makedirs(CACHE, exist_ok=True)

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def skill(a, b):
    return (a - b) / (1 - b)


def wauc(y, s, w=None):
    """가중 AUC(동점 0.5). w=None이면 보통 AUC."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    # 동점 묶음
    brk = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(brk) - 1
    wp = np.bincount(gid, weights=w * y); wn = np.bincount(gid, weights=w * (~y))
    cn = np.cumsum(wn) - wn  # 이 묶음보다 낮은 음성 가중합
    num = (wp * (cn + 0.5 * wn)).sum()
    den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else np.nan


def boot(y, pm, pb, cl, w=None, n=1000, seed=0):
    """군집 부트스트랩(다항 재표집 = 군집 중복 횟수 가중). skill·AUC 95% CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True)
    K = len(u); sk, am, ab = [], [], []
    for _ in range(n):
        c = rng.multinomial(K, np.full(K, 1.0 / K))
        ww = w * c[inv]
        if (ww * y).sum() == 0 or (ww * (1 - y)).sum() == 0:
            continue
        a1, a0 = wauc(y, pm, ww), wauc(y, pb, ww)
        sk.append(skill(a1, a0)); am.append(a1); ab.append(a0)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(skill_ci=q(sk), auc_ci=q(am), auc_base_ci=q(ab), n_boot=len(sk))


def score(y, pm, pb, cl, w=None, n=1000, seed=0):
    y = np.asarray(y).astype(float)
    a1, a0 = wauc(y, pm, w), wauc(y, pb, w)
    r = dict(n=int(len(y)), pos=int(y.sum()), rate=float(np.average(y, weights=w)) if len(y) else np.nan,
             auc=a1, auc_base=a0, skill=skill(a1, a0))
    if n and y.sum() >= 5 and (1 - y).sum() >= 5:
        r.update(boot(y, pm, pb, cl, w, n, seed))
    return r


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def tvd(a, b):
    pa = pd.Series(a).fillna("NA").value_counts(normalize=True)
    pb = pd.Series(b).fillna("NA").value_counts(normalize=True)
    ix = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(ix, fill_value=0) - pb.reindex(ix, fill_value=0)).abs().sum())


SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청북", "충청남", "충북", "충남", "전라북", "전북", "전라남", "전남", "경상북", "경북", "경상남", "경남", "제주"]
SIDO_MAP = {"충청북": "충북", "충청남": "충남", "전라북": "전북", "전라남": "전남", "경상북": "경북", "경상남": "경남"}


def sido_of(addr):
    a = str(addr or "").strip()
    for k in SIDO:
        if a.startswith(k):
            return SIDO_MAP.get(k, k)
    return "기타/해외"
