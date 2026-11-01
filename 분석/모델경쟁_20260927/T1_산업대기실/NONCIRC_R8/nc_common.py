# -*- coding: utf-8 -*-
r"""T1 산업 대기실 — 순환 없는 필수성 검증(R8) 공용 함수.
쓰기는 이 폴더(T1_산업대기실\NONCIRC_R8)에만. 원본 DB·_shared·동결 트랙 파일은 읽기만.
특허 표: 제공범위 KR 등록특허(registry 원출원인 코드, 법인=코드 1로 시작) + 메타의 메인/전체 IPC
         + 제공 KSIC(50) + 제공 KSIC의 연계표 그룹(27, grp) + 공식 연계표 배정 그룹(27, cw; label_validity 캐시).
"""
import os, sys, time, re, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
OUT = os.path.join(MC, r"T1_산업대기실\NONCIRC_R8")
CACHE = os.path.join(OUT, "cache")
EXT = os.path.join(MC, r"_shared\ext")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
os.makedirs(CACHE, exist_ok=True)
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def sc_of(ipc):
    s = re.sub(r"\s+", "", str(ipc or "")).upper()
    m = re.match(r"^([A-H]\d{2}[A-Z])", s)
    return m.group(1) if m else None


def load_patents():
    """1행 = (법인 출원인, 출원). 열: applicantCd, appl_no, ay, ky(가시 연도), sc(메인 서브클래스), sc_all(전체 서브클래스 '|'),
    ksic(제공 50), pgrp(제공 라벨의 연계표 그룹 27), cw(공식 연계표 배정 그룹 27, 미매칭 None)"""
    f = os.path.join(CACHE, "patents.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    import duckdb
    af = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\cache\app_firm.parquet"))
    af = af[(af.src == "registry") & af.applicantCd.astype(str).str.startswith("1")]
    af = af[["applicantCd", "appl_no", "ay", "known_date", "ksic"]].drop_duplicates(["applicantCd", "appl_no"])
    af["ky"] = (af.known_date // 10000).astype("Int64")
    m = duckdb.sql(f"select appl_no, main_ipc, ipc_all from read_parquet('{META}') where cc='KR'").df()
    m["sc"] = m.main_ipc.map(sc_of)
    m["sc_all"] = m.ipc_all.fillna("").map(lambda s: "|".join(sorted({x for x in (sc_of(t) for t in s.split("|")) if x})))
    cw = pd.read_parquet(os.path.join(MC, r"_shared_report\label_validity\cache_kr_cw.parquet"), columns=["appl_no", "grp", "cw"])
    P = af.merge(m[["appl_no", "sc", "sc_all"]], on="appl_no", how="left").merge(cw, on="appl_no", how="left")
    P = P.rename(columns={"grp": "pgrp"})
    P["ay"] = P.ay.astype(int)
    P["ky"] = P.ky.astype(int)
    P = P[P.sc.notna()].drop(columns="known_date")
    P.to_parquet(f, index=False)
    return P


# ---------- 평가 함수 ----------
def auc_w(y, s, w=None):
    """가중 AUC(동점 평균). y 0/1, s 점수, w 행 가중."""
    y = np.asarray(y, dtype=np.int8); s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    # 동점 묶음
    _, idx, cnt = np.unique(s, return_index=True, return_counts=True)
    wp = np.add.reduceat(w * (y == 1), idx); wn = np.add.reduceat(w * (y == 0), idx)
    cn = np.cumsum(wn) - wn  # 이전 묶음 음성 누적
    num = (wp * (cn + 0.5 * wn)).sum()
    den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else np.nan


def cluster_boot_delta(y, s1, s2, cl, B=300, seed=0):
    """법인 군집 Poisson 부트스트랩으로 ΔAUC(s1−s2) 95% CI."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    y = np.asarray(y, dtype=np.int8)

    def prep(s):
        s = np.asarray(s, dtype=np.float64); o = np.argsort(s, kind="mergesort")
        _, idx = np.unique(s[o], return_index=True)
        return o, idx, (y[o] == 1)

    def fast(pr, w):
        o, idx, yp = pr; w = w[o]
        wp = np.add.reduceat(w * yp, idx); wn = np.add.reduceat(w * ~yp, idx)
        cn = np.cumsum(wn) - wn
        return (wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())
    p1, p2 = prep(s1), prep(s2)
    d = []
    for _ in range(B):
        wc = rng.poisson(1.0, len(u)).astype(np.float64)
        w = wc[inv]
        d.append(fast(p1, w) - fast(p2, w))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(np.array(d) > 0))


def topk_hit(y, s, k):
    o = np.argsort(-np.asarray(s))[:k]
    return float(np.asarray(y)[o].mean())
