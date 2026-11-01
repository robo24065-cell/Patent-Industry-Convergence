# -*- coding: utf-8 -*-
"""(1) 라벨 무관 결과 A: 법인이 (t, t+4]에 '그 법인에 처음인' IPC 메인 서브클래스로 출원(이후 제공범위 등록)하는가.
결과·위험집합·모집단은 KSIC 라벨을 전혀 쓰지 않는다. 피처의 '산업 맥락'만 라벨 체계(P50/P27/CW27)별로 만든다.
사용: python a_build.py 2002 2003 ...   → cache/A_o<t>.parquet
- 모집단: 기준시점 t 연말까지 가시(ky≤t)한 제공범위 KR 등록특허 ≥2건 법인
- 후보 서브클래스: 최근 5년(t-4..t 출원, ky≤t) 가시 특허 ≥30건
- 위험집합: t까지 출원된 그 법인 특허 중 메인 서브클래스 = s 인 것 0건(최종 등록 기준 정의, T1과 같은 방식)
- 결과: (t, t+4] 출원, 메인 서브클래스 s → first_ky(가장 이른 가시 연도) 저장. 학습 레이블은 평가 원점 c 이전 가시분(first_ky≤c)만.
"""
from nc_common import *
import scipy.sparse as sp

P = load_patents()
LAB = {"P50": "ksic", "P27": "pgrp", "CW27": "cw"}


def rownorm(M):
    s = np.asarray(M.sum(1)).ravel(); s[s == 0] = 1
    return sp.diags(1 / s) @ M


def build(t):
    V = P[P.ky <= t]                                   # 가시 특허
    firms_all = np.sort(V.applicantCd.unique()); fi = {f: i for i, f in enumerate(firms_all)}
    SC = np.sort(P.sc.unique()); si = {s: i for i, s in enumerate(SC)}
    nF, nS = len(firms_all), len(SC)
    vf = V.applicantCd.map(fi).values; vs = V.sc.map(si).values
    # 법인×서브클래스 (메인) 건수
    C = sp.csr_matrix((np.ones(len(V)), (vf, vs)), shape=(nF, nS))
    H = (C > 0).astype(np.float64)
    W = rownorm(C)
    # 보조 IPC
    sec = V[["applicantCd", "sc", "sc_all"]].copy()
    sec["s2"] = sec.sc_all.str.split("|"); sec = sec.explode("s2")
    sec = sec[(sec.s2 != sec.sc) & sec.s2.isin(si)]
    Csec = sp.csr_matrix((np.ones(len(sec)), (sec.applicantCd.map(fi).values, sec.s2.map(si).values)), shape=(nF, nS))
    # 서브클래스 동시출현 관련도(특허 내 전체 IPC 서브클래스)
    pp = V[["appl_no", "sc_all"]].drop_duplicates("appl_no")
    ex = pp.assign(s2=pp.sc_all.str.split("|")).explode("s2")
    ex = ex[ex.s2.isin(si)]
    pi = pd.factorize(ex.appl_no)[0]
    X = sp.csr_matrix((np.ones(len(ex)), (pi, ex.s2.map(si).values)), shape=(pi.max() + 1, nS))
    X.data[:] = 1
    CO = (X.T @ X).toarray(); dg = np.sqrt(np.diag(CO)); dg[dg == 0] = 1
    R = CO / dg[:, None] / dg[None, :]; np.fill_diagonal(R, 0)
    # 라벨 무관 IPC 동료(협업필터): A_ipc[s', s] = Σ_f W[f,s'] H[f,s] / Σ_f W[f,s']
    Wd = W.toarray(); Hd = H.toarray()
    den = Wd.sum(0); den[den == 0] = 1
    A_ipc = (Wd.T @ Hd) / den[:, None]; np.fill_diagonal(A_ipc, 0)
    # 법인 첫 가시 서브클래스 연도(최근 진입)
    fy = V.groupby(["applicantCd", "sc"]).ay.min().reset_index()
    Hn = sp.csr_matrix(((fy.ay >= t - 3).astype(float).values, (fy.applicantCd.map(fi).values, fy.sc.map(si).values)), shape=(nF, nS)).toarray()
    # 모집단·후보
    nvis = np.asarray(C.sum(1)).ravel()
    pop = np.where(nvis >= 2)[0]
    rec = V[V.ay >= t - 4].sc.value_counts()
    cand = np.array(sorted(si[s] for s in rec[rec >= 30].index))
    # 위험집합: t까지 출원(최종 등록) 메인 s 0건
    E = P[(P.ay <= t) & P.applicantCd.isin(set(firms_all[pop]))]
    have = set(zip(E.applicantCd.map(fi).values, E.sc.map(si).values))
    ff, ss = np.meshgrid(pop, cand, indexing="ij"); ff = ff.ravel(); ss = ss.ravel()
    keep = np.array([(a, b) not in have for a, b in zip(ff, ss)])
    ff, ss = ff[keep], ss[keep]
    D = pd.DataFrame({"applicantCd": firms_all[ff], "sc": SC[ss], "t": t})
    # 결과
    Y = P[(P.ay > t) & (P.ay <= t + 4)].groupby(["applicantCd", "sc"]).ky.min().rename("first_ky").reset_index()
    D = D.merge(Y, on=["applicantCd", "sc"], how="left")
    # ---- 라벨 무관 피처(BASE)
    vrec3 = V[V.ay >= t - 2].groupby("applicantCd").size()
    fmin = V.groupby("applicantCd").ay.min()
    nsc = np.asarray(H.sum(1)).ravel()
    D["f_n"] = np.log1p(nvis[ff]); D["f_rec3"] = np.log1p(D.applicantCd.map(vrec3).fillna(0).values)
    D["f_nsc"] = nsc[ff]; D["f_age"] = t - D.applicantCd.map(fmin).values
    sN = np.asarray(C.sum(0)).ravel()
    g1 = V[V.ay.between(t - 2, t)].sc.value_counts(); g0 = V[V.ay.between(t - 5, t - 3)].sc.value_counts()
    sg = np.log1p(pd.Series(SC).map(g1).fillna(0).values) - np.log1p(pd.Series(SC).map(g0).fillna(0).values)
    sF = np.asarray(H.sum(0)).ravel(); sNew = Hn.sum(0)
    D["s_n"] = np.log1p(sN[ss]); D["s_grow"] = sg[ss]; D["s_nfirm"] = np.log1p(sF[ss])
    D["s_newrate"] = (sNew / np.maximum(sF, 1))[ss]
    D["p_sec"] = np.log1p(np.asarray(Csec[ff, ss]).ravel())
    D["p_relmean"] = np.einsum("ij,ij->i", Wd[ff], R[:, ss].T)
    Hb = Hd.astype(bool)
    D["p_relmax"] = np.array([R[Hb[a], b].max() if Hb[a].any() else 0 for a, b in zip(ff, ss)])
    cls = np.array([s[:3] for s in SC]); sect = np.array([s[0] for s in SC])
    Wc = pd.DataFrame(Wd, columns=cls).T.groupby(level=0).sum().T  # 법인×IPC 클래스
    Wsec = pd.DataFrame(Wd, columns=sect).T.groupby(level=0).sum().T
    D["p_cls"] = Wc.values[ff, Wc.columns.get_indexer(cls[ss])]
    D["p_sect"] = Wsec.values[ff, Wsec.columns.get_indexer(sect[ss])]
    D["p_cf"] = np.einsum("ij,ij->i", Wd[ff], A_ipc[:, ss].T)
    # ---- 라벨 체계별 산업 맥락 피처
    for arm, col in LAB.items():
        Vl = V[V[col].notna()]
        K = np.sort(Vl[col].unique()); ki = {k: i for i, k in enumerate(K)}
        M = rownorm(sp.csr_matrix((np.ones(len(Vl)), (Vl.applicantCd.map(fi).values, Vl[col].map(ki).values)), shape=(nF, len(K)))).toarray()
        Q = rownorm(sp.csr_matrix((np.ones(len(Vl)), (Vl.sc.map(si).values, Vl[col].map(ki).values)), shape=(nS, len(K)))).toarray()
        nm = np.linalg.norm(M, axis=1); nq = np.linalg.norm(Q, axis=1)
        dot = np.einsum("ij,ij->i", M[ff], Q[ss])
        D[f"{arm}_dot"] = dot
        D[f"{arm}_cos"] = dot / np.maximum(nm[ff] * nq[ss], 1e-12)
        mden = M.sum(0); mden[mden == 0] = 1
        A = (M.T @ Hd) / mden[:, None]          # 산업 k 법인의 s 보유율(소프트)
        En = (M.T @ Hn) / mden[:, None]         # 산업 k 법인의 최근 4년 s 신규 진입률
        D[f"{arm}_adopt"] = np.einsum("ij,ij->i", M[ff], A[:, ss].T)
        D[f"{arm}_entry"] = np.einsum("ij,ij->i", M[ff], En[:, ss].T)
        main = M.argmax(1); hasm = M.sum(1) > 0
        D[f"{arm}_qmain"] = np.where(hasm[ff], Q[ss, main[ff]], 0)
        g1 = Vl[Vl.ay.between(t - 2, t)][col].value_counts(); g0 = Vl[Vl.ay.between(t - 5, t - 3)][col].value_counts()
        kg = np.log1p(pd.Series(K).map(g1).fillna(0).values) - np.log1p(pd.Series(K).map(g0).fillna(0).values)
        D[f"{arm}_mommain"] = np.where(hasm[ff], kg[main[ff]], 0)
        D[f"{arm}_sindmom"] = Q[ss] @ kg
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D.to_parquet(os.path.join(CACHE, f"A_o{t}.parquet"), index=False)
    log(t, "firms_vis", nF, "pop", len(pop), "cand", len(cand), "pairs", len(D), "pos(final)", int(D.first_ky.notna().sum()),
        "rate", round(float(D.first_ky.notna().mean()), 4))


if __name__ == "__main__":
    for t in map(int, sys.argv[1:]):
        build(t)
