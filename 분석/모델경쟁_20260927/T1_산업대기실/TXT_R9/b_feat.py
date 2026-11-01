# -*- coding: utf-8 -*-
"""[2] T1 라벨 무관 결과 A: TXT·TXT-FREE 5자리 라벨로 R8 a_build.py와 같은 7개 산업 맥락 피처를 만든다.
행(법인×서브클래스)은 NONCIRC_R8 cache/A_o<t>.parquet의 키를 그대로 쓴다(읽기만). P50 피처도 다시 계산해 R8 값과 같은지 점검.
사용: python b_feat.py 2002 2003 ... → cache/X_o<t>.parquet (applicantCd, sc, TXT50_*, FREE50_*)"""
from r9_common import *
import scipy.sparse as sp

P = pd.read_parquet(os.path.join(NC8, "cache", "patents.parquet"))
T = pd.read_parquet(os.path.join(CACHE, "kr_txt.parquet"), columns=["appl_no", "k_txt"]).merge(
    pd.read_parquet(os.path.join(CACHE, "kr_txtfree.parquet")), on="appl_no")
P = P.merge(T, on="appl_no", how="left")
LAB = {"P50": "ksic", "TXT50": "k_txt", "FREE50": "k_free"}
CTX = ["dot", "cos", "adopt", "entry", "qmain", "mommain", "sindmom"]


def rownorm(M):
    s = np.asarray(M.sum(1)).ravel(); s[s == 0] = 1
    return sp.diags(1 / s) @ M


def build(t):
    V = P[P.ky <= t]
    firms_all = np.sort(V.applicantCd.unique()); fi = {f: i for i, f in enumerate(firms_all)}
    SC = np.sort(P.sc.unique()); si = {s: i for i, s in enumerate(SC)}
    nF, nS = len(firms_all), len(SC)
    C = sp.csr_matrix((np.ones(len(V)), (V.applicantCd.map(fi).values, V.sc.map(si).values)), shape=(nF, nS))
    Hd = (C > 0).astype(np.float64).toarray()
    fy = V.groupby(["applicantCd", "sc"]).ay.min().reset_index()
    Hn = sp.csr_matrix(((fy.ay >= t - 3).astype(float).values, (fy.applicantCd.map(fi).values, fy.sc.map(si).values)), shape=(nF, nS)).toarray()
    A0 = pd.read_parquet(os.path.join(NC8, "cache", f"A_o{t}.parquet"), columns=["applicantCd", "sc"] + [f"P50_{x}" for x in CTX])
    ff = A0.applicantCd.map(fi).values; ss = A0.sc.map(si).values
    assert not np.isnan(ff.astype(float)).any() and not np.isnan(ss.astype(float)).any()
    ff = ff.astype(int); ss = ss.astype(int)
    D = A0[["applicantCd", "sc"]].copy()
    chk = {}
    for arm, col in LAB.items():
        Vl = V[V[col].notna()]
        K = np.sort(Vl[col].unique()); ki = {k: i for i, k in enumerate(K)}
        M = rownorm(sp.csr_matrix((np.ones(len(Vl)), (Vl.applicantCd.map(fi).values, Vl[col].map(ki).values)), shape=(nF, len(K)))).toarray()
        Q = rownorm(sp.csr_matrix((np.ones(len(Vl)), (Vl.sc.map(si).values, Vl[col].map(ki).values)), shape=(nS, len(K)))).toarray()
        nm = np.linalg.norm(M, axis=1); nq = np.linalg.norm(Q, axis=1)
        dot = np.einsum("ij,ij->i", M[ff], Q[ss])
        F = {}
        F["dot"] = dot
        F["cos"] = dot / np.maximum(nm[ff] * nq[ss], 1e-12)
        mden = M.sum(0); mden[mden == 0] = 1
        Aa = (M.T @ Hd) / mden[:, None]
        En = (M.T @ Hn) / mden[:, None]
        F["adopt"] = np.einsum("ij,ij->i", M[ff], Aa[:, ss].T)
        F["entry"] = np.einsum("ij,ij->i", M[ff], En[:, ss].T)
        main = M.argmax(1); hasm = M.sum(1) > 0
        F["qmain"] = np.where(hasm[ff], Q[ss, main[ff]], 0)
        g1 = Vl[Vl.ay.between(t - 2, t)][col].value_counts(); g0 = Vl[Vl.ay.between(t - 5, t - 3)][col].value_counts()
        kg = np.log1p(pd.Series(K).map(g1).fillna(0).values) - np.log1p(pd.Series(K).map(g0).fillna(0).values)
        F["mommain"] = np.where(hasm[ff], kg[main[ff]], 0)
        F["sindmom"] = Q[ss] @ kg
        if arm == "P50":
            chk = {x: float(np.max(np.abs(F[x].astype(np.float32) - A0[f"P50_{x}"].values))) for x in CTX}
        else:
            for x in CTX:
                D[f"{arm}_{x}"] = F[x].astype(np.float32)
    D.to_parquet(os.path.join(CACHE, f"X_o{t}.parquet"), index=False)
    log(t, "rows", len(D), "P50 재계산 최대 오차", chk)


if __name__ == "__main__":
    for t in map(int, sys.argv[1:]):
        build(t)
