# -*- coding: utf-8 -*-
"""(3) 대칭 교차평가 2×2 — 산업 진입 결과를 두 라벨 체계로 각각 정의하고, 두 라벨 체계 피처로 각각 맞힌다.
해상도는 공통 27개 연계표 그룹(제공 라벨은 50개를 그룹으로 접은 pgrp, 연계표는 메인 IPC 배정 cw).
결과 Y_D (D∈{P,C}): (t, t+4] 출원 특허 중 D 그룹 = g 인 첫 특허(이후 제공범위 등록). 위험집합: t까지 출원된 D 그룹 g 특허 0건.
피처 F_L (L∈{P,C}): 법인의 L 그룹 구성·L 동시보유·L 동료 신규진입·그룹 g의 L-IPC 프로파일 적합도·L 그룹 통계.
사용: python c_build.py 2002 ... → cache/C_o<t>_<D>.parquet
"""
from nc_common import *
import scipy.sparse as sp

P = load_patents()
DEF = {"P": "pgrp", "C": "cw"}


def rownorm(M):
    s = np.asarray(M.sum(1)).ravel(); s[s == 0] = 1
    return sp.diags(1 / s) @ M


def build(t):
    V = P[P.ky <= t]
    firms_all = np.sort(V.applicantCd.unique()); fi = {f: i for i, f in enumerate(firms_all)}
    SC = np.sort(P.sc.unique()); si = {s: i for i, s in enumerate(SC)}
    G = np.sort(P.pgrp.dropna().unique()); gi = {g: i for i, g in enumerate(G)}   # 27
    nF, nS, nG = len(firms_all), len(SC), len(G)
    C = sp.csr_matrix((np.ones(len(V)), (V.applicantCd.map(fi).values, V.sc.map(si).values)), shape=(nF, nS))
    Wd = rownorm(C).toarray()
    nvis = np.asarray(C.sum(1)).ravel(); pop = np.where(nvis >= 2)[0]
    vrec3 = V[V.ay >= t - 2].groupby("applicantCd").size(); fmin = V.groupby("applicantCd").ay.min()
    base = pd.DataFrame({"applicantCd": firms_all[pop], "f_n": np.log1p(nvis[pop]),
                         "f_nsc": np.asarray((C > 0).sum(1)).ravel()[pop]})
    base["f_rec3"] = np.log1p(base.applicantCd.map(vrec3).fillna(0).values)
    base["f_age"] = t - base.applicantCd.map(fmin).values
    # 라벨 체계별 피처 행렬 (firm × g)
    feats = {}
    for L, col in DEF.items():
        Vl = V[V[col].notna()]
        fidx = Vl.applicantCd.map(fi).values; gidx = Vl[col].map(gi).values
        Cn = sp.csr_matrix((np.ones(len(Vl)), (fidx, gidx)), shape=(nF, nG)).toarray()
        M = Cn / np.maximum(Cn.sum(1, keepdims=True), 1)
        Hh = (Cn > 0).astype(float)
        nk = Hh.sum(0); Pco = (Hh.T @ Hh) / np.maximum(nk[:, None], 1); np.fill_diagonal(Pco, 0)  # P(have g | have k)
        fy = Vl.groupby(["applicantCd", col]).ay.min().reset_index()
        Hn = np.zeros((nF, nG)); Hn[fy.applicantCd.map(fi).values, fy[col].map(gi).values] = (fy.ay >= t - 3).astype(float).values
        mden = M.sum(0); mden[mden == 0] = 1
        En = (M.T @ Hn) / mden[:, None]; np.fill_diagonal(En, 0)
        Gs = sp.csr_matrix((np.ones(len(Vl)), (gidx, Vl.sc.map(si).values)), shape=(nG, nS)).toarray()
        Gs = Gs / np.maximum(Gs.sum(1, keepdims=True), 1)
        fit = (Wd @ Gs.T) / np.maximum(np.linalg.norm(Wd, axis=1, keepdims=True) * np.linalg.norm(Gs, axis=1)[None, :], 1e-12)
        g1 = Vl[Vl.ay.between(t - 2, t)][col].value_counts(); g0 = Vl[Vl.ay.between(t - 5, t - 3)][col].value_counts()
        gN = Cn.sum(0)
        ggrow = np.log1p(pd.Series(G).map(g1).fillna(0).values) - np.log1p(pd.Series(G).map(g0).fillna(0).values)
        gnew = Hn.sum(0) / np.maximum(nk, 1)
        ent = -(np.where(M > 0, M * np.log(np.where(M > 0, M, 1)), 0)).sum(1)
        feats[L] = dict(m=M, n=np.log1p(Cn), co=M @ Pco, en=M @ En, fit=fit, gN=np.log1p(gN), ggrow=ggrow, gnew=gnew,
                        mmax=M.max(1), ent=ent, nmatched=np.log1p(Cn.sum(1)))
    for D, col in DEF.items():
        E = P[(P.ay <= t) & P[col].notna()]
        have = set(zip(E.applicantCd.map(lambda x: fi.get(x, -1)).values, E[col].map(gi).values))
        ff, gg = np.meshgrid(pop, np.arange(nG), indexing="ij"); ff = ff.ravel(); gg = gg.ravel()
        keep = np.array([(a, b) not in have for a, b in zip(ff, gg)])
        ff, gg = ff[keep], gg[keep]
        X = pd.DataFrame({"applicantCd": firms_all[ff], "g": G[gg], "t": t})
        X = X.merge(base, on="applicantCd", how="left")
        for L, F in feats.items():
            X[f"{L}_m"] = F["m"][ff, gg]; X[f"{L}_n"] = F["n"][ff, gg]
            X[f"{L}_co"] = F["co"][ff, gg]; X[f"{L}_en"] = F["en"][ff, gg]; X[f"{L}_fit"] = F["fit"][ff, gg]
            X[f"{L}_gN"] = F["gN"][gg]; X[f"{L}_ggrow"] = F["ggrow"][gg]; X[f"{L}_gnew"] = F["gnew"][gg]
            X[f"{L}_mmax"] = F["mmax"][ff]; X[f"{L}_ent"] = F["ent"][ff]; X[f"{L}_nm"] = F["nmatched"][ff]
        Y = P[(P.ay > t) & (P.ay <= t + 4) & P[col].notna()].groupby(["applicantCd", col]).ky.min().rename("first_ky").reset_index()
        Y.columns = ["applicantCd", "g", "first_ky"]
        X = X.merge(Y, on=["applicantCd", "g"], how="left")
        for c in X.columns:
            if X[c].dtype == np.float64:
                X[c] = X[c].astype(np.float32)
        X.to_parquet(os.path.join(CACHE, f"C_o{t}_{D}.parquet"), index=False)
        log(t, D, "pairs", len(X), "pos", int(X.first_ky.notna().sum()), "rate", round(float(X.first_ky.notna().mean()), 4))


if __name__ == "__main__":
    for t in map(int, sys.argv[1:]):
        build(t)
