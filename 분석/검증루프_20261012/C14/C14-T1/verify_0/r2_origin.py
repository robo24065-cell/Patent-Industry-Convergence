# -*- coding: utf-8 -*-
r"""verify_0 R2: 원점별 분리 후 통합. 원점마다 CGM(선형화)을 계산하고 (a) 원점 독립 가정 ΣV_c/25, (b) 역분산 가중 평균 Δ
(원점 독립 SE와 원점 간 상관 반영 SE 둘 다). ψ는 cache/psi14.npz(작업 e3_e5.py 산출) 대신 여기서 sklearn 순위로 다시 계산해 대조.
→ verify_0/r2_origin.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import rankdata, t as tdist
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
z = np.load(os.path.join(W, "cache", "data14.npz")); zp = np.load(os.path.join(W, "cache", "psi14.npz"))
CS = [2008, 2009, 2010, 2011, 2012]
PAIRS = [("P50", "FREE50"), ("P50", "FREEIPC50"), ("P50", "HSC1_50"), ("P50", "TXT50")]
G = {k: int(z[f"G_{k}"]) for k in ("firm", "sc", "ksic_t", "cell_fs", "cell_fk")}


def psi_rank(y, s):
    """DeLong 구성요소를 rankdata(평균 순위)로: V10_i = (R_all_i − R_pos_i)/n0, V01_j = 1 − (R_all_j − R_neg_j)/n1"""
    y = y.astype(bool); n1 = y.sum(); n0 = (~y).sum()
    ra = rankdata(s); rp = rankdata(s[y]); rn = rankdata(s[~y])
    v10 = (ra[y] - rp) / n0; v01 = 1 - (ra[~y] - rn) / n1
    A = v10.mean(); psi = np.empty(len(s)); psi[y] = (v10 - A) / n1; psi[~y] = (v01 - A) / n0
    return A, psi


def V(sumvec, g, Gn):
    S = np.bincount(g, weights=sumvec, minlength=Gn); m = (np.bincount(g, minlength=Gn) > 0).sum()
    return m / (m - 1) * (S ** 2).sum(), m


out = {"psi_maxdiff_vs_work": 0.0, "auc_maxdiff_sklearn": 0.0}
PS = {}
for c in CS:
    y = z[f"y_{c}"]
    for a in ("P50", "FREE50", "FREEIPC50", "HSC1_50", "TXT50"):
        A, p = psi_rank(y, z[f"{a}_{c}"]); PS[(c, a)] = (A, p)
        out["psi_maxdiff_vs_work"] = max(out["psi_maxdiff_vs_work"], float(np.abs(p - zp[f"{c}_{a}"]).max()))
        out["auc_maxdiff_sklearn"] = max(out["auc_maxdiff_sklearn"], abs(A - roc_auc_score(y, z[f"{a}_{c}"])))
log("psi", out)
for a, b in PAIRS:
    for s, (A_, B_, AB_) in {"fs": ("firm", "sc", "cell_fs"), "fk": ("firm", "ksic_t", "cell_fk")}.items():
        per = {}
        for c in CS:
            d = PS[(c, a)][1] - PS[(c, b)][1]
            vA, gA = V(d, z[f"{A_}_{c}"], G[A_]); vB, gB = V(d, z[f"{B_}_{c}"], G[B_]); vAB, _ = V(d, z[f"{AB_}_{c}"], G[AB_])
            v = vA + vB - vAB
            per[c] = {"d": PS[(c, a)][0] - PS[(c, b)][0], "v": v if v > 0 else max(vA, vB), "df": min(gA, gB) - 1}
        dm = np.mean([per[c]["d"] for c in CS])
        v_ind = sum(per[c]["v"] for c in CS) / 25
        wc = {c: (1 / per[c]["v"]) / sum(1 / per[k]["v"] for k in CS) for c in CS}
        d_iv = sum(wc[c] * per[c]["d"] for c in CS); v_iv_ind = 1 / sum(1 / per[c]["v"] for c in CS)
        # 원점 간 상관 반영(군집 합을 원점 가중으로 묶음)
        def joint(wt):
            tot = {}
            for comp in (A_, B_, AB_):
                Ssum = np.zeros(G[comp])
                for c in CS:
                    Ssum += np.bincount(z[f"{comp}_{c}"], weights=wt[c] * (PS[(c, a)][1] - PS[(c, b)][1]), minlength=G[comp])
                m = (Ssum != 0).sum(); tot[comp] = m / (m - 1) * (Ssum ** 2).sum()
            v = tot[A_] + tot[B_] - tot[AB_]
            return v if v > 0 else max(tot[A_], tot[B_])
        v_eq_joint = joint({c: 0.2 for c in CS}); v_iv_joint = joint(wc)
        df = min(per[c]["df"] for c in CS); q = tdist.ppf(0.975, df)
        out[f"{s}|{a}-{b}"] = {"per_origin": {str(c): per[c] for c in CS},
                                "eq_mean": dm, "eq_indep_ci": [dm - q * np.sqrt(v_ind), dm + q * np.sqrt(v_ind)],
                                "eq_joint_ci": [dm - q * np.sqrt(v_eq_joint), dm + q * np.sqrt(v_eq_joint)],
                                "iv_mean": d_iv, "iv_weights": {str(c): wc[c] for c in CS},
                                "iv_indep_ci": [d_iv - q * np.sqrt(v_iv_ind), d_iv + q * np.sqrt(v_iv_ind)],
                                "iv_joint_ci": [d_iv - q * np.sqrt(v_iv_joint), d_iv + q * np.sqrt(v_iv_joint)],
                                "se_eq_indep": float(np.sqrt(v_ind)), "se_eq_joint": float(np.sqrt(v_eq_joint))}
        log(s, a, b, out[f"{s}|{a}-{b}"]["eq_joint_ci"], out[f"{s}|{a}-{b}"]["eq_indep_ci"], out[f"{s}|{a}-{b}"]["iv_joint_ci"])
json.dump(out, open(os.path.join(HERE, "r2_origin.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
