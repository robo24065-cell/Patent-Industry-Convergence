# -*- coding: utf-8 -*-
r"""자기 검증: 합성 자료(S1 교차 / S2 중첩)에서 E1·E2·E3·E4·E4b의 95% CI 포함률.
사용: python -B selftest.py <S1|S2> <rep_lo> <rep_hi> → cache/st_<S>_<lo>_<hi>.npz
각 반복: 무작위 효과를 새로 뽑아 자료 생성 → Δ=AUC(s1)−AUC(s2) 점추정, 각 추정량의 SE·CI."""
import sys
sys.dont_write_bytecode = True
from lib14 import *
from scipy.stats import t as tdist

B = 200


def gen(S, rng):
    if S == "S1":
        NF, NS = 300, 60
        f, s = np.meshgrid(np.arange(NF), np.arange(NS), indexing="ij"); f = f.ravel(); s = s.ravel()
        keep = rng.random(len(f)) < 0.5; f, s = f[keep], s[keep]
    else:
        NS, FP, RP = 60, 5, 30
        NF = NS * FP
        f = np.repeat(np.arange(NF), RP); s = f // FP
    a = rng.normal(0, 0.5, NF); b = rng.normal(0, 0.5, NS)
    p = 1 / (1 + np.exp(-(-2.6 + a[f] + b[s])))
    y = (rng.random(len(f)) < p).astype(np.int8)
    eta = 1.2 * y + rng.normal(0, 1, len(f))
    s1 = eta + rng.normal(0, 0.6, len(f))
    u = rng.normal(0, 0.45, NF); v = rng.normal(0, 0.45, NS)
    s2 = eta + rng.normal(0, 1, len(f)) * 0.6 * np.exp(u[f] + v[s])
    # 칸(법인, sc): S1은 행=칸, S2는 칸=법인
    cell = f * NS + s
    _, cell = np.unique(cell, return_inverse=True)
    return y, s1, s2, f, s, cell, NF, NS, int(cell.max() + 1)


def one(S, r):
    rng = np.random.default_rng([777, 1 if S == "S1" else 2, r])
    y, s1, s2, f, s, cell, NF, NS, NC = gen(S, rng)
    out = {}
    A1, p1 = placement(y, s1); A2, p2 = placement(y, s2); pt = A1 - A2; out["pt"] = pt
    psi = p1 - p2
    vA, GA = vclust(psi, f, NF); vB, GB = vclust(psi, s, NS); vAB, _ = vclust(psi, cell, NC)
    v3 = vA + vB - vAB; fix3 = v3 <= 0
    if fix3: v3 = max(vA, vB)
    df = min(GA, GB) - 1; q = tdist.ppf(0.975, df)
    out["E3_se"] = np.sqrt(v3); out["E3_fix"] = fix3
    out["E1lin_se"] = np.sqrt(vA)
    P = [AucPrep(y, sc_, [f, s, cell]) for sc_ in (s1, s2)]
    brng = np.random.default_rng([778, 1 if S == "S1" else 2, r])
    d = {k: np.empty(B) for k in ("E1", "E2", "cell", "add")}
    for bb in range(B):
        wf = brng.poisson(1.0, NF).astype(float); ws = brng.poisson(1.0, NS).astype(float); wc = brng.poisson(1.0, NC).astype(float)
        one_ = np.ones(NF); ones_s = np.ones(NS); ones_c = np.ones(NC)
        res = []
        for P_ in P:
            e1 = P_.auc([wf, ones_s, ones_c]); e2 = P_.auc([wf, ws, ones_c]); ec = P_.auc([one_, ones_s, wc])
            wn = 0.5 * (wf[P_.neg_cl[0]] + ws[P_.neg_cl[1]]); wp = 0.5 * (wf[P_.pos_cl[0]] + ws[P_.pos_cl[1]])
            ea = P_.auc_rowweights(wn, wp)
            res.append((e1, e2, ec, ea))
        d["E1"][bb] = res[0][0] - res[1][0]; d["E2"][bb] = res[0][1] - res[1][1]; d["cell"][bb] = res[0][2] - res[1][2]; d["add"][bb] = res[0][3] - res[1][3]
    out["E1_lo"], out["E1_hi"] = np.percentile(d["E1"], [2.5, 97.5]); out["E1_se"] = d["E1"].std(ddof=1)
    out["E2_lo"], out["E2_hi"] = np.percentile(d["E2"], [2.5, 97.5]); out["E2_se"] = d["E2"].std(ddof=1)
    vcell = d["cell"].var(ddof=1)
    v4 = d["E2"].var(ddof=1) - 2 * vcell; fix4 = v4 <= 0
    if fix4: v4 = max(vA, vB)
    v4b = 4 * d["add"].var(ddof=1) - vcell; fix4b = v4b <= 0
    if fix4b: v4b = max(vA, vB)
    out["E4_se"] = np.sqrt(v4); out["E4_fix"] = fix4; out["E4b_se"] = np.sqrt(v4b); out["E4b_fix"] = fix4b
    out["E3_lo"], out["E3_hi"] = pt - q * out["E3_se"], pt + q * out["E3_se"]
    out["E4_lo"], out["E4_hi"] = pt - q * out["E4_se"], pt + q * out["E4_se"]
    out["E4b_lo"], out["E4b_hi"] = pt - q * out["E4b_se"], pt + q * out["E4b_se"]
    out["vA"], out["vB"], out["vAB"] = vA, vB, vAB
    out["prev"] = y.mean(); out["n"] = len(y)
    return out


if __name__ == "__main__":
    S, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = []
    for r in range(lo, hi):
        rows.append(one(S, r))
        if (r - lo) % 25 == 0:
            log(S, r)
    keys = rows[0].keys()
    np.savez(safe_path(os.path.join(CACHE, f"st_{S}_{lo}_{hi}.npz")), **{k: np.array([x[k] for x in rows], dtype=float) for k in keys})
    log("done", S, lo, hi)
