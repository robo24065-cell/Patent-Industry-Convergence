# -*- coding: utf-8 -*-
r"""verify_0 재시도 M3: 곱 가중 과대 기제(2·V_AB) 독립 재계산.
작업 캐시(data14.npz, psi14.npz) 미사용. 원천 = NONCIRC_R8 A_o{c}·patents, C10-T2 공식 seed 예측, C10-T2 boot npz.
ψ는 scipy rankdata(중간 순위)로, 군집 합은 pandas 문자열 키 groupby로 계산(작업의 searchsorted/bincount와 다른 경로).
쓰기: verify_0\m3_mech.json 만."""
import os, sys, json, time, glob
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import rankdata

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
NC8 = os.path.join(BASE, r"분석\모델경쟁_20260927\T1_산업대기실\NONCIRC_R8\cache")
C10C = os.path.join(BASE, r"분석\검증루프_20261012\C10\C10-T2\cache")
OUT = os.path.join(BASE, r"분석\검증루프_20261012\C14\C14-T1\verify_0")
T0 = time.time()
def log(*a): print(*a, f"[{round(time.time()-T0)}s]", flush=True)
CO = [2008, 2009, 2010, 2011, 2012]

P = pd.read_parquet(os.path.join(NC8, "patents.parquet"), columns=["applicantCd", "sc", "ksic", "ky"])
P = P[P.ky <= 2012]
g = P.groupby(["sc", "ksic"]).size().reset_index(name="n").sort_values(["sc", "n", "ksic"], ascending=[True, False, True])
KT = g.drop_duplicates("sc").set_index("sc").ksic
del P, g


def psi_rank(y, s):
    y = y.astype(bool); n1 = y.sum(); n0 = (~y).sum()
    ra = rankdata(s)                      # 전체 중간 순위
    rp = rankdata(s[y]); rn = rankdata(s[~y])
    v10 = (ra[y] - rp) / n0               # 음성 중 아래 + 0.5 동점
    v01 = (n1 - (ra[~y] - rn)) / n1       # 양성 중 위 + 0.5 동점
    A = v10.mean()
    out = np.empty(len(s)); out[y] = (v10 - A) / n1; out[~y] = (v01 - A) / n0
    return A, out


frames = []; pts = []
for c in CO:
    a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"])
    p = pd.read_parquet(os.path.join(C10C, f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0"])
    assert len(a) == len(p)
    y = a.first_ky.notna().values
    A1, s1 = psi_rank(y, p.P50_p0.values.astype(np.float64))
    A2, s2 = psi_rank(y, p.FREE50_p0.values.astype(np.float64))
    pts.append(A1 - A2)
    f = pd.DataFrame({"firm": a.applicantCd.astype(str).values, "sc": a.sc.astype(str).values, "o": c, "y": y.astype(np.int8),
                      "d": (s1 - s2) / 5.0})
    f["kt"] = a.sc.map(KT).astype(str).values
    assert f.kt.ne("nan").all()
    frames.append(f); log("origin", c, len(f), int(y.sum()), A1 - A2)
D = pd.concat(frames, ignore_index=True); del frames
R = {"point": float(np.mean(pts)), "n_rows": int(len(D)), "n_pos": int(D.y.sum()), "sum_psi": float(D.d.sum())}


def V(keys):
    S = D.groupby(keys, sort=False).d.sum().values
    G = len(S)
    return float(G / (G - 1) * (S ** 2).sum()), G, float((S ** 2).sum())

for nm, keys in {"firm": ["firm"], "sc": ["sc"], "kt": ["kt"], "cell_fs": ["firm", "sc"], "cell_fk": ["firm", "kt"],
                 "cell_fso": ["firm", "sc", "o"], "cell_fko": ["firm", "kt", "o"]}.items():
    v, G, raw = V(keys); R[f"V_{nm}"] = v; R[f"G_{nm}"] = G; R[f"raw_{nm}"] = raw
    log(nm, v, G)
R["row_iid"] = float((D.d ** 2).sum())

# 구조(교차/중첩)
fsn = D.groupby("firm").sc.nunique(); R["firm_mean_distinct_sc"] = float(fsn.mean())
cnt = D.groupby(["firm", "sc"]).size()
R["firm_modal_sc_share_rowweighted"] = float(cnt.groupby(level=0).max().sum() / cnt.sum())
R["rows_per_cell_fs"] = float(cnt.mean()); R["cells_fs"] = int(len(cnt))
R["cells_possible_fs"] = int(D.firm.nunique() * D.sc.nunique())
R["dup_firm_sc_origin"] = int(D.duplicated(["firm", "sc", "o"]).sum())
pc = D.groupby(["firm", "sc"]).y.sum(); R["pos_cells_fs_max_count"] = int(pc.max()); R["pos_cells_fs_n"] = int((pc > 0).sum())
R["pos_cells_fs_dist"] = {str(k): int(v) for k, v in pc[pc > 0].value_counts().sort_index().items()}

for s, A_, B_, AB_, ABO in (("fs", "firm", "sc", "cell_fs", "cell_fso"), ("fk", "firm", "kt", "cell_fk", "cell_fko")):
    vA, vB, vAB = R[f"V_{A_}"], R[f"V_{B_}"], R[f"V_{AB_}"]
    e = {"vA": vA, "vB": vB, "vAB": vAB, "share": vAB / (vA + vB),
         "pigeon": vA + vB + vAB, "cgm": vA + vB - vAB, "pred_se_ratio": float(np.sqrt((vA + vB + vAB) / (vA + vB - vAB))),
         "vAB_over_rowiid": vAB / R["row_iid"], "vAB_origin_split(=칸×원점)": R[f"V_{ABO}"],
         "cross_origin_cov_part": vAB - R[f"V_{ABO}"],
         "pred_se_ratio_if_no_repeat": float(np.sqrt((vA + vB + R[f"V_{ABO}"]) / (vA + vB - R[f"V_{ABO}"])))}
    R[s] = e

# 선형 통계량에 C10과 같은 곱 가중 Poisson을 몬테카를로로 적용: Var(Σ w_f w_s ψ) vs V_A+V_B+V_AB(이론, 보정계수 없는 raw)
fi = pd.factorize(D.firm)[0]; si = pd.factorize(D.sc)[0]; ki = pd.factorize(D.kt)[0]; d = D.d.values
rng = np.random.default_rng(777)
for s, bi in (("fs", si), ("fk", ki)):
    Sab = pd.Series(d).groupby([fi, bi]).sum()
    a_ = Sab.index.get_level_values(0).values; b_ = Sab.index.get_level_values(1).values; v_ = Sab.values
    T = np.empty(4000)
    for r in range(4000):
        wf = rng.poisson(1.0, fi.max() + 1); wb = rng.poisson(1.0, bi.max() + 1)
        T[r] = (wf[a_] * wb[b_] * v_).sum()
    raw_th = R["raw_firm"] + R["raw_" + ("sc" if s == "fs" else "kt")] + R["raw_" + ("cell_fs" if s == "fs" else "cell_fk")]
    R[s]["mc_linear_product_var"] = float(T.var(ddof=1)); R[s]["theory_raw_pigeon"] = raw_th
    R[s]["mc_over_theory"] = float(T.var(ddof=1) / raw_th)
    log("mc", s, R[s]["mc_over_theory"])

# C10 E2·단일 부트스트랩 원본 npz에서 직접 분산
def bootvar(clt):
    fs = [f for f in glob.glob(os.path.join(C10C, f"boot_{clt}_*.npz")) if os.path.basename(f)[5:-4].rsplit("_", 2)[0] == clt]
    parts = sorted([(int(os.path.basename(f)[:-4].split("_")[-2]), np.load(f)) for f in fs], key=lambda x: x[0])
    arms = list(parts[0][1]["arms"]); A = np.concatenate([q[1]["auc"] for q in parts])
    db = (A[:, arms.index("P50"), :] - A[:, arms.index("FREE50"), :]).mean(1)
    return float(np.var(db, ddof=1)), len(db)
for clt in ("firm", "sc", "ksic_t", "firm_sc", "firm_ksic_t"):
    R[f"boot_{clt}"] = bootvar(clt)
for s, clt in (("fs", "firm_sc"), ("fk", "firm_ksic_t")):
    bv = R[f"boot_{clt}"][0]
    R[s]["boot_e2_var"] = bv; R[s]["obs_se_ratio"] = float(np.sqrt(bv / R[s]["cgm"])); R[s]["boot_over_pigeon"] = bv / R[s]["pigeon"]
R["lin_over_boot_single"] = {"firm": R["V_firm"] / R["boot_firm"][0], "sc": R["V_sc"] / R["boot_sc"][0], "kt": R["V_kt"] / R["boot_ksic_t"][0]}
json.dump(R, open(os.path.join(OUT, "m3_mech.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("done")
