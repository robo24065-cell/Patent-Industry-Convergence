# -*- coding: utf-8 -*-
r"""E3 CGM 이원 분산(placement value 선형화) + E5 구조 진단 + 선형화 점검(E1·E2 부트스트랩 분산 대비).
→ res/e3.json, structure_diag.json(폴더 루트), cache/psi14.npz"""
import sys
sys.dont_write_bytecode = True
from lib14 import *
from scipy.stats import t as tdist

D, G = build_data()
BT = np.load(os.path.join(CACHE, "boot_c10_all.npz"))
# ψ: 원점별·팔별
PSI = {}; AUCV = {}
for c in C_ORIG:
    for arm in ARMS:
        AUCV[(c, arm)], PSI[(c, arm)] = placement(D[c]["y"], D[c][arm])
np.savez_compressed(safe_path(os.path.join(CACHE, "psi14.npz")), **{f"{c}_{a}": PSI[(c, a)] for c in C_ORIG for a in ARMS})
STR = {"fs": ("firm", "sc", "cell_fs"), "fk": ("firm", "ksic_t", "cell_fk")}


def comps_for(a, b, cs=C_ORIG, wt=None):
    """Δ ψ를 군집별로 합산: 반환 dict comp->V, G"""
    out = {}
    for comp in ("firm", "sc", "ksic_t", "cell_fs", "cell_fk"):
        S = np.zeros(G[comp]); cnt = np.zeros(G[comp])
        for c in cs:
            w = (1.0 / len(cs)) if wt is None else wt[c]
            S += np.bincount(D[c][comp], weights=w * (PSI[(c, a)] - PSI[(c, b)]), minlength=G[comp])
            cnt += np.bincount(D[c][comp], minlength=G[comp])
        Gn = int((cnt > 0).sum())
        out[comp] = (float(Gn / (Gn - 1) * (S ** 2).sum()), Gn)
    # 행 단위(iid) 분산
    vi = 0.0
    for c in cs:
        w = (1.0 / len(cs)) if wt is None else wt[c]
        vi += float(((w * (PSI[(c, a)] - PSI[(c, b)])) ** 2).sum())
    out["row"] = (vi, None)
    return out


def cgm(V, sA, sB, sAB, pt):
    vA, GA = V[sA]; vB, GB = V[sB]; vAB, _ = V[sAB]
    v = vA + vB - vAB; fixed = False
    if v <= 0:
        v = max(vA, vB); fixed = True
    df = min(GA, GB) - 1; q = tdist.ppf(0.975, df); se = np.sqrt(v)
    return {"vA": vA, "vB": vB, "vAB": vAB, "v": v, "se": se, "fixed": fixed, "df": df, "lo": pt - q * se, "hi": pt + q * se,
            "lo_z": pt - 1.96 * se, "hi_z": pt + 1.96 * se, "GA": GA, "GB": GB}


res = {}
for a, b in PAIRS:
    pt = float(np.mean([AUCV[(c, a)] - AUCV[(c, b)] for c in C_ORIG]))
    V = comps_for(a, b)
    ia, ib = ARMS.index(a), ARMS.index(b)
    bvar = {clt: float(np.var((BT[clt][:, ia, :] - BT[clt][:, ib, :]).mean(1), ddof=1)) for clt in ("firm", "firm_sc", "firm_ksic_t", "sc", "ksic_t")}
    r = {"point": pt, "lin_var": {k: v[0] for k, v in V.items()}, "G": {k: v[1] for k, v in V.items()}, "boot_var_c10": bvar}
    for s, (A_, B_, AB_) in STR.items():
        e = cgm(V, A_, B_, AB_, pt)
        e["pigeon_lin_var"] = e["vA"] + e["vB"] + e["vAB"]
        e["boot_e2_var"] = bvar["firm_sc" if s == "fs" else "firm_ksic_t"]
        e["ratio_boot_e2_to_lin_pigeon"] = e["boot_e2_var"] / e["pigeon_lin_var"]
        e["vAB_share"] = e["vAB"] / (e["vA"] + e["vB"])
        e["pred_se_ratio_e2_e3"] = float(np.sqrt(e["pigeon_lin_var"] / e["v"]))
        e["obs_se_ratio_e2_e3"] = float(np.sqrt(e["boot_e2_var"] / e["v"]))
        r[s] = e
    r["lin_check"] = {"firm_lin/boot": V["firm"][0] / bvar["firm"], "sc_lin/boot": V["sc"][0] / bvar["sc"], "ksic_t_lin/boot": V["ksic_t"][0] / bvar["ksic_t"]}
    res[f"{a}-{b}"] = r
    log(a, b, {s: (round(r[s]["lo"], 6), round(r[s]["hi"], 6), round(r[s]["se"], 7)) for s in STR}, r["lin_check"])
json.dump(res, open(safe_path(os.path.join(RES, "e3.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# ---------- E5 구조 진단
def nest(A_, B_):
    """A가 B 안에 중첩된 정도: A 단위별 행 중 최빈 B 몫(행 가중 평균), B 하나뿐인 A 비율"""
    a = np.concatenate([D[c][A_] for c in C_ORIG]); b = np.concatenate([D[c][B_] for c in C_ORIG])
    df = pd.DataFrame({"a": a, "b": b})
    g = df.groupby(["a", "b"]).size().rename("n").reset_index()
    tot = g.groupby("a").n.sum(); mx = g.groupby("a").n.max(); nb = g.groupby("a").size()
    return {"row_weighted_modal_share": float(mx.sum() / tot.sum()), "share_units_single": float((nb == 1).mean()),
            "median_distinct": float(nb.median()), "mean_distinct": float(nb.mean()), "max_possible": int(G[B_])}


def sizes(comp):
    n = np.zeros(G[comp]); npos = np.zeros(G[comp])
    for c in C_ORIG:
        n += np.bincount(D[c][comp], minlength=G[comp]); npos += np.bincount(D[c][comp], weights=D[c]["y"], minlength=G[comp])
    q = lambda x: {f"q{int(p*100)}": float(np.quantile(x, p)) for p in (0, .1, .25, .5, .75, .9, .99, 1)}
    top = np.sort(n)[::-1]; topp = np.sort(npos)[::-1]
    return {"G": int((n > 0).sum()), "rows": q(n), "pos": q(npos), "top10_row_share": float(top[:10].sum() / n.sum()),
            "top10_pos_share": float(topp[:10].sum() / npos.sum()), "share_clusters_with_pos": float((npos > 0).mean()),
            "total_rows": int(n.sum()), "total_pos": int(npos.sum())}


SD = {"nesting": {"firm_in_sc": nest("firm", "sc"), "sc_in_firm": nest("sc", "firm"), "firm_in_ksic_t": nest("firm", "ksic_t"),
                  "ksic_t_in_firm": nest("ksic_t", "firm"), "sc_in_ksic_t": nest("sc", "ksic_t")},
      "sizes": {k: sizes(k) for k in ("firm", "sc", "ksic_t", "cell_fs", "cell_fk")},
      "rows_per_cell_fs_mean(=원점 반복)": None}
SD["rows_per_cell_fs_mean(=원점 반복)"] = SD["sizes"]["cell_fs"]["total_rows"] / SD["sizes"]["cell_fs"]["G"]
SD["variance_decomposition_P50-FREE50"] = {s: {k: res["P50-FREE50"][s][k] for k in ("vA", "vB", "vAB", "vAB_share", "pred_se_ratio_e2_e3",
                                                                                     "obs_se_ratio_e2_e3", "ratio_boot_e2_to_lin_pigeon")} for s in STR}
SD["variance_decomposition_P50-FREE50"]["row_iid_var"] = res["P50-FREE50"]["lin_var"]["row"]
SD["note"] = ("법인은 sc·ksic_t 안에 중첩되지 않고 거의 완전 교차(법인당 서로 다른 sc 수 참조). 칸(법인,sc)은 원점 간 같은 행의 반복. "
              "곱 가중 Poisson 1차 분산 = V_A+V_B+V_AB, CGM = V_A+V_B−V_AB → 과대분 2·V_AB.")
json.dump(SD, open(safe_path(os.path.join(OUT, "structure_diag.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("E5", json.dumps(SD["nesting"], default=float)[:800])
log(json.dumps(SD["variance_decomposition_P50-FREE50"], default=float))
