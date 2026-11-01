# -*- coding: utf-8 -*-
r"""N7 R7 사후 분석(공식 성적 불변): 최종시험 구간(학습 t0 2015–2020 → 시험 t0=2022, 과제연도 2023–24)에서 같은 변형을 재학습.
동결 final_panel(읽기 전용)을 쓰고, 연계표·IPC 대체 피처만 같은 코드로 다시 계산. 결과로 모형을 고치지 않는다."""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
sys.path.insert(0, TR); sys.path.insert(0, os.path.join(MC, "T4_특허포기예측", "ESSENTIAL_R7", "common")); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import n7lib as L
from n7feat import f_C06
from xwalk import load_app_xw
from ky_xw import build as build_ky
SPEC = json.load(open(os.path.join(TR, "final_spec_r5.json"), encoding="utf-8"))
FE, M0F, SEEDS = SPEC["features"], SPEC["baseline_M0"], tuple(SPEC["model"]["seeds"])
D = pd.read_parquet(os.path.join(TR, "cache", "final_panel.parquet"))
KS = ["n_ksic", "k_rnd_g", "k_rnd_sh", "peer_entry_k"]
fx = os.path.join(HERE, "ksic_xi_final.parquet")
if os.path.exists(fx):
    X = pd.read_parquet(fx)
else:
    AF = pd.read_parquet(os.path.join(TR, "cache", "app_firm.parquet"))
    AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()].copy()
    ax = load_app_xw().set_index("appl_no")
    AFx = AF.copy(); AFx["ksic"] = AFx.appl_no.astype(str).map(ax.xw)
    AFi = AF.copy(); AFi["ksic"] = AFi.appl_no.astype(str).map(ax.main_ipc.fillna("").str.replace(" ", "").str[:4].replace("", np.nan))
    R = {k: pd.read_parquet(os.path.join(MC, "N4_특허행태_기업폐업조기경보", "cache", f"reg_{k}.parquet")) for k in ["ap", "pt", "lapnp", "trf", "py", "rec"]}
    ns = L.load_nts(); first = L.ntis_first_year(L.load_ntis("final"))
    KY = pd.read_parquet(os.path.join(MC, "_shared", "ext", "ntis_project_ksic_year.parquet")); KY = KY[KY.method == "a_link_pit"]
    KYx = build_ky("xw", 2022); KYi = build_ky("ipc4", 2022)
    parts = []
    for t0, B in D.groupby("t0"):
        B = B[["applicantCd"]].copy(); o = pd.DataFrame(index=B.index); end = t0 * 10000 + 1231
        for tag, af, ky in [("chk", AF, KY), ("x", AFx, KYx), ("i", AFi, KYi)]:
            kn = af[(af.known_date <= end) & af.applicantCd.isin(set(B.applicantCd))]
            o[f"n_ksic_{tag}"] = B.applicantCd.map(kn.groupby("applicantCd").ksic.nunique()).fillna(0).values
            m3 = L.f_M3(B, t0, af, ky); o[f"k_rnd_g_{tag}"] = m3.k_rnd_g.values; o[f"k_rnd_sh_{tag}"] = m3.k_rnd_sh.values
            o[f"peer_entry_k_{tag}"] = f_C06(B, t0, dict(AF=af, R=R, ns=ns, first=first)).peer_entry_k.values
        parts.append(o); print("x", t0, flush=True)
    X = pd.concat(parts).loc[D.index]; X.to_parquet(fx)
D = pd.concat([D, X], axis=1)
chk = {c: float(np.mean(np.isclose(D[c].astype(float).fillna(-9), D[f"{c}_chk"].astype(float).fillna(-9), rtol=1e-6, atol=1e-9))) for c in KS}
PROV = KS + ["log_app", "yrs_since_last", "app_trend", "known_1y", "live_ratio", "g_part_any", "g_part_n", "g_part_proj", "g_part_univ", "g_part_gri", "g_part_firm", "g_part_rec", "g_part_share"]
REGB = ["log_live", "lapse_1y", "lapse_2y", "pay_1y", "pay_trend", "trf_2y", "late_share_2y", "delay_mean_2y", "overdue_share", "rec_2y", "inv_n_all", "inv_n_3y", "inv_per_pat", "inv_growth"]
V = {"CH": FE, "R_KSIC": [f for f in FE if f not in KS], "S_KSIC": [f"{f}_x" if f in KS else f for f in FE],
     "S_IPC4": [f"{f}_i" if f in KS else f for f in FE], "R_PROV": [f for f in FE if f not in PROV], "R_PAT": [f for f in FE if f not in PROV + REGB]}
tr = D[D.t0.between(2015, 2020)]; te = D[D.t0 == 2022]; y = te.y.values
p0 = L.predict(tr[M0F].astype(float), tr.y, te[M0F].astype(float), model="xgb", seeds=SEEDS); a0 = float(L.auc(y, p0))
out = dict(note="사후 분석 — 공식 성적(final_result_r5.json) 불변", recompute_match=chk, auc_M0=a0, official_auc=SPEC.get("final", {}), res={})
for k, f in V.items():
    p = L.predict(tr[f].astype(float), tr.y, te[f].astype(float), model="ens", seeds=SEEDS); a = float(L.auc(y, p))
    out["res"][k] = dict(auc=a, skill=(a - a0) / (1 - a0)); print(k, round(a, 4), round((a - a0) / (1 - a0), 4), flush=True)
json.dump(out, open(os.path.join(HERE, "posthoc_final_r7.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("M0", a0, chk)
