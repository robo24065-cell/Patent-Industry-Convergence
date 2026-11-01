# -*- coding: utf-8 -*-
r"""N7 R7 필수성 검증: 동결 챔피언(R03, ens, 시드 0·1·2) 설정 그대로 피처만 바꿔 개발 원점 2017–2020 재학습.
python e7_n7.py <VARIANT ...>  (CH R_KSIC R_PROV R_PAT S_KSIC S_KSIC_ONLYK)
읽기: ..\cache\dev_panel.parquet·app_firm.parquet, N4\cache\reg_*.parquet(같은 코드로 만든 등록원부 파생표), _shared\ext(NTIS) — 읽기 전용."""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
sys.path.insert(0, TR); sys.path.insert(0, os.path.join(MC, "T4_특허포기예측", "ESSENTIAL_R7", "common")); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import n7lib as L
from n7feat import f_C06
from firmrun import run_variant
from xwalk import load_app_xw
from ky_xw import build as build_ky

SPEC = json.load(open(os.path.join(TR, "final_spec_r5.json"), encoding="utf-8"))
FE, M0F, SEEDS = SPEC["features"], SPEC["baseline_M0"], tuple(SPEC["model"]["seeds"])
D = pd.read_parquet(os.path.join(TR, "cache", "dev_panel.parquet"))

KS = ["n_ksic", "k_rnd_g", "k_rnd_sh", "peer_entry_k"]
PROV = KS + ["log_app", "yrs_since_last", "app_trend", "known_1y", "live_ratio",
             "g_part_any", "g_part_n", "g_part_proj", "g_part_univ", "g_part_gri", "g_part_firm", "g_part_rec", "g_part_share"]
REGB = ["log_live", "lapse_1y", "lapse_2y", "pay_1y", "pay_trend", "trf_2y", "late_share_2y", "delay_mean_2y", "overdue_share", "rec_2y",
        "inv_n_all", "inv_n_3y", "inv_per_pat", "inv_growth"]


def add_x(D):
    fx = os.path.join(HERE, "ksic_xi_dev.parquet")
    if os.path.exists(fx):
        return pd.read_parquet(fx)
    AF = pd.read_parquet(os.path.join(TR, "cache", "app_firm.parquet"))
    AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()].copy()
    ax = load_app_xw().set_index("appl_no")
    AFx = AF.copy(); AFx["ksic"] = AFx.appl_no.astype(str).map(ax.xw)
    AFi = AF.copy(); AFi["ksic"] = AFi.appl_no.astype(str).map(ax.main_ipc.fillna("").str.replace(" ", "").str[:4].replace("", np.nan))
    R = {k: pd.read_parquet(os.path.join(MC, "N4_특허행태_기업폐업조기경보", "cache", f"reg_{k}.parquet")) for k in ["ap", "pt", "lapnp", "trf", "py", "rec"]}
    ns = L.load_nts(); first = L.ntis_first_year(L.load_ntis("dev"))
    KY = pd.read_parquet(os.path.join(MC, "_shared", "ext", "ntis_project_ksic_year.parquet")); KY = KY[KY.method == "a_link_pit"]
    KYx = build_ky("xw", 2022); KYi = build_ky("ipc4", 2022)
    parts = []
    for t0, B in D.groupby("t0"):
        B = B[["applicantCd"]].copy()
        o = pd.DataFrame(index=B.index)
        end = t0 * 10000 + 1231
        for tag, af, ky in [("chk", AF, KY), ("x", AFx, KYx), ("i", AFi, KYi)]:
            kn = af[(af.known_date <= end) & af.applicantCd.isin(set(B.applicantCd))]
            o[f"n_ksic_{tag}"] = B.applicantCd.map(kn.groupby("applicantCd").ksic.nunique()).fillna(0).values
            m3 = L.f_M3(B, t0, af, ky)
            o[f"k_rnd_g_{tag}"] = m3.k_rnd_g.values; o[f"k_rnd_sh_{tag}"] = m3.k_rnd_sh.values
            c6 = f_C06(B, t0, dict(AF=af, R=R, ns=ns, first=first))
            o[f"peer_entry_k_{tag}"] = c6.peer_entry_k.values
        parts.append(o)
        print("x", t0, flush=True)
    X = pd.concat(parts).loc[D.index]
    X.to_parquet(fx)
    return X


X = add_x(D)
D = pd.concat([D, X], axis=1)
chk = {c: float(np.mean(np.isclose(D[c].astype(float).fillna(-9), D[f"{c}_chk"].astype(float).fillna(-9), rtol=1e-6, atol=1e-9))) for c in KS}
SX = {c: f"{c}_x" for c in KS}; SI = {c: f"{c}_i" for c in KS}
VAR = {
    "CH": (FE, "동결 챔피언 재현"),
    "R_KSIC": ([f for f in FE if f not in KS], "제거: 제공 KSIC 라벨 피처 4개(n_ksic, k_rnd_g, k_rnd_sh, peer_entry_k)"),
    "R_PROV": ([f for f in FE if f not in PROV], "제거: KSIC 4개 + 제공 KR 출원 리스트 피처 5개 + 제공 특허×NTIS 성과 연결 8개(g_part_*)"),
    "R_PAT": ([f for f in FE if f not in PROV + REGB], "제거: 제공 범위 특허 전체(위 + 등록원부 행태·발명자) = NPS·NTIS 이력 없음 모집단의 외부데이터만"),
    "S_KSIC": ([SX.get(f, f) for f in FE], "대체: KSIC 4개를 공식 IPC-KSIC 연계표 배정으로 다시 계산(기업 구성·NTIS 산업 배분표·동종 진입률 모두)"),
    "S_IPC4": ([SI.get(f, f) for f in FE], "대체(보조): KSIC 4개를 메인 IPC 서브클래스(4자리) 그룹으로 다시 계산 — 연계표 없이 IPC만 쓰는 경우"),
}
fb = os.path.join(HERE, "base_M0_preds.json")
if os.path.exists(fb):
    BP = {int(k): np.array(v) for k, v in json.load(open(fb)).items()}
else:
    BP = {}
    for v in L.DEV_ORIGINS:
        tr = D[(D.t0 >= 2015) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        BP[v] = L.predict(tr[M0F].astype(float), tr.y, te[M0F].astype(float), model="xgb", seeds=SEEDS)
    json.dump({str(k): v.tolist() for k, v in BP.items()}, open(fb, "w"))
VAR["R_nksic"] = ([f for f in FE if f != "n_ksic"], "제거(분해): n_ksic만")
VAR["R_krnd"] = ([f for f in FE if f not in ("k_rnd_g", "k_rnd_sh")], "제거(분해): KSIC 정부연구비 성장·비중 2개만")
VAR["R_peerk"] = ([f for f in FE if f != "peer_entry_k"], "제거(분해): 동종 KSIC 진입률만")
VAR["S_krnd_only"] = ([SX.get(f, f) if f in ("k_rnd_g", "k_rnd_sh") else f for f in FE], "대체(분해): k_rnd 2개만 연계표로")
VAR["S_peerk_only"] = ([SX.get(f, f) if f == "peer_entry_k" else f for f in FE], "대체(분해): peer_entry_k만 연계표로")
for name in sys.argv[1:]:
    feats, desc = VAR[name]
    o = run_variant(D, feats, L.DEV_ORIGINS, L.predict, L.auc, SEEDS, "ens", os.path.join(HERE, f"res_{name}.json"), BP,
                    extra=dict(variant=name, desc=desc, recompute_match=chk, frozen_champ_auc=SPEC["dev"]["champ_auc"]))
    print(name, round(o["mean_auc"], 4), round(o["mean_skill"], 4), {k: round(r["auc"], 4) for k, r in o["by_origin"].items()}, o["sec"], "s", flush=True)
print("recompute_match", chk)
