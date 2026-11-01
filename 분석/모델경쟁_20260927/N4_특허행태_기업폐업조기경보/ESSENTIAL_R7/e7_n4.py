# -*- coding: utf-8 -*-
r"""N4 R7 필수성 검증: 동결 챔피언(C12, ens, 시드 0·1·2) 설정 그대로 피처만 바꿔 개발 원점 2017–2021 재학습.
python e7_n4.py <VARIANT ...>   (VARIANT: CH R_KSIC R_PROV R_PAT S_KSIC)
읽기: ..\cache\dev_panel.parquet, ..\cache\app_firm.parquet, ..\final_spec_r4.json (읽기 전용). 쓰기: 이 폴더만."""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
sys.path.insert(0, TR); sys.path.insert(0, os.path.join(MC, "T4_특허포기예측", "ESSENTIAL_R7", "common"))
import numpy as np, pandas as pd
from n4lib import predict, auc, DEV_ORIGINS
from firmrun import run_variant
from xwalk import load_app_xw

SPEC = json.load(open(os.path.join(TR, "final_spec_r4.json"), encoding="utf-8"))
FE, M0F, SEEDS = SPEC["features"], SPEC["baseline_M0"], tuple(SPEC["model"]["seeds"])
D = pd.read_parquet(os.path.join(TR, "cache", "dev_panel.parquet"))

KS = ["n_ksic"]
PROV = ["n_ksic", "log_app", "yrs_since_last", "app_trend", "known_1y", "live_ratio"]
REGB = ["log_live", "lapse_1y", "lapse_2y", "pay_1y", "pay_trend", "trf_2y", "late_share_2y", "delay_mean_2y", "overdue_share",
        "rec_2y", "trf_out_2y", "merge_2y", "surrender_2y", "trf_out_1y"]


def add_x(D):
    fx = os.path.join(HERE, "n_ksic_x_dev.parquet")
    if os.path.exists(fx):
        x = pd.read_parquet(fx)
    else:
        AF = pd.read_parquet(os.path.join(TR, "cache", "app_firm.parquet"))
        AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()].copy()
        xw = load_app_xw().set_index("appl_no").xw
        AF["xw"] = AF.appl_no.astype(str).map(xw)
        rows = []
        for t0 in sorted(D.t0.unique()):
            kn = AF[AF.known_date <= t0 * 10000 + 1231]
            g = kn.groupby("applicantCd")
            rows.append(pd.DataFrame({"t0": t0, "n_ksic_chk": g.ksic.nunique(), "n_ksic_x": g.xw.nunique(),
                                      "xw_cover": g.xw.apply(lambda s: s.notna().mean())}).reset_index())
        x = pd.concat(rows, ignore_index=True)
        x.to_parquet(fx, index=False)
    D = D.merge(x, on=["applicantCd", "t0"], how="left")
    return D


D = add_x(D)
chk = float((D.n_ksic_chk == D.n_ksic).mean())
VAR = {
    "CH": (FE, "동결 챔피언 재현"),
    "R_KSIC": ([f for f in FE if f not in KS], "제거: 제공 KSIC 라벨 피처(n_ksic)"),
    "R_PROV": ([f for f in FE if f not in PROV], "제거: 제공 KR 출원 리스트 피처(n_ksic·출원 수·최근성·추세·공개 1년·생존비)"),
    "R_PAT": ([f for f in FE if f not in PROV + REGB], "제거: 제공 범위 특허 전체(출원 리스트+등록원부 행태) = 외부데이터(NPS·NTIS·DART·국세청)만"),
    "S_KSIC": ([("n_ksic_x" if f == "n_ksic" else f) for f in FE], "대체: n_ksic → 공식 IPC-KSIC 연계표 배정 코드 수(n_ksic_x)"),
}

fb = os.path.join(HERE, "base_M0_preds.json")
if os.path.exists(fb):
    BP = {int(k): np.array(v) for k, v in json.load(open(fb)).items()}
else:
    BP = {}
    for v in DEV_ORIGINS:
        tr = D[(D.t0 >= 2015) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        BP[v] = predict(tr[M0F].astype(float), tr.y, te[M0F].astype(float), model="xgb", seeds=SEEDS)
    json.dump({str(k): v.tolist() for k, v in BP.items()}, open(fb, "w"))
for name in sys.argv[1:]:
    feats, desc = VAR[name]
    o = run_variant(D, feats, DEV_ORIGINS, predict, auc, SEEDS, "ens", os.path.join(HERE, f"res_{name}.json"), BP,
                    extra=dict(variant=name, desc=desc, n_ksic_recompute_match=chk,
                               frozen_champ_auc=SPEC["dev"]["champ_auc"]))
    print(name, round(o["mean_auc"], 4), round(o["mean_skill"], 4), {k: round(r["auc"], 4) for k, r in o["by_origin"].items()}, o["sec"], "s", flush=True)
