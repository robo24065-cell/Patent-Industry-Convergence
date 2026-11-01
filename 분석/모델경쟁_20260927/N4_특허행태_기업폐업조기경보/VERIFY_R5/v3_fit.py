# -*- coding: utf-8 -*-
"""R5 레드팀(N4) 3단계: 새 패널로 동결 스펙 재학습. part=main(공식 재현) | sens1·sens2(강화 기준선·묶음 제거) | xgbonly(보정용)"""
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import n4lib
n4lib.CACHE = os.path.join(HERE, "cache")
from n4lib import predict, auc, skill
from n4feat import BUNDLES
import numpy as np, pandas as pd
part = sys.argv[1]
SPEC = json.load(open(os.path.join(TR, "final_spec_r4.json"), encoding="utf-8"))
FE, M0F, M1F = SPEC["features"], SPEC["baseline_M0"], SPEC["M1_pilot"]
D = pd.read_parquet(os.path.join(HERE, "cache", "final_panel.parquet"))
tr = D[D.t0.between(2015, 2021)]; te = D[D.t0 == 2023]
assert tr.t0.max() + 2 < 2024 and tr.y.notna().all() and te.y.notna().all()
t = time.time()
fit = lambda f, m: predict(tr[f].astype(float), tr.y, te[f].astype(float), model=m, seeds=(0, 1, 2))
P = {}
if part == "main":
    P["champ"] = fit(FE, "ens"); P["M0"] = fit(M0F, "xgb"); P["M0_ens"] = fit(M0F, "ens"); P["M1"] = fit(M1F, "xgb")
elif part == "sens1":
    EXT = M0F + BUNDLES["C01_nps_exit"] + BUNDLES["C02_ntis_project"] + ["dart_has", "dart_impair", "dart_neg_eq"]
    P["EXT_ens"] = fit(EXT, "ens")                       # 강화 기준선: 동결 모형의 비특허(외부) 피처 전부, 같은 모형
    P["M0C01_ens"] = fit(M0F + BUNDLES["C01_nps_exit"], "ens")   # 강화 기준선: NPS 전체
elif part == "xgbonly":
    P["champ_xgb"] = fit(FE, "xgb")   # 보정 곡선용 확률(동결 챔피언은 순위 평균이라 확률이 아님)
elif part == "sens2":
    P["champ_noC11"] = fit([f for f in FE if f not in BUNDLES["C11_disposal"]], "ens")
    P["champ_noC04"] = fit([f for f in FE if f not in BUNDLES["C04_pay_delay"]], "ens")
    P["champ_noDART"] = fit([f for f in FE if not f.startswith("dart_")], "ens")
print({k: round(float(auc(te.y, v)), 5) for k, v in P.items()}, round(time.time() - t), flush=True)
out = pd.DataFrame({"applicantCd": te.applicantCd.values, "y": te.y.values, **{f"p_{k}": v for k, v in P.items()}})
out.to_parquet(os.path.join(HERE, "cache", f"preds_{part}.parquet"), index=False)
