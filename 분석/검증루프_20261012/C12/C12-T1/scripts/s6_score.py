# -*- coding: utf-8 -*-
"""C12-T1 s6: 시나리오 재학습·채점(동결 스펙 E21: 피처 25개, XGB 파라미터·시드 0·1·2). 사용: python s6_score.py <scen>
scen: off | s1 | s2 | s3 | s4 | s4b. s2 = off 패널에서 상위 5개 선언 기업 대응 출원인 단위를 학습·시험 모두 제외.
출력: cache/preds_<scen>.parquet, cache/score_<scen>.json"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import n5lib
from n5lib import fit_pred, auc, skill, cluster_boot_ci
TR = os.path.join(n5lib.MC, "N5_표준특허선언레이더")
SPEC = json.load(open(os.path.join(TR, "final_spec_r4.json"), encoding="utf-8"))
FE = SPEC["champion_features"]; F0 = SPEC["baseline_M0"]; FS = F0 + ["sg_rate", "sg_logn", "app_rec_rate", "mg_rec_rate"]
assert SPEC["xgb_params"] == n5lib.XGB_PARAMS and tuple(SPEC["seeds"]) == (0, 1, 2)
TOP5_APPS = ["삼성전자", "광동오포모바일텔레커뮤니케이션즈코포레이션리미티드", "퀄컴인코포레이티드", "엘지전자", "비보모바일커뮤니케이션컴퍼니리미티드"]
t = time.time()
scen = sys.argv[1]
src = "off" if scen == "s2" else scen
P = lambda t0: pd.read_parquet(os.path.join(W, "cache", "panels", f"{src}_{t0}.parquet"))
tr = pd.concat([P(t0) for t0 in range(2012, 2021)], ignore_index=True); te = P(2022)
assert tr.t0.max() + 2 < 2023
if scen == "s2":
    tr = tr[~tr.app.isin(TOP5_APPS)].reset_index(drop=True); te = te[~te.app.isin(TOP5_APPS)].reset_index(drop=True)
pr = {}
for nm, f in [("champ", FE), ("M0", F0), ("M0plus", FS)]:
    pr[nm] = fit_pred(tr[f].astype(float), tr.y, te[f].astype(float), model="xgb", seeds=(0, 1, 2))
    n5lib.tlog(t, scen, nm)
out = pd.DataFrame({"appl_no": te.appl_no.values, "app": te.app.values, "y": te.y.values, "ksic": te.ksic.astype(str).values,
                    **{f"p_{k}": v for k, v in pr.items()}})
out.to_parquet(os.path.join(W, "cache", f"preds_{scen}.parquet"), index=False)


def metrics(d, label):
    y = d.y.values
    a = {k: float(auc(y, d[f"p_{k}"].values)) for k in ["champ", "M0", "M0plus"]}
    (lo, hi), _ = cluster_boot_ci(y, d.p_champ.values, d.p_M0.values, d.app.values, n=1000, seed=2022)
    (los, his), _ = cluster_boot_ci(y, d.p_champ.values, d.p_M0plus.values, d.app.values, n=1000, seed=55)
    return dict(label=label, n=int(len(d)), pos=int(y.sum()), n_train=int(len(tr)), pos_train=int(tr.y.sum()), auc=a,
                skill_M0=skill(a["champ"], a["M0"]), ci_M0=[float(lo), float(hi)],
                skill_strong=skill(a["champ"], a["M0plus"]), ci_strong=[float(los), float(his)])


R = {"scen": scen, "main": metrics(out, "전체")}
if scen == "s3":
    R["sub26429"] = metrics(out[out.ksic == "26429"].reset_index(drop=True), "26429 부분집합(S3 학습 모형)")
    R["test_ksic"] = out.groupby("ksic").y.agg(["size", "sum"]).astype(int).to_dict("index")
R["secs"] = round(time.time() - t, 1)
json.dump(R, open(os.path.join(W, "cache", f"score_{scen}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
