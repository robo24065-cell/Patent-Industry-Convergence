# -*- coding: utf-8 -*-
r"""동결 스펙(R03 47피처, XGB 3시드+로지스틱 순위 평균)으로 학습 t0 2015–2020 → 시험 t0=2022 재학습·채점.
--panel orig : 재현(동결 final_test_pred_r5.parquet와 예측 비교)
--panel alias: NPS 별칭 보정 패널 → 공식 성적은 바꾸지 않고 영향만 보고
기준선: M0(xgb3, 공식), SB(xgb3), M0-ens, SB-ens, 외부데이터 전부(EXT_ALL ens·xgb3). 사업자번호 군집 부트스트랩 1,000회."""
import argparse
from w_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--panel", required=True, choices=["orig", "alias"]); ap.add_argument("--nboot", type=int, default=1000); a = ap.parse_args()
D = pd.read_parquet(os.path.join(VC, f"final_panel_{a.panel}.parquet"))
tr = D[D.t0 <= 2020].reset_index(drop=True); te = D[D.t0 == 2022].reset_index(drop=True)
assert tr.t0.max() + 2 < n7lib.F_YEAR and 2021 not in set(D.t0)
y = te.y.values; cl = te.bizno.values; y1 = te.y1.values
fit = lambda feats, model, yc="y": predict(tr[feats].astype(float), tr[yc], te[feats].astype(float), model=model, seeds=SEEDS)
P = {}
P["champ"] = fit(FE, "ens"); log("champ")
P["M0"] = fit(M0F, "xgb"); P["SB"] = fit(SBF, "xgb"); P["pilot"] = fit(PLF, "xgb")
P["M0_ens"] = fit(M0F, "ens"); P["SB_ens"] = fit(SBF, "ens"); P["logit"] = fit(FE, "logit")
P["champ_xgb"] = fit(FE, "xgb")
P["EXT_ens"] = fit(EXT_ALL, "ens"); P["EXT_xgb"] = fit(EXT_ALL, "xgb"); log("baselines")
P["champ_y1"] = fit(FE, "ens", "y1"); P["M0_y1"] = fit(M0F, "xgb", "y1"); log("y1")
pr = te[["applicantCd", "bizno", "t0", "y", "y1", "nps_present"]].copy()
for k, v in P.items():
    pr["p_" + k] = v
pr.to_parquet(os.path.join(VC, f"preds_{a.panel}.parquet"), index=False)
A = {k: float(auc(y if not k.endswith("_y1") else y1, v)) for k, v in P.items()}
res = dict(panel=a.panel, n=int(len(te)), pos=int(y.sum()), auc=A)
res["skill"] = {b: float(skill(A["champ"], A[b])) for b in ["M0", "SB", "M0_ens", "SB_ens", "EXT_ens", "EXT_xgb"]}
res["skill_ci"] = {}
for i, b in enumerate(["M0", "SB", "M0_ens", "EXT_ens"]):
    c = cluster_boot(y, P["champ"], P[b], cl, n=a.nboot, seed=2022 + i)
    res["skill_ci"][b] = c["skill_ci"]
    if b == "M0":
        res["auc_champ_ci"] = c["auc_model_ci"]; res["auc_M0_ci"] = c["auc_base_ci"]
log("boot")
res["y1"] = dict(pos=int(y1.sum()), auc_champ=A["champ_y1"], auc_M0=A["M0_y1"], skill=float(skill(A["champ_y1"], A["M0_y1"])),
                 skill_ci=cluster_boot(y1, P["champ_y1"], P["M0_y1"], cl, n=a.nboot, seed=7)["skill_ci"])
res["topk"] = {}
for k in ["champ", "M0", "SB", "EXT_ens"]:
    res["topk"][k] = {nm: dict(zip(["precision", "recall"], topk(y, P[k], fr))) for fr, nm in [(0.01, "top1pct"), (0.05, "top5pct"), (0.10, "top10pct")]}
m = te.nps_present.values == 0
res["nonnps"] = dict(n=int(m.sum()), pos=int(y[m].sum()), rate=float(y[m].mean()), auc_M0=float(auc(y[m], P["M0"][m])), auc_champ=float(auc(y[m], P["champ"][m])),
                     auc_EXT=float(auc(y[m], P["EXT_ens"][m])))
if a.panel == "orig":
    F = pd.read_parquet(os.path.join(TR, "final_test_pred_r5.parquet"))
    mm = F.merge(pr, on=["applicantCd", "t0"], suffixes=("_f", ""))
    res["repro_vs_frozen"] = dict(rows=len(mm), maxdiff_champ=float((mm.p_champ_f - mm.p_champ).abs().max()), maxdiff_M0=float((mm.p_M0_f - mm.p_M0).abs().max()),
                                  maxdiff_SB=float((mm.p_SB_f - mm.p_SB).abs().max()), y_match=float((mm.y_f == mm.y).mean()))
json.dump(res, open(os.path.join(HERE, f"w2_fit_{a.panel}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(json.dumps({k: res[k] for k in ["auc", "skill", "skill_ci", "nonnps"]}, ensure_ascii=False))
