# -*- coding: utf-8 -*-
r"""N7 동결 실행기. python final_run_r5.py --mode dev|final|future
- dev   : 개발 원점 2017–2020을 동결 명세로 처음부터 다시 만들어 채점 → final_dev_check_r5.json (champ.json 수치와 대조). NTIS 과제연도 ≤2022만 읽는다.
- final : 🔒 학습 t0 2015–2020(과제연도 ≤2022) → 시험 t0=2022(과제연도 2023–2024 첫 수행, F=2023). **1회만 실행** → final_result_r5.json, final_test_pred_r5.parquet
          AUC(챔피언·M0·SB·파일럿·M0-ens·SB-ens·로지스틱만), skill(vs M0 주 지표, vs SB, vs M0-ens), 상위 1/5/10% 적중률·상위 10% 재현율,
          사업자번호 군집 부트스트랩 95% CI(1,000회), 사전등록 보조: 1년 결과창(2023만), NPS 미가입 하위집단.
- future: 학습 t0 2015–2022 → 예측 t0=2024(과제연도 2025–26, 레이블 없음) → future_pred_r5.parquet, future_ksic_entry_r5.csv
동결 명세: final_spec_r5.json. 공용 코드: n7lib.py, n7feat.py (동결 후 수정 금지). 중간 산출물은 cache/<mode>_* 로만 쓴다."""
import os, sys, json, argparse, time, warnings
import numpy as np, pandas as pd
from n7lib import *
from n7feat import load_ctx, build_features
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["dev", "final", "future"])
ap.add_argument("--nboot", type=int, default=None)
args = ap.parse_args()
MODE = args.mode
SPEC = json.load(open(os.path.join(OUT, "final_spec_r5.json"), encoding="utf-8"))
FE, M0F, SBF, PLF = SPEC["features"], SPEC["baseline_M0"], SPEC["strong_baseline_SB"], SPEC["pilot_champion"]
SEEDS = tuple(SPEC["model"]["seeds"]); CM = SPEC["model"]["type"]
FRS = [(0.01, "top1pct"), (0.05, "top5pct"), (0.10, "top10pct")]

if MODE == "dev":
    TASKS = [(list(range(TRAIN_MIN_T0, v - 1)), v) for v in DEV_ORIGINS]; NB = args.nboot or 300; FN = "final_dev_check_r5.json"
elif MODE == "final":
    TASKS = [(list(range(TRAIN_MIN_T0, 2021)), FINAL_T0)]; NB = args.nboot or 1000; FN = "final_result_r5.json"
    if os.path.exists(os.path.join(OUT, FN)):
        sys.exit("final_result_r5.json이 이미 있다 — 최종시험은 1회만 실행한다.")
else:
    TASKS = [(list(range(TRAIN_MIN_T0, FINAL_T0 + 1)), FUTURE_T0)]; NB = 0; FN = "future_run_r5.json"

# 봉인 점검: 학습 레이블 결과창(t0+1..t0+2)이 시험 원점 이후·F와 겹치지 않는다
for tr, te in TASKS:
    assert max(tr) + 2 <= te, (tr, te)
    if MODE != "future":
        assert max(tr) + 2 < F_YEAR, "학습 레이블은 F(2023) 이전에 끝나야 한다"
        assert 2021 not in tr
log("mode", MODE, "tasks", TASKS)

# ---------- 패널·피처 ----------
D = build_panel(MODE)
if MODE == "dev":
    assert D.t0.max() + 2 < F_YEAR
ctx = load_ctx(MODE)
X = build_features(D, ctx, SPEC["feature_bundles_needed"])
D = pd.concat([D, X], axis=1)
D.to_parquet(os.path.join(CACHE, f"{MODE}_panel.parquet"), index=False)
log("panel", D.shape)


def fit(tr, te, feats, model, ycol="y"):
    return predict(tr[feats].astype(float), tr[ycol], te[feats].astype(float), model=model, seeds=SEEDS)


def topd(y, p):
    o = {}
    for fr, nm in FRS:
        prec, rec = topk(y, p, fr)
        o[nm] = dict(k=int(max(1, round(fr * len(y)))), precision=prec, recall=rec)
    return o


out = {"mode": MODE, "spec": "final_spec_r5.json", "run_at": time.strftime("%Y-%m-%d %H:%M"), "source": "NTIS Open API 활용", "results": []}
for tr_t0, te_t0 in TASKS:
    tr = D[D.t0.isin(tr_t0)]; te = D[D.t0 == te_t0]
    assert tr.y.notna().all()
    pc = fit(tr, te, FE, CM)
    if MODE == "future":
        fut = te[["applicantCd", "bizno", "t0", "nps_present"]].copy(); fut["score"] = pc; fut["score_rank"] = rank01(pc)
        fut.to_parquet(os.path.join(OUT, "future_pred_r5.parquet"), index=False)
        # 특허 KSIC 라벨로 산업별 '정부R&D 신규 진입 대기' 지수(보고서용, 검증 대상 아님). 개별 기업은 공개하지 않는다.
        W = ksic_weights(te, te_t0, ctx["AF"]).pivot_table(index="applicantCd", columns="ksic", values="w", fill_value=0)
        r = fut.set_index("applicantCd").score_rank.reindex(W.index)
        K = pd.DataFrame({"firm_weight": W.sum(), "mean_rank": W.mul(r, axis=0).sum() / W.sum(),
                          "top10pct_share": W.mul((r >= 0.9).astype(float), axis=0).sum() / W.sum()})
        K.sort_values("mean_rank", ascending=False).to_csv(os.path.join(OUT, "future_ksic_entry_r5.csv"), encoding="utf-8-sig")
        out["results"].append(dict(t0=te_t0, n=len(te), train_t0=tr_t0, note="레이블 없음 — 예측만"))
        continue
    assert te.y.notna().all()
    y = te.y.values; cl = te.bizno.values
    p0 = fit(tr, te, M0F, "xgb"); psb = fit(tr, te, SBF, "xgb"); ppl = fit(tr, te, PLF, "xgb")
    p0e = fit(tr, te, M0F, "ens"); psbe = fit(tr, te, SBF, "ens"); pl = fit(tr, te, FE, "logit")
    a = dict(champ=auc(y, pc), M0=auc(y, p0), SB=auc(y, psb), pilot=auc(y, ppl), M0_ens=auc(y, p0e), SB_ens=auc(y, psbe), champ_logit_only=auc(y, pl))
    ci = cluster_boot(y, pc, p0, cl, n=NB, seed=te_t0)
    cs = cluster_boot(y, pc, psb, cl, n=NB, seed=te_t0 + 1)
    ce = cluster_boot(y, pc, p0e, cl, n=NB, seed=te_t0 + 2)
    # 사전등록 보조: 1년 결과창(t0+1만) — 같은 학습 원점, 학습 레이블도 1년창
    q1 = fit(tr, te, FE, CM, "y1"); q0 = fit(tr, te, M0F, "xgb", "y1"); y1 = te.y1.values
    c1 = cluster_boot(y1, q1, q0, cl, n=NB, seed=te_t0 + 3)
    m = te.nps_present.values == 0
    res = dict(t0=te_t0, train_t0=tr_t0, n=int(len(te)), pos=int(y.sum()), base_rate=float(y.mean()), n_clusters=int(te.bizno.nunique()),
               auc={k: float(v) for k, v in a.items()}, auc_champ_ci95=ci["auc_model_ci"], auc_M0_ci95=ci["auc_base_ci"],
               skill_vs_M0=float(skill(a["champ"], a["M0"])), skill_vs_M0_ci95=ci["skill_ci"],
               skill_vs_SB=float(skill(a["champ"], a["SB"])), skill_vs_SB_ci95=cs["skill_ci"],
               skill_vs_M0ens=float(skill(a["champ"], a["M0_ens"])), skill_vs_M0ens_ci95=ce["skill_ci"],
               skill_pilot_vs_M0=float(skill(a["pilot"], a["M0"])), n_boot=ci["n_boot"],
               topk=dict(champ=topd(y, pc), M0=topd(y, p0), SB=topd(y, psb)),
               one_year_window=dict(pos=int(y1.sum()), base_rate=float(y1.mean()), auc_champ=float(auc(y1, q1)), auc_M0=float(auc(y1, q0)),
                                    skill_vs_M0=float(skill(auc(y1, q1), auc(y1, q0))), skill_ci95=c1["skill_ci"]),
               nonnps=dict(n=int(m.sum()), rate=float(y[m].mean()), auc_M0=float(auc(y[m], p0[m])), auc_champ=float(auc(y[m], pc[m]))))
    out["results"].append(res)
    if MODE == "final":
        pr = te[["applicantCd", "bizno", "t0", "y", "y1"]].copy(); pr["p_champ"] = pc; pr["p_M0"] = p0; pr["p_SB"] = psb
        pr.to_parquet(os.path.join(OUT, "final_test_pred_r5.parquet"), index=False)
    log(te_t0, "AUC champ", round(a["champ"], 4), "M0", round(a["M0"], 4), "SB", round(a["SB"], 4), "skill", round(res["skill_vs_M0"], 4), ci["skill_ci"])

if MODE == "dev":
    ref = {int(k): v for k, v in SPEC["dev"]["champ_auc"].items()}
    out["reproduce_check"] = {str(r["t0"]): dict(auc_now=r["auc"]["champ"], auc_frozen=ref[r["t0"]], diff=r["auc"]["champ"] - ref[r["t0"]]) for r in out["results"]}
    out["mean_skill_vs_M0"] = float(np.mean([r["skill_vs_M0"] for r in out["results"]]))
    out["mean_skill_vs_SB"] = float(np.mean([r["skill_vs_SB"] for r in out["results"]]))
    out["mean_auc_champ"] = float(np.mean([r["auc"]["champ"] for r in out["results"]]))
json.dump(out, open(os.path.join(OUT, FN), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("saved", FN)
