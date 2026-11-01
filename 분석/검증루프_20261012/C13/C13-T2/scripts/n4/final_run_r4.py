# -*- coding: utf-8 -*-
r"""N4 동결 실행기. python final_run_r4.py --mode dev|final|future
- dev   : 개발 원점 2017–2021을 동결 명세로 처음부터 다시 만들어 채점 → final_dev_check_r4.json (champ.json 수치와 대조)
- final : 🔒 학습 t0 2015–2021(폐업 ≤2023) → 시험 t0=2023(폐업 2024-01-01~2025-12-31, F=2024). 1회만 실행. → final_result_r4.json
- future: 학습 t0 2015–2023 → 예측 t0=2024(폐업 2025–26, 레이블 없음) → future_pred_r4.parquet, future_ksic_risk_r4.csv
동결 명세: final_spec_r4.json. 공용 코드: n4lib.py, n4feat.py (동결 후 수정 금지).
중간 산출물은 cache/<mode>_* 로만 쓴다(dev 캐시와 섞지 않음)."""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
from n4lib import *
from n4feat import load_ctx, build_all

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["dev", "final", "future"])
ap.add_argument("--nboot", type=int, default=None)
args = ap.parse_args()
MODE = args.mode
SPEC = json.load(open(os.path.join(OUT, "final_spec_r4.json"), encoding="utf-8"))
FE, M0F = SPEC["features"], SPEC["baseline_M0"]
M1F = SPEC["M1_pilot"]
SEEDS = tuple(SPEC["model"]["seeds"])

if MODE == "dev":
    T0S = list(range(TRAIN_MIN_T0, 2022)); TASKS = [(list(range(TRAIN_MIN_T0, v - 1)), v) for v in DEV_ORIGINS]; NB = args.nboot or 300
elif MODE == "final":
    T0S = list(range(TRAIN_MIN_T0, 2022)) + [FINAL_T0]; TASKS = [(list(range(TRAIN_MIN_T0, 2022)), FINAL_T0)]; NB = args.nboot or 1000
else:
    T0S = list(range(TRAIN_MIN_T0, FUTURE_T0 + 1)); TASKS = [(list(range(TRAIN_MIN_T0, FINAL_T0 + 1)), FUTURE_T0)]; NB = 0

# 봉인 점검: 학습 레이블의 결과창이 시험 원점 이후와 겹치지 않는다
for tr, te in TASKS:
    assert max(tr) + 2 <= te or MODE == "future" and max(tr) + 2 <= FUTURE_T0 + 1, (tr, te)
    if MODE != "future":
        assert max(tr) + 2 < F_YEAR, "학습 레이블은 F(2024) 이전에 끝나야 한다"
if MODE == "dev":
    assert max(T0S) + 2 < F_YEAR
log("mode", MODE, "t0", T0S)

# ---------- 패널 구축 ----------
AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
R = load_registry(); ns = load_nts(); NPS = load_nps(MODE); A2K = a2key()
rows = []
for t0 in T0S:
    B = build_base(t0, MODE, AF, R, ns, NPS, A2K)
    rows.append(B); log("base", t0, len(B), "y=", B.y.sum() if B.y.notna().all() else "NaN")
D = pd.concat(rows, ignore_index=True)
ctx = load_ctx(MODE, set(D.applicantCd))
X = pd.concat([build_all(B, t0, ctx, AF, ns) for t0, B in D.groupby("t0")]).reindex(D.index)
D = pd.concat([D, X], axis=1)
D.to_parquet(os.path.join(CACHE, f"{MODE}_panel.parquet"), index=False)
log("panel", D.shape)


def fit(tr, te, feats, model):
    return predict(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, seeds=SEEDS)


out = {"mode": MODE, "spec": "final_spec_r4.json", "run_at": time.strftime("%Y-%m-%d %H:%M"), "results": []}
for tr_t0, te_t0 in TASKS:
    tr = D[D.t0.isin(tr_t0)]; te = D[D.t0 == te_t0]
    assert tr.y.notna().all()
    pc = fit(tr, te, FE, "ens")
    if MODE == "future":
        fut = te[["applicantCd", "bizno", "t0", "nps_present"]].copy(); fut["risk"] = pc; fut["risk_rank"] = rank01(pc)
        fut.to_parquet(os.path.join(OUT, "future_pred_r4.parquet"), index=False)
        # 특허 KSIC 라벨로 산업별 퇴조 위험 지수(보고서용 트렌드 산출물, 검증 대상 아님)
        end = te_t0 * 10000 + 1231
        W = AF[(AF.known_date <= end) & AF.applicantCd.isin(set(te.applicantCd))].groupby(["applicantCd", "ksic"]).appl_no.nunique().unstack(fill_value=0)
        W = W.div(W.sum(1), axis=0)
        r = fut.set_index("applicantCd").risk.reindex(W.index)
        K = pd.DataFrame({"firm_weight": W.sum(), "exp_risk": W.mul(r, axis=0).sum() / W.sum(),
                          "top10pct_share": W.mul((fut.set_index("applicantCd").risk_rank.reindex(W.index) >= 0.9).astype(float), axis=0).sum() / W.sum()})
        K.sort_values("exp_risk", ascending=False).to_csv(os.path.join(OUT, "future_ksic_risk_r4.csv"), encoding="utf-8-sig")
        out["results"].append(dict(t0=te_t0, n=len(te), train_t0=tr_t0, note="레이블 없음 — 예측만"))
        continue
    assert te.y.notna().all()
    y = te.y.values
    p0 = fit(tr, te, M0F, "xgb"); p0e = fit(tr, te, M0F, "ens"); p1 = fit(tr, te, M1F, "xgb"); pl = fit(tr, te, FE, "logit")
    k = max(1, int(0.01 * len(te)))
    top = lambda p: float(y[np.argsort(-p)[:k]].mean())
    ci = cluster_boot(y, pc, p0, te.bizno.values, n=NB, seed=te_t0)
    ce = cluster_boot(y, pc, p0e, te.bizno.values, n=NB, seed=te_t0 + 1)
    m = te.nps_present.values == 0
    a = dict(champ=auc(y, pc), M0_xgb=auc(y, p0), M0_ens=auc(y, p0e), M1_xgb=auc(y, p1), champ_logit_only=auc(y, pl))
    res = dict(t0=te_t0, train_t0=tr_t0, n=int(len(te)), pos=int(y.sum()), base_rate=float(y.mean()), n_clusters=int(te.bizno.nunique()),
               auc={k_: float(v) for k_, v in a.items()},
               skill_vs_M0=float(skill(a["champ"], a["M0_xgb"])), skill_vs_M0_ci95=ci["skill_ci"],
               skill_vs_M0ens=float(skill(a["champ"], a["M0_ens"])), skill_vs_M0ens_ci95=ce["skill_ci"],
               auc_champ_ci95=ci["auc_model_ci"], auc_M0_ci95=ci["auc_base_ci"], n_boot=ci["n_boot"],
               skill_M1_vs_M0=float(skill(a["M1_xgb"], a["M0_xgb"])),
               prauc=dict(champ=float(prauc(y, pc)), M0=float(prauc(y, p0))),
               top1pct_precision=dict(champ=top(pc), M0=top(p0), M1=top(p1)),
               nonnps=dict(n=int(m.sum()), rate=float(y[m].mean()), auc_M0=float(auc(y[m], p0[m])), auc_champ=float(auc(y[m], pc[m]))))
    out["results"].append(res)
    log(te_t0, "AUC champ", round(a["champ"], 4), "M0", round(a["M0_xgb"], 4), "skill", round(res["skill_vs_M0"], 4), ci["skill_ci"])

if MODE == "dev":
    ref = {int(k): v for k, v in SPEC["dev"]["champ_auc"].items()}
    out["reproduce_check"] = {str(r["t0"]): dict(auc_now=r["auc"]["champ"], auc_frozen=ref[r["t0"]], diff=r["auc"]["champ"] - ref[r["t0"]]) for r in out["results"]}
    out["mean_skill_vs_M0"] = float(np.mean([r["skill_vs_M0"] for r in out["results"]]))
    fn = "final_dev_check_r4.json"
elif MODE == "final":
    fn = "final_result_r4.json"
else:
    fn = "future_run_r4.json"
json.dump(out, open(os.path.join(OUT, fn), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("saved", fn)
