# -*- coding: utf-8 -*-
"""N6 동결 실행기 (final_spec_r5.json).
  --mode dev    : 캐시 없이 패널을 다시 만들어 dev 원점 2014–2019에서 동결 챔피언·M0를 재채점하고 experiments.csv 값과 대조 → final_dev_check_r5.json
  --mode final  : 🔒 한 번만. 학습 t0 2012–2019(결과창 끝 <= 2021), 시험 t0=2021 → 출원 2022-01-01~2023-12-31 (F=2022).
                  AUC·skill(M0 대비)·상위 1/5/10% 적중률·사업자번호 군집 부트스트랩 95% CI(1,000회) → final_result_r5.json, final_preds_r5.parquet
                  보조: 로지스틱 M0+M1 vs 로지스틱 M0, 챔피언 vs M0-앙상블(강화 기준선).
  --mode future : final 이후에만. 학습 t0 2012–2021(결과 <= 2023), 점수 t0=2023(→ 2024–25 첫 특허 후보) → future_scores_r5.csv, future_ksic_r5.csv
이번 라운드(5R)는 --mode dev만 실행했다. final·future는 실행하지 않았다."""
import sys, json, argparse, subprocess, hashlib, warnings
import numpy as np, pandas as pd
from n6_lib import *
from n6_features import build
warnings.filterwarnings("ignore")

SPEC = json.load(open(os.path.join(OUT, "final_spec_r5.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; MSPEC = SPEC["model"]; M0F = SPEC["baseline"]["feats"]
M1F = ["ph_dev", "ph_app"] + ["t6_" + x for x in ["IT", "BT", "NT", "ET", "ST", "CT"]] \
    + ["cl_" + x for x in ["기계", "전기전자", "정보통신", "화공", "재료", "보건의료", "농림수산", "생명과학"]] + ["k50_assigned", "k50_score", "b1_score"]


def check_code():
    bad = [f for f, h in SPEC["code_sha256"].items() if hashlib.sha256(open(os.path.join(OUT, f), "rb").read()).hexdigest() != h]
    return bad


def metrics(y, p, pb):
    r = dict(n=int(len(y)), pos=int(y.sum()), base_rate=float(y.mean()), auc=float(auc(y, p)), auc_M0=float(auc(y, pb)))
    r["skill"] = float(skill(r["auc"], r["auc_M0"]))
    for fr in (0.01, 0.05, 0.10):
        k = f"top{int(fr * 100)}"
        r[k] = topk(y, p, fr); r[k + "_M0"] = topk(y, pb, fr); r[k + "_lift"] = r[k] / r["base_rate"]
    return r


def dev():
    D = build("dev"); D = D[D.t0 >= MIN_TRAIN]
    out = {}
    for v in DEV_ORIGINS:
        tr = D[(D.t0 <= v - 2)]; te = D[D.t0 == v]
        p = predict(tr, te, FEATS, spec=MSPEC); pb = predict(tr, te, M0F, spec={"model": "xgb", "seeds": [0]})
        out[v] = metrics(te.y.values, p, pb)
        log(v, round(out[v]["auc"], 4), round(out[v]["auc_M0"], 4), round(out[v]["skill"], 4))
    ref = SPEC["dev_reference"]
    match = all(abs(round(out[v]["auc"], 4) - a) < 1e-9 for v, a in zip(ref["origins"], ref["auc_champ"])) and \
        all(abs(round(out[v]["auc_M0"], 4) - a) < 1e-9 for v, a in zip(ref["origins"], ref["auc_M0"]))
    res = dict(mode="dev", code_changed=check_code(), match_experiments_csv=bool(match), by_origin=out,
               mean_skill=float(np.mean([out[v]["skill"] for v in DEV_ORIGINS])), mean_auc=float(np.mean([out[v]["auc"] for v in DEV_ORIGINS])),
               mean_auc_M0=float(np.mean([out[v]["auc_M0"] for v in DEV_ORIGINS])),
               mean_top1=float(np.mean([out[v]["top1"] for v in DEV_ORIGINS])), mean_top5=float(np.mean([out[v]["top5"] for v in DEV_ORIGINS])),
               mean_top10=float(np.mean([out[v]["top10"] for v in DEV_ORIGINS])), mean_base_rate=float(np.mean([out[v]["base_rate"] for v in DEV_ORIGINS])))
    json.dump(res, open(os.path.join(OUT, "final_dev_check_r5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("dev check match", match, "mean skill", round(res["mean_skill"], 4))


def final(force=False):
    fp = os.path.join(OUT, "final_result_r5.json")
    if os.path.exists(fp) and not force:
        raise SystemExit("final_result_r5.json 이 이미 있다 — 최종시험은 한 번만")
    assert not check_code(), f"동결 이후 코드 변경: {check_code()}"
    D = build("final")
    tr = D[(D.t0 >= SPEC["final"]["train_t0"][0]) & (D.t0 <= SPEC["final"]["train_t0"][1])]
    assert tr.t0.max() + 2 < SPEC["final"]["F"], "학습 레이블 결과창이 F 이후와 겹침"
    te = D[D.t0 == SPEC["final"]["test_t0"]]
    p = predict(tr, te, FEATS, spec=MSPEC); pb = predict(tr, te, M0F, spec={"model": "xgb", "seeds": [0]})
    y = te.y.values
    res = dict(mode="final", spec_champion=SPEC["champion"], train_n=int(len(tr)), train_t0=sorted(tr.t0.unique().tolist()), **metrics(y, p, pb))
    ci = boot_ci(y, p, pb, n=1000, seed=2021, groups=te.bz.values)
    res.update(skill_ci95=ci["skill_ci"], auc_ci95=ci["auc_ci"], auc_M0_ci95=ci["auc_m0_ci"])
    # 보조
    pl0 = predict(tr, te, M0F, spec={"model": "logit"}); pl1 = predict(tr, te, M0F + M1F, spec={"model": "logit"})
    ci2 = boot_ci(y, pl1, pl0, n=1000, seed=2022, groups=te.bz.values)
    res["secondary_logit_M0M1_vs_logit_M0"] = dict(auc=float(auc(y, pl1)), auc_ref=float(auc(y, pl0)), skill=float(skill(auc(y, pl1), auc(y, pl0))), skill_ci95=ci2["skill_ci"])
    pe = predict(tr, te, M0F, spec={"model": "ens", "seeds": [0], "w_xgb": 0.5})
    ci3 = boot_ci(y, p, pe, n=1000, seed=2023, groups=te.bz.values)
    res["secondary_vs_M0_ensemble"] = dict(auc_ref=float(auc(y, pe)), skill=float(skill(auc(y, p), auc(y, pe))), skill_ci95=ci3["skill_ci"])
    lk = te.linked.values == 1
    res["subgroup_linked_only"] = dict(n=int(lk.sum()), auc=float(auc(y[lk], p[lk])), auc_M0=float(auc(y[lk], pb[lk])))
    res["note"] = "KR 2023-12 출원 약 10% 미공개(공개 지연) — 모든 기업 공통, 보정하지 않음"
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame({"bz": te.bz.values, "y": y, "p_champ": p, "p_m0": pb}).to_parquet(os.path.join(OUT, "final_preds_r5.parquet"), index=False)
    log("final", json.dumps({k: res[k] for k in ["auc", "auc_M0", "skill", "skill_ci95", "top1", "top5", "top10"]}, ensure_ascii=False))


def future():
    assert os.path.exists(os.path.join(OUT, "final_result_r5.json")), "future는 final 이후에만"
    subprocess.run([sys.executable, os.path.join(OUT, "n6_text.py"), "2023"], check=True)   # 2022–23 과제 텍스트 유사도(중심은 고정)
    D = build("future", t0_list=T0_ALL_DEV + [2020, 2021, 2023])
    tr = D[(D.t0 >= MIN_TRAIN) & (D.t0 <= 2021)]; te = D[D.t0 == 2023].copy()
    te["score"] = predict(tr, te, FEATS, spec=MSPEC); te["rank_pct"] = te.score.rank(pct=True)
    pts = pd.read_parquet(os.path.join(CACHE, "proj_ptsim.parquet"), columns=["pjt_no", "year", "pt_top_ksic"])
    nt = load_ntis(2021, 2023).merge(pts, on=["pjt_no", "year"], how="left")
    top = nt.groupby("bz").pt_top_ksic.agg(lambda s: s.value_counts().index[0] if s.notna().any() else None)
    te["ksic_text"] = te.bz.map(top)
    te[["bz", "score", "rank_pct", "ksic_text"]].to_csv(os.path.join(OUT, "future_scores_r5.csv"), index=False, encoding="utf-8-sig")
    g = te.groupby("ksic_text").agg(n_firms=("bz", "size"), mean_rank=("rank_pct", "mean"), n_top10=("rank_pct", lambda s: int((s >= 0.9).sum())))
    g.sort_values("n_top10", ascending=False).to_csv(os.path.join(OUT, "future_ksic_r5.csv"), encoding="utf-8-sig")
    log("future", len(te))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["dev", "final", "future"], required=True); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"dev": dev, "final": lambda: final(a.force), "future": future}[a.mode]()
