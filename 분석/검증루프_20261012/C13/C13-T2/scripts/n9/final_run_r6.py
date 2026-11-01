# -*- coding: utf-8 -*-
"""N9 동결 실행기 (final_spec_r6.json). 출처: NTIS Open API 활용.
  --mode dev --part 1 : 캐시 없이 중심점 유사도(n9_text.py)·dev 패널·텍스트 분류기 점수를 다시 만들어 cache/devcheck_* 에 저장, 기존 캐시와 대조
  --mode dev --part 2 : devcheck 산출물로 dev 원점 2014–2020에서 동결 챔피언·기준선들을 재채점, experiments.csv 값과 대조 → final_dev_check_r6.json
  --mode final        : 🔒 한 번만. 학습 t 2010–2020(결과창 끝 <= 2021 < F), 시험 t=2022 → 출원 2022-01-01~2023-12-31 (F=2022).
                        AUC·skill(M0_L2 대비, 주)·원판 M0 대비·강한 기준선(M0_L2+M2_L2) 대비·상위 1/5/10%·상위 100건·수행기관 군집 부트스트랩 95% CI(1,000회)
                        보조: 원판 챔피언, 파일럿 설계 모형(M0+M1+M2 지연판·원판), 로지스틱 챔피언, 수행주체별 하위집단 → final_result_r6.json, final_preds_r6.parquet
  --mode future       : final 이후에만. 과제연도 2023·2024 점수(학습 t <= v-2) → future_scores_r6.csv, future_ksic_r6.csv
6라운드에서는 --mode dev만 실행했다. final·future는 실행하지 않았다."""
import sys, json, argparse, hashlib, subprocess, warnings
import numpy as np, pandas as pd
from n9_lib import *
warnings.filterwarnings("ignore")

SPEC = json.load(open(os.path.join(OUT, "final_spec_r6.json"), encoding="utf-8"))
FEATS = SPEC["feats"]
BASE_SETS = {k: v["feats"] for k, v in SPEC["references"].items()}


def check_code():
    return [f for f, h in SPEC["code_sha256"].items() if hashlib.sha256(open(os.path.join(OUT, f), "rb").read()).hexdigest() != h]


def with_txt(D, mode, v):
    tc = pd.read_parquet(f"{CACHE}/txtclf_{mode}_{v}.parquet")
    X = D.copy(); X["txt_clf"] = np.nan; X.loc[tc.row_id.values, "txt_clf"] = tc.txt_clf.values
    return X


def train_test(D, v, train_lo, train_hi):
    rng = np.random.default_rng(0)
    keep = (D.y.values == 1) | (rng.random(len(D)) < NEG_FRAC)
    w = np.where(D.y.values == 1, 1.0, 1.0 / NEG_FRAC)
    msk = (D.t0 >= train_lo).values & (D.t0 <= train_hi).values & keep
    return msk, w


def predict(X, v, feats, train_lo, train_hi, model="xgb"):
    msk, w = train_test(X, v, train_lo, train_hi)
    tr = X[msk]; te = X[X.t0 == v]
    assert (tr.y >= 0).all()
    return fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, w=w[msk])


def dev(part):
    import n9_build, n9_textclf
    if part == 1:
        subprocess.run([sys.executable, os.path.join(OUT, "n9_text.py")], check=True)   # 중심점 유사도 재계산(같은 파일에 결정적으로 다시 씀)
        D = n9_build.build("dev", save=False)
        old = pd.read_parquet(f"{CACHE}/panel_dev.parquet")
        same = bool(len(D) == len(old) and all(np.allclose(pd.to_numeric(D[c], errors="coerce").astype(float).fillna(-999), pd.to_numeric(old[c], errors="coerce").astype(float).fillna(-999))
                                                  if D[c].dtype.kind in "fiu" else (D[c].astype(str).values == old[c].astype(str).values).all() for c in old.columns))
        D.to_parquet(f"{CACHE}/panel_devcheck.parquet", index=False)
        n9_textclf.run("devcheck", DEV_ORIGINS, D=D[["pjt_no", "t0", "y"]])
        tsame = all(np.allclose(pd.read_parquet(f"{CACHE}/txtclf_devcheck_{v}.parquet").txt_clf, pd.read_parquet(f"{CACHE}/txtclf_dev_{v}.parquet").txt_clf, atol=1e-5) for v in DEV_ORIGINS)
        json.dump(dict(panel_same_as_cache=same, txtclf_same_as_cache=bool(tsame)), open(f"{CACHE}/devcheck_part1.json", "w"), indent=1)
        log("part1 panel same", same, "txtclf same", tsame)
        return
    D = pd.read_parquet(f"{CACHE}/panel_devcheck.parquet").reset_index(drop=True)
    p1 = json.load(open(f"{CACHE}/devcheck_part1.json"))
    out = {}
    for v in DEV_ORIGINS:
        X = with_txt(D, "devcheck", v); te = X[X.t0 == v]; y = te.y.values
        p = predict(X, v, FEATS, 2010, v - GAP)
        refp = {k: predict(X, v, fs, 2010, v - GAP) for k, fs in BASE_SETS.items()}
        r = metrics(y, p, refp["M0_L2"], prefix_ref="M0L2")
        for k, pp in refp.items():
            r[f"auc_{k}"] = float(auc(y, pp)); r[f"skill_vs_{k}"] = float(skill(r["auc"], r[f"auc_{k}"]))
        out[v] = r; log(v, round(r["auc"], 4), round(r["auc_M0L2"], 4), round(r["skill"], 4))
    ref = SPEC["dev_reference"]
    match = all(abs(round(out[v]["auc"], 4) - a) < 1e-9 for v, a in zip(ref["origins"], ref["auc_champ"])) and \
        all(abs(round(out[v]["auc_M0L2"], 4) - a) < 1e-9 for v, a in zip(ref["origins"], ref["auc_M0_L2"]))
    mean = lambda k: float(np.mean([out[v][k] for v in DEV_ORIGINS]))
    res = dict(mode="dev", code_changed=check_code(), part1=p1, match_experiments_csv=bool(match), by_origin=out,
               mean_auc=mean("auc"), mean_auc_M0L2=mean("auc_M0L2"), mean_skill=mean("skill"),
               mean_skill_vs={k: mean(f"skill_vs_{k}") for k in BASE_SETS}, mean_top1=mean("top1"), mean_top5=mean("top5"), mean_top10=mean("top10"),
               mean_top1_M0L2=mean("top1_M0L2"), mean_hit100=mean("hit100"), mean_base_rate=mean("base_rate"))
    json.dump(res, open(os.path.join(OUT, "final_dev_check_r6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("dev check match", match, "mean skill", round(res["mean_skill"], 4))


def final(force=False):
    import n9_build, n9_textclf
    fp = os.path.join(OUT, "final_result_r6.json")
    if os.path.exists(fp) and not force:
        raise SystemExit("final_result_r6.json 이 이미 있다 — 최종시험은 한 번만")
    assert not check_code(), f"동결 이후 코드 변경: {check_code()}"
    fz = SPEC["final"]
    D = n9_build.build("final", save=True).reset_index(drop=True)
    assert set(D.t0.unique()) <= set(range(fz["train_t"][0], fz["train_t"][1] + 1)) | {fz["test_t"]}
    assert fz["train_t"][1] + 1 < F, "학습 레이블 결과창이 F 이후와 겹침"
    n9_textclf.run("final", [fz["test_t"]], D=D[["pjt_no", "t0", "y"]])
    v = fz["test_t"]; X = with_txt(D, "final", v); te = X[X.t0 == v]; y = te.y.values; g = te.org.values
    p = predict(X, v, FEATS, *fz["train_t"])
    refp = {k: predict(X, v, fs, *fz["train_t"]) for k, fs in BASE_SETS.items()}
    res = dict(mode="final", spec_champion=SPEC["champion"], train_n=int(((X.t0 >= fz["train_t"][0]) & (X.t0 <= fz["train_t"][1])).sum()),
               **metrics(y, p, refp["M0_L2"], prefix_ref="M0L2"))
    ci = boot_ci(y, p, refp["M0_L2"], groups=g, n=1000, seed=2022)
    res.update(skill_ci95=ci["skill_ci"], auc_ci95=ci["auc_ci"], auc_M0L2_ci95=ci["auc_ref_ci"])
    for k, pp in refp.items():
        res[f"auc_{k}"] = float(auc(y, pp)); res[f"skill_vs_{k}"] = float(skill(res["auc"], res[f"auc_{k}"]))
        res[f"top1_{k}"] = topk(y, pp, 0.01)[0]
    ci2 = boot_ci(y, p, refp["strong_L2"], groups=g, n=1000, seed=2023); res["skill_vs_strong_L2_ci95"] = ci2["skill_ci"]
    ci3 = boot_ci(y, p, refp["M0_orig"], groups=g, n=1000, seed=2024); res["skill_vs_M0_orig_ci95"] = ci3["skill_ci"]
    # 보조
    orig = [f[:-3] if f.endswith("_L2") and f[:-3] in LAGGED else f for f in FEATS]
    po = predict(X, v, orig, *fz["train_t"])
    res["secondary_orig_champion"] = dict(auc=float(auc(y, po)), skill_vs_M0_orig=float(skill(auc(y, po), res["auc_M0_orig"])), top1=topk(y, po, 0.01)[0])
    pl = predict(X, v, FEATS, *fz["train_t"], model="logit"); pl0 = predict(X, v, BASE_SETS["M0_L2"], *fz["train_t"], model="logit")
    res["secondary_logit_champion"] = dict(auc=float(auc(y, pl)), auc_logit_M0L2=float(auc(y, pl0)), skill_vs_logit_M0L2=float(skill(auc(y, pl), auc(y, pl0))))
    res["pilot_design_model"] = dict(L2_auc=res["auc_pilot_L2"], L2_skill_vs_M0L2=float(skill(res["auc_pilot_L2"], res["auc_M0_L2"])),
                                     orig_auc=res["auc_pilot_orig"], orig_skill_vs_M0_orig=float(skill(res["auc_pilot_orig"], res["auc_M0_orig"])))
    sub = {}
    for name, m in {"기업": te.firm.values == 1, "대학": te.perform_grp.astype(str).values == "대학",
                    "출연연·국공립연": te.perform_grp.astype(str).isin(["출연연", "국공립연"]).values, "신규수행기관": te.org_nproj.values == 0,
                    "선행과제없음": te.has_pre.values == 0}.items():
        if y[m].sum() >= 10:
            a1, a0 = auc(y[m], p[m]), auc(y[m], refp["M0_L2"][m])
            sub[name] = dict(n=int(m.sum()), pos=int(y[m].sum()), auc=float(a1), auc_M0L2=float(a0), skill=float(skill(a1, a0)), top1=topk(y[m], p[m], 0.01)[0])
    res["subgroups"] = sub
    res["note"] = ("KR 2023-12 출원 약 10% 미공개(공개 지연)·NTIS 성과 보고 지연(2026-09 스냅샷 기준 출원 후 약 2.7~3.7년) — 레이블 과소 관측 가능. "
                   "원점 안 순위 지표라 공통 지연 영향은 작지만 수행주체별 보고 속도 차이는 편향 가능 → 하위집단 병기")
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame({"pjt_no": te.pjt_no.values, "org": g, "y": y, "p_champ": p, **{f"p_{k}": pp for k, pp in refp.items()}}).to_parquet(os.path.join(OUT, "final_preds_r6.parquet"), index=False)
    log("final", json.dumps({k: res[k] for k in ["auc", "auc_M0L2", "skill", "skill_ci95", "top1", "top5", "top10"]}, ensure_ascii=False))


def future():
    import n9_build, n9_textclf
    assert os.path.exists(os.path.join(OUT, "final_result_r6.json")), "future는 final 이후에만"
    D = n9_build.build("future", save=True).reset_index(drop=True)
    rows = []
    for v in (2023, 2024):
        n9_textclf.run("future", [v], D=D[["pjt_no", "t0", "y"]])
        X = with_txt(D, "future", v); te = X[X.t0 == v].copy()
        te["score"] = predict(X, v, FEATS, 2010, v - GAP); te["rank_pct"] = te.score.rank(pct=True)
        rows.append(te[["pjt_no", "t0", "org", "perform_grp", "top_ksic", "score", "rank_pct"]])
    S = pd.concat(rows); S.to_csv(os.path.join(OUT, "future_scores_r6.csv"), index=False, encoding="utf-8-sig")
    g = S.groupby(["t0", "top_ksic"]).agg(n_proj=("pjt_no", "size"), exp_pos=("score", "sum"), n_top10=("rank_pct", lambda s: int((s >= 0.9).sum())))
    g.reset_index().to_csv(os.path.join(OUT, "future_ksic_r6.csv"), index=False, encoding="utf-8-sig")
    log("future", len(S))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["dev", "final", "future"], required=True)
    ap.add_argument("--part", type=int, default=1); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"dev": lambda: dev(a.part), "final": lambda: final(a.force), "future": future}[a.mode]()
