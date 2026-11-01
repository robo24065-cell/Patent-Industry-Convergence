# -*- coding: utf-8 -*-
"""N8 (J2) 특허 분쟁 표적 레이더 — 동결 실행기 (final_spec_r6.json).
  --mode dev --part 1 [--origins 2012,2016,2020] : 봉인 추출본(등록원부 사건 < 2023-01-01)에서 지정 원점 패널을 다시 만들어
                                                  cache/panel_dev.parquet 와 열 단위로 대조 → cache/devcheck_part1.json
  --mode dev --part 2 : dev 원점 2012–2020에서 동결 챔피언(XGB M0+M1)·M0·강한 기준선 SB·로지스틱 챔피언·로지스틱 파일럿 설계를 재채점,
                        experiments.csv 값과 대조, AUC·skill·상위 k·권리자 군집 부트스트랩(200회) → final_dev_check_r6.json
  --mode final  : 🔒 한 번만(이미 final_result_r6.json이 있으면 거부). 학습 t0 2008–2020(결과창 끝 ≤ 2022-12-31 < F=2023),
                  시험 t0=2022 → 무효·권리범위 심판 예고등록 2023-01-01~2024-12-31.
                  AUC·skill(M0 대비, 주)·강한 기준선 SB 대비 skill·상위 1/5/10%·상위 100건·권리자 군집 부트스트랩 95% CI(1,000회),
                  보조: 로지스틱 챔피언·로지스틱 파일럿 설계, 사전 등록 하위집단(신규/후속 분쟁) → final_result_r6.json, final_preds_r6.parquet
  --mode future : final 이후에만. t0=2024 살아 있는 권리 점수(학습 t0 2008–2022, 결과 ≤ 2024-12-31) → future_scores_r6.parquet, future_ksic_r6.csv
6라운드에서는 --mode dev만 실행했다. final·future는 실행하지 않았다."""
import sys, os, json, argparse, hashlib
sys.dont_write_bytecode = True
from n8lib import *

SPEC = json.load(open(os.path.join(HERE, "final_spec_r6.json"), encoding="utf-8"))
CH = SPEC["champion"]
REFS = SPEC["references"]
NEG_FRAC, NEG_SEED = CH["neg_frac"], CH["neg_seed"]


def check_code():
    return [f for f, h in SPEC["code_sha256"].items()
            if hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() != h]


def all_feats():
    s = list(CH["feats"])
    for r in REFS.values():
        s += r["feats"]
    return list(dict.fromkeys(s))


def fit_score(TR, TE, feats, model):
    """음성 5% 표본(시드 0, 가중 1/0.05)으로 학습, TE 전체를 채점."""
    rng = np.random.default_rng(NEG_SEED)
    y = TR.y.values.astype(int)
    keep = (y == 1) | (rng.random(len(TR)) < NEG_FRAC)
    w = np.where(y == 1, 1.0, 1.0 / NEG_FRAC)[keep]
    return fit_pred(TR.loc[keep, feats].astype(float), y[keep], TE[feats].astype(float), model, w=w)


def metrics(y, p, prefix=""):
    y = np.asarray(y); n = len(y); npos = int(y.sum())
    r = {f"{prefix}auc": float(auc(y, p))}
    for nm, fr in (("1", 0.01), ("5", 0.05), ("10", 0.10)):
        prec, hit, k = topk(y, p, fr)
        r[f"{prefix}top{nm}pct_prec"] = prec; r[f"{prefix}top{nm}pct_hits"] = hit; r[f"{prefix}top{nm}pct_capture"] = hit / max(npos, 1); r[f"k_top{nm}pct"] = k
    r[f"{prefix}hit100"] = topk(y, p, k=100)[1]
    return r


def evaluate(te, preds, n_boot, seed):
    """preds: {'champ':..., 'M0':..., 'SB':..., 'logit_champ':..., 'logit_pilot':...}"""
    y = te.y.values.astype(int); g = cluster_key(te)
    out = dict(n=int(len(y)), pos=int(y.sum()), base_rate_permil=float(1000 * y.mean()))
    for k, p in preds.items():
        out.update(metrics(y, p, prefix=f"{k}_"))
    a = out["champ_auc"]
    for ref in ("M0", "SB"):
        out[f"skill_vs_{ref}"] = float(skill(a, out[f"{ref}_auc"]))
        ci, cia, cib = boot_ci(y, preds["champ"], preds[ref], g, n=n_boot, seed=seed + (0 if ref == "M0" else 1))
        out[f"skill_vs_{ref}_ci"] = ci; out[f"{ref}_auc_ci"] = cib
        if ref == "M0":
            out["champ_auc_ci"] = cia
    for k in ("logit_champ", "logit_pilot"):
        if k in preds:
            out[f"{k}_skill_vs_M0"] = float(skill(out[f"{k}_auc"], out["M0_auc"]))
            out[f"{k}_skill_vs_M0_ci"] = boot_ci(y, preds[k], preds["M0"], g, n=n_boot, seed=seed + 2)[0]
    # 사전 등록 하위집단: 신규 분쟁(자기 과거 분쟁·기타 심판 없음) / 후속 분쟁(이력 있음)
    fresh = (te.own_disp.values == 0) & (te.own_trial.values == 0)
    sub = {}
    for nm, s in (("신규(이력 없음)", fresh), ("후속(이력 있음)", ~fresh)):
        ys = y[s]; d = dict(n=int(s.sum()), pos=int(ys.sum()))
        if 3 <= ys.sum() < len(ys):
            for k, p in preds.items():
                d[f"{k}_auc"] = float(auc(ys, p[s]))
            d["skill_vs_M0"] = float(skill(d["champ_auc"], d["M0_auc"])); d["skill_vs_SB"] = float(skill(d["champ_auc"], d["SB_auc"]))
            d["skill_vs_M0_ci"] = boot_ci(ys, preds["champ"][s], preds["M0"][s], g[s], n=n_boot, seed=seed + 3)[0]
            d["skill_vs_SB_ci"] = boot_ci(ys, preds["champ"][s], preds["SB"][s], g[s], n=n_boot, seed=seed + 4)[0]
            d["champ_top1pct_capture"] = topk(ys, preds["champ"][s], 0.01)[1] / ys.sum()
            d["M0_top1pct_capture"] = topk(ys, preds["M0"][s], 0.01)[1] / ys.sum()
        sub[nm] = d
    out["subgroups"] = sub
    return out


def predict_all(TR, TE):
    P = {"champ": fit_score(TR, TE, CH["feats"], CH["model"])}
    for k, r in REFS.items():
        P[k] = fit_score(TR, TE, r["feats"], r["model"])
        log("fitted", k)
    return P


def load_dev_panel():
    cols = list(dict.fromkeys(["rgst_no", "t0", "y", "ksic", "owner_v2", "own_disp", "own_trial"] + all_feats()))
    return pd.read_parquet(f"{CACHE}/panel_dev.parquet", columns=cols).reset_index(drop=True)


# ------------------------------------------------------------------ dev
def dev_part1(origins):
    R = extract(SEAL)                       # 봉인 추출본(사건 < 2023-01-01). assert는 extract 안에 있다
    X = prep(R); del R
    FT = family_tokens(X["pt"]); EXTU = external_unit(); KI = ksic_industry()
    old = pd.read_parquet(f"{CACHE}/panel_dev.parquet")
    rep = {}
    for t0 in origins:
        P = to32(build_t0(t0, X, FT, EXTU, KI)).reset_index(drop=True)
        O = old[old.t0 == t0].reset_index(drop=True)
        same_rows = bool(len(P) == len(O) and (P.rgst_no.values == O.rgst_no.values).all())
        bad = []
        if same_rows:
            for c in O.columns:
                if O[c].dtype.kind in "fiu":
                    a = pd.to_numeric(P[c], errors="coerce").astype(float).values; b = O[c].astype(float).values
                    if not np.allclose(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999), atol=1e-4):
                        bad.append(c)
                elif not (P[c].astype(str).values == O[c].astype(str).values).all():
                    bad.append(c)
        rep[int(t0)] = dict(rows=int(len(P)), pos=int(P.y.sum()), same_rows=same_rows, mismatched_cols=bad)
        log("part1", t0, rep[int(t0)])
    json.dump(rep, open(f"{CACHE}/devcheck_part1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def dev_part2(n_boot=200):
    D = load_dev_panel()
    ex = pd.read_csv(os.path.join(HERE, "experiments.csv"), encoding="utf-8-sig")
    ref = {(r.exp_id, r.family): r for r in ex.itertuples()}
    out = {}
    for v in DEV:
        TR = D[(D.t0 >= TRAIN_MIN) & (D.t0 <= v - 2)]; TE = D[D.t0 == v].reset_index(drop=True)
        assert TR.t0.max() + 2 <= v
        P = predict_all(TR, TE)
        r = evaluate(TE, P, n_boot, seed=v)
        r["match"] = {
            "champ_vs_E01_xgb": abs(round(r["champ_auc"], 4) - getattr(ref[("E01", "xgb")], f"auc_{v}")) < 1e-9,
            "M0_vs_E00_xgb": abs(round(r["M0_auc"], 4) - getattr(ref[("E00", "xgb")], f"auc_{v}")) < 1e-9,
            "logit_champ_vs_E09_logit": abs(round(r["logit_champ_auc"], 4) - getattr(ref[("E09", "logit")], f"auc_{v}")) < 1e-9,
        }
        out[v] = r
        log(v, round(r["champ_auc"], 4), round(r["M0_auc"], 4), round(r["skill_vs_M0"], 4), r["match"])
    m = lambda k: float(np.mean([out[v][k] for v in DEV]))
    res = dict(mode="dev", code_changed=check_code(), origins=DEV,
               part1=json.load(open(f"{CACHE}/devcheck_part1.json", encoding="utf-8")) if os.path.exists(f"{CACHE}/devcheck_part1.json") else None,
               all_match=bool(all(all(out[v]["match"].values()) for v in DEV)),
               mean_auc_champ=m("champ_auc"), mean_auc_M0=m("M0_auc"), mean_auc_SB=m("SB_auc"),
               mean_skill_vs_M0=m("skill_vs_M0"), win_vs_M0=float(np.mean([out[v]["champ_auc"] > out[v]["M0_auc"] for v in DEV])),
               mean_skill_vs_SB=m("skill_vs_SB"), win_vs_SB=float(np.mean([out[v]["champ_auc"] > out[v]["SB_auc"] for v in DEV])),
               mean_logit_champ_auc=m("logit_champ_auc"), mean_logit_champ_skill_vs_M0=m("logit_champ_skill_vs_M0"),
               mean_logit_pilot_auc=m("logit_pilot_auc"), mean_logit_pilot_skill_vs_M0=m("logit_pilot_skill_vs_M0"),
               mean_top1pct_prec=m("champ_top1pct_prec"), mean_top1pct_capture=m("champ_top1pct_capture"), mean_top1pct_capture_M0=m("M0_top1pct_capture"),
               mean_top1pct_capture_SB=m("SB_top1pct_capture"), mean_top5pct_capture=m("champ_top5pct_capture"), mean_top10pct_capture=m("champ_top10pct_capture"),
               mean_hit100=m("champ_hit100"), mean_hit100_M0=m("M0_hit100"), mean_hit100_SB=m("SB_hit100"),
               by_origin={str(v): out[v] for v in DEV})
    json.dump(res, open(os.path.join(HERE, "final_dev_check_r6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log("dev check all_match", res["all_match"], "mean skill", round(res["mean_skill_vs_M0"], 4), "vs SB", round(res["mean_skill_vs_SB"], 4))


# ------------------------------------------------------------------ 🔒 final
def final():
    fr = os.path.join(HERE, "final_result_r6.json")
    if os.path.exists(fr):
        sys.exit("final_result_r6.json이 이미 있다 — 최종시험은 한 번만 실행한다.")
    assert not check_code(), f"동결 이후 코드가 바뀌었다: {check_code()}"
    TR = load_dev_panel()
    TR = TR[(TR.t0 >= SPEC["final"]["train_t0"][0]) & (TR.t0 <= SPEC["final"]["train_t0"][1])].reset_index(drop=True)
    assert (TR.t0 + 2).max() * 10000 + 1231 < F_YEAR * 10000 + 101, "학습 결과창이 F를 넘음"
    R = extract(SEAL); X = prep(R); del R            # 피처: 사건 < 2023-01-01
    FT = family_tokens(X["pt"]); EXTU = external_unit(); KI = ksic_industry()
    L = final_labels("20230101", "20241231")         # 🔒 이 호출이 최종 결과창을 처음 읽는다
    L["d"] = pd.to_numeric(L.d, errors="coerce")
    TE = to32(build_t0(2022, X, FT, EXTU, KI, label_disp=L, label_end=20241231)).reset_index(drop=True)
    log("final test", len(TE), int(TE.y.sum()))
    P = predict_all(TR, TE)
    r = evaluate(TE, P, n_boot=1000, seed=2022)
    r.update(mode="final", t0=2022, outcome_window=["2023-01-01", "2024-12-31"], F=F_YEAR, train_t0=SPEC["final"]["train_t0"],
             n_label_events=int(len(L)), champion=CH["id"], code_changed=check_code())
    json.dump(r, open(fr, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    pd.DataFrame({"rgst_no": TE.rgst_no.values, "ksic": TE.ksic.values, "y": TE.y.values, **{f"p_{k}": v for k, v in P.items()}}).to_parquet(
        os.path.join(HERE, "final_preds_r6.parquet"), index=False)
    log("FINAL auc", round(r["champ_auc"], 4), "M0", round(r["M0_auc"], 4), "skill", round(r["skill_vs_M0"], 4), r["skill_vs_M0_ci"],
        "vs SB", round(r["skill_vs_SB"], 4), r["skill_vs_SB_ci"])


# ------------------------------------------------------------------ future
def future():
    assert os.path.exists(os.path.join(HERE, "final_result_r6.json")), "future는 final 이후에만 실행한다."
    cut = "20250101"
    R = extract(cut); X = prep(R); del R
    FT = family_tokens(X["pt"]); EXTU = external_unit(); KI = ksic_industry()
    TR = [load_dev_panel()]
    for t0 in (2021, 2022):    # 결과창 끝 2023-12-31, 2024-12-31 (최종시험 이후라 허용)
        TR.append(to32(build_t0(t0, X, FT, EXTU, KI, label_disp=X["disp"], label_end=(t0 + 2) * 10000 + 1231))[TR[0].columns])
    TR = pd.concat(TR, ignore_index=True)
    TE = to32(build_t0(2024, X, FT, EXTU, KI, no_label=True)).reset_index(drop=True)
    TE["y"] = 0
    p = fit_score(TR, TE, CH["feats"], CH["model"])
    S = pd.DataFrame({"rgst_no": TE.rgst_no.values, "ksic": TE.ksic.values, "p": p})
    S["top1pct"] = S.p.rank(ascending=False, method="first") <= max(1, round(0.01 * len(S)))
    S.to_parquet(os.path.join(HERE, "future_scores_r6.parquet"), index=False)
    S.groupby("ksic").agg(n=("p", "size"), exp_disputes=("p", "sum"), mean_p=("p", "mean"), n_top1pct=("top1pct", "sum")).reset_index().to_csv(
        os.path.join(HERE, "future_ksic_r6.csv"), index=False, encoding="utf-8-sig")
    log("future scored", len(S))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dev", "final", "future"])
    ap.add_argument("--part", type=int, default=2)
    ap.add_argument("--origins", default="2012,2016,2020")
    a = ap.parse_args()
    if a.mode == "dev":
        dev_part1([int(x) for x in a.origins.split(",")]) if a.part == 1 else dev_part2()
    elif a.mode == "final":
        final()
    else:
        future()
