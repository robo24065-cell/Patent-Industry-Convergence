# -*- coding: utf-8 -*-
"""N5 동결 실행기 (final_spec_r4.json).
  --mode dev     : 패널을 n5lib.build_panel로 다시 만들고 개발 원점 2014–2020 전진 검증 → final_dev_check_r4.json (E표 재현 확인)
  --mode final   : 🔒 t0=2022(반영 2023–24, F=2023) 1회 채점. 학습 t0=2012–2020(레이블 반영 ≤2022). --confirm-final 필요.
                   모형·기준선 AUC, skill, 출원인 군집 부트스트랩 95% CI → final_result_r4.json
  --mode future  : t0=2024 단위 점수(레이블 없음). 최종시험 채점 후에만(학습에 t0=2021·2022 추가). → future_scores_r4.parquet
"""
import os, sys, json, time, argparse, hashlib
import numpy as np, pandas as pd
from n5lib import *

SPEC = json.load(open(os.path.join(OUT, "final_spec_r4.json"), encoding="utf-8"))
FE = SPEC["champion_features"]; F0 = SPEC["baseline_M0"]; F1 = FEATS["M0"] + FEATS["M1"]
SEEDS = tuple(SPEC["seeds"])
t = time.time()


def check_lib():
    h = hashlib.sha256(open(os.path.join(OUT, "n5lib.py"), "rb").read()).hexdigest()
    if h != SPEC["code"]["lib_sha256"]:
        print("경고: n5lib.py 해시가 동결 시점과 다르다", h[:12], SPEC["code"]["lib_sha256"][:12], flush=True)


def panels(t0s, mode):
    k = load_kr(KS_MAIN); ex = load_family(k, KS_MAIN); E = load_ext(KS_MAIN)
    out = []
    for t0 in t0s:
        out.append(build_panel(t0, KS_MAIN, mode, k, ex, E)); tlog(t, "panel", t0, mode, len(out[-1]))
    return pd.concat(out, ignore_index=True)


def fp(tr, te, feats, model="xgb"):
    return fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), model=model, seeds=SEEDS if model == "xgb" else (0,))


def topk(y, p, k):
    return int(np.asarray(y)[np.argsort(-np.asarray(p))[:k]].sum())


def mode_dev():
    D = panels(range(2012, 2021), "dev")
    res = {}
    for v in SPEC["dev"]["eval_origins"]:
        tr = D[(D.t0 <= v - 2)]; te = D[D.t0 == v]
        a1 = auc(te.y, fp(tr, te, FE)); a0 = auc(te.y, fp(tr, te, F0))
        e1 = SPEC["dev"]["auc_champ"][str(v)]; e0 = SPEC["dev"]["auc_M0"][str(v)]
        res[v] = dict(auc_champ=a1, auc_M0=a0, skill=skill(a1, a0), exp_auc_champ=e1, exp_auc_M0=e0,
                      diff_champ=round(a1 - e1, 6), diff_M0=round(a0 - e0, 6))
        tlog(t, v, round(a1, 5), round(a0, 5), "diff", res[v]["diff_champ"], res[v]["diff_M0"])
    ms = float(np.mean([r["skill"] for r in res.values()]))
    ok = all(abs(r["diff_champ"]) < 1e-4 and abs(r["diff_M0"]) < 1e-4 for r in res.values())
    json.dump(dict(mode="dev", mean_skill_vs_M0=ms, reproduced=ok, by_origin=res), open(os.path.join(OUT, "final_dev_check_r4.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("mean skill", round(ms, 4), "reproduced", ok)


def mode_final():
    assert not os.path.exists(os.path.join(OUT, "final_result_r4.json")), "최종시험은 1회만"
    tr = panels(range(2012, 2021), "dev")          # 레이블 반영 ≤ 2022 < F
    assert tr.t0.max() + 2 < F_YEAR
    te = panels([T0_FINAL], "final")
    P = {"champ": fp(tr, te, FE), "M0": fp(tr, te, F0), "M1_pilot": fp(tr, te, F1),
         "champ_logit": fp(tr, te, FE, "logit"), "M0_logit": fp(tr, te, F0, "logit")}
    y = te.y.values
    A = {k: float(auc(y, p)) for k, p in P.items()}
    (lo, hi), bm = cluster_boot_ci(y, P["champ"], P["M0"], te.app.values, n=1000, seed=2022)
    (lo1, hi1), _ = cluster_boot_ci(y, P["M1_pilot"], P["M0"], te.app.values, n=1000, seed=2022)
    (lol, hil), _ = cluster_boot_ci(y, P["champ_logit"], P["M0_logit"], te.app.values, n=1000, seed=2022)
    k1 = max(1, int(0.01 * len(te)))
    R = dict(mode="final", t0=T0_FINAL, outcome="ETSI 반영 2023-01-01~2024-12-31", n=int(len(te)), pos=int(y.sum()), base_rate=float(y.mean()),
             auc=A, skill_champ_vs_M0=skill(A["champ"], A["M0"]), ci95_cluster_app=[float(lo), float(hi)], boot_mean=bm,
             skill_M1pilot_vs_M0=skill(A["M1_pilot"], A["M0"]), ci95_M1pilot=[float(lo1), float(hi1)],
             skill_logit_vs_M0logit=skill(A["champ_logit"], A["M0_logit"]), ci95_logit=[float(lol), float(hil)],
             top100_hits={k: topk(y, P[k], 100) for k in ["champ", "M0", "M1_pilot"]},
             top1pct_prec={k: topk(y, P[k], k1) / k1 for k in ["champ", "M0", "M1_pilot"]},
             spec=SPEC["champion_exp"], train_origins=sorted(tr.t0.unique().tolist()))
    pd.DataFrame({"appl_no": te.appl_no, "app": te.app, "y": y, **{f"p_{k}": v for k, v in P.items()}}).to_parquet(os.path.join(OUT, "final_preds_r4.parquet"), index=False)
    json.dump(R, open(os.path.join(OUT, "final_result_r4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


def mode_future():
    assert os.path.exists(os.path.join(OUT, "final_result_r4.json")), "최종시험 채점 후에만"
    tr = pd.concat([panels(range(2012, 2021), "dev"), panels([2021, 2022], "postfinal")], ignore_index=True)
    te = panels([T0_FUTURE], "nolabel")
    te["score"] = fp(tr, te, FE)
    te[["appl_no", "app", "mg", "ay", "score"]].sort_values("score", ascending=False).to_parquet(os.path.join(OUT, "future_scores_r4.parquet"), index=False)
    print("future scored", len(te))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "final", "future"], required=True)
    ap.add_argument("--confirm-final", action="store_true")
    a = ap.parse_args()
    check_lib()
    if a.mode == "dev":
        mode_dev()
    elif a.mode == "final":
        assert a.confirm_final, "🔒 최종시험은 --confirm-final로만 실행"
        mode_final()
    else:
        mode_future()
