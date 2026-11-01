# -*- coding: utf-8 -*-
"""N2 동결 실행기 (final_spec_r2.json, 챔피언 E37).
  --mode dev    : 개발 원점 2016..2023 재채점(2024 조사값 미로드). 동결 수치 재현 확인 → R2/dev_frozen_check.json
  --mode final  : 🔒 최종시험 t=2024 (F=2024). 학습은 결과연도 ≤2023 레이블만. 1회만 실행(결과 파일 있으면 거부).
                  DART 2024 사업보고서·Comtrade 2024 수출을 이 시점에 수집 → 예측 → 광업제조업조사 2024 로 채점.
                  모형·M1 MAE/skill, KSIC 50 군집 부트스트랩 95% CI, 업종코드 배분 대조(K2)·특허 제거(K1) → final_result.json
  --mode future : 2025 나우캐스트(학습 2014..2024). 최종시험 이후에만. → future_nowcast_2025.csv
개발 단계에서는 dev 만 실행했다(final·future 미실행)."""
import os, sys, json, subprocess, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
sys.path.insert(0, HERE)
import n_lib as L
import n04_panel

SPEC = json.load(open(os.path.join(HERE, "final_spec_r2.json"), encoding="utf-8"))
CH = L.load_state()
assert CH["exp_id"] == SPEC["champion_exp_id"], "champion.json 과 동결 명세 불일치"
spec = CH["spec"]


def sh(args, env=None):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=HERE, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise SystemExit(f"실패: {args}")


def ensure_dart(y, flag):
    fn = os.path.join(EXT, f"R2N2_dart_fnltt_{y}.jsonl")
    done = open(os.path.join(C, "n01_done.txt")).read().split() if os.path.exists(os.path.join(C, "n01_done.txt")) else []
    for _ in range(4):                      # 한 실행 480초 예산 → 미완료면 재실행(완료 청크 건너뜀)
        if sum(1 for d in done if d.startswith(f"{y}_")) >= 96:
            break
        sh(["n01_dart_collect.py", str(y), flag])
        done = open(os.path.join(C, "n01_done.txt")).read().split()
    assert os.path.exists(fn), f"DART {y} 보고서 파일 없음"


def ensure_export(y):
    fn = os.path.join(EXT, f"R2N2_kr_export_hs6_{y}.csv")
    if not os.path.exists(fn):
        sh(["n05_fetch_export.py", str(y)], env={"N2_FINAL_OK": "1"})


def boot1(P, Q, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    e1 = (P.y - P.p).abs().values; e0 = (Q.y - Q.p).abs().values; k = len(e1); out = []
    for _ in range(n):
        s = rng.integers(0, k, k)
        out.append(1 - e1[s].sum() / e0[s].sum())
    return np.percentile(out, [2.5, 50, 97.5]).round(4).tolist(), float(np.mean(np.array(out) > 0))


def score1(P, Q):
    m, b = float(np.mean(np.abs(P.y - P.p))), float(np.mean(np.abs(Q.y - Q.p)))
    from scipy.stats import spearmanr
    return {"mae_model": round(m, 5), "mae_M1": round(b, 5), "skill": round(1 - m / b, 4),
            "rho": round(float(spearmanr(P.y, P.p).statistic), 3),
            "dir": round(float(np.mean(np.sign(P.y - P.y.median()) == np.sign(P.p - P.p.median()))), 3),
            "medae": round(float(np.median(np.abs(P.y - P.p))), 5)}


def mode_dev():
    base = L.baseline(); P = L.evaluate(spec); m = L.metrics(P, base)
    ci, pp = L.boot_ci(P, base)
    out = {"mode": "dev", "champion": CH["exp_id"], "mean_skill": round(m["mean_skill"], 4), "mean_mae": round(m["mean_mae"], 5),
           "boot_ci": ci.round(4).tolist(), "P_gt0": pp, "rho": round(m["rho"], 3), "dir": round(m["dir"], 3), "medae": round(m["medae"], 5),
           "per_origin": {t: {"mae": round(m["mae"][t], 4), "mae_M1": round(L.mae(base)[t], 4), "skill": round(m["skill"][t], 4)} for t in L.DEV},
           "matches_frozen": abs(m["mean_skill"] - SPEC["dev"]["mean_skill"]) < 1e-3}
    json.dump(out, open(os.path.join(L.R2, "dev_frozen_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def mode_final(force=False):
    outf = os.path.join(HERE, "final_result.json")
    if os.path.exists(outf) and not force:
        raise SystemExit("최종시험은 1회만: final_result.json 존재")
    ensure_dart(2024, "--final"); ensure_export(2024)
    sh(["n02_fin.py", "2024"])
    L._cache["fin_rt2.parquet"] = pd.read_parquet(os.path.join(C, "fin_rt2_2024.parquet"))
    L._cache["fin_rt.parquet"] = pd.read_parquet(os.path.join(C, "fin_rt_2024.parquet"))
    Y = n04_panel.build(maxy=2024)          # 🔒 여기서 처음으로 2024 조사값 로드
    L.PANEL_OVERRIDE = Y
    tr_max = Y[Y.t <= 2023].t.max(); assert tr_max == 2023
    Yb = L.build(spec)
    P = L.run(Yb, spec, origins=[2024])[2024]
    Q = L.run(L.panel(), L.BASE_SPEC, origins=[2024])[2024]
    P, Q = P[P.y.notna()], Q[Q.y.notna()]
    res = {"mode": "final", "t": 2024, "champion": CH["exp_id"], "n_ksic": int(len(P)), **score1(P, Q)}
    res["boot_ci_skill"], res["P_skill_gt0"] = boot1(P, Q)
    ctl = {}
    for k, upd in [("K2_induty(E11식 업종코드 배분)", {"induty": True, "feats": [("gd_induty_f" if x == "gd_f" else x) for x in spec["feats"]]}),
                   ("K1_nopatent(특허 브리지 제거)", {"feats": [x for x in spec["feats"] if x != "gd_f"]})]:
        s2 = {**spec, **upd}; P2 = L.run(L.build(s2), s2, origins=[2024])[2024]; P2 = P2[P2.y.notna()]
        ctl[k] = {**score1(P2, Q), "boot_ci_skill": boot1(P2, Q)[0], "champion_minus_control_ci": boot1(P, P2)[0]}
    res["controls"] = ctl
    pd.concat([P.assign(model="champion"), Q.assign(model="M1")]).to_csv(os.path.join(HERE, "final_predictions_2024.csv"), index=False, encoding="utf-8-sig")
    json.dump(res, open(outf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def mode_future():
    assert os.path.exists(os.path.join(HERE, "final_result.json")), "미래 예측은 최종시험 이후에만"
    ensure_dart(2025, "--future")
    sh(["n02_fin.py", "2025"]); sh(["n03_alloc_pit.py", "2024"])
    fn = os.path.join(EXT, "R2N2_kr_export_hs6_2025.csv")
    if not os.path.exists(fn):
        sh(["n05_fetch_export.py", "2025"], env={"N2_FINAL_OK": "1"})   # 2024 는 최종시험 때 이미 수집(R2N2_..._2024.csv) → 2025 만
    L._cache["fin_rt2.parquet"] = pd.read_parquet(os.path.join(C, "fin_rt2_2025.parquet"))
    L._cache["alloc_pit.parquet"] = pd.read_parquet(os.path.join(C, "alloc_pit_2024.parquet"))
    Y = n04_panel.build(maxy=2024, feature_years=[2025], idx_maxy=2025)
    L.PANEL_OVERRIDE = Y
    Yb = L.build(spec)
    P = L.run(Yb, spec, origins=[2025])[2025]
    Q = L.run(L.panel(), L.BASE_SPEC, origins=[2025])[2025]
    out = P[["ksic", "p"]].rename(columns={"p": "nowcast_g_2025"}).assign(M1_g_2025=Q.p.values)
    out.to_csv(os.path.join(HERE, "future_nowcast_2025.csv"), index=False, encoding="utf-8-sig")
    print(out.describe())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["dev", "final", "future"], required=True)
    ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    {"dev": mode_dev, "final": lambda: mode_final(a.force), "future": mode_future}[a.mode]()
