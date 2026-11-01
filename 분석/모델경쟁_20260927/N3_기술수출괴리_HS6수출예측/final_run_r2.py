# -*- coding: utf-8 -*-
"""N3 동결 실행기 — final_spec_r2.json 을 그대로 실행한다.
  python final_run_r2.py --mode dev     개발 원점(2010–2018) 재현: 동결 모형·B1·B0 지표 + KSIC 군집 CI → dev_result_r2.json
  python final_run_r2.py --mode prereg  (결과 미사용) t0=2021 피처만으로 2022–24 조기경보 점수 사전등록 → prereg_t2021_{X,M}.csv
  python final_run_r2.py --mode final   🔒 최종시험 t0=2021 → 2022–24. 학습은 t0 ≤ 2018(결과 ≤2021 < F=2022)만 → final_result.json
  python final_run_r2.py --mode future  t0=2024 → 2025–27 조기경보(최종시험 이후에만; 학습 t0 ≤ 2021)
final·future 는 개발 단계에서 실행 금지. final 은 한 번만 실행한다(재실행 시 기존 final_result.json 이 있으면 중단).
"""
import os, sys, json, time, argparse, hashlib, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common_n3 import *
from n3_build import build
from n3_lib import derive, evaluate, _fit_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

SPEC = json.load(open(os.path.join(HERE, "final_spec_r2.json"), encoding="utf-8"))
SEEDS = SPEC["seeds"]


def model_spec(g):
    s = dict(SPEC[g]["spec"]); s["params"] = {k: v for k, v in SPEC[g]["params"].items() if k not in ("nthread",)}
    s["params"]["nthread"] = 2
    return s


def boot_ci(df, a, b, n=1000, seed=0):
    agg = df.groupby("k5")[[a, b]].sum(); ks = agg.index.values; rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        s = agg.loc[rng.choice(ks, len(ks))].sum(); bs.append(1 - s[a] / s[b])
    s = agg.sum()
    return {"pooled": round(float(1 - s[a] / s[b]), 4), "ci95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]}


def fit_multi(tr, te, feats, spec):
    return np.array([_fit_predict(tr, te, feats, spec, s) for s in SEEDS])   # (seeds, n)


def score(te, PM, PB):
    y = te.y.values; top = (te.y >= te.y.quantile(0.8)).astype(int)
    mM = np.mean([np.abs(y - p).mean() for p in PM]); mB = np.mean([np.abs(y - p).mean() for p in PB]); m0 = np.abs(y).mean()
    out = {"n": int(len(te)), "mae_model": round(float(mM), 4), "mae_B1": round(float(mB), 4), "mae_B0": round(float(m0), 4),
           "skill_vsB1": round(float(1 - mM / mB), 4), "skill_vsB0": round(float(1 - mM / m0), 4), "B1_skill_vsB0": round(float(1 - mB / m0), 4),
           "rho_model": round(float(np.mean([spearmanr(p, y).correlation for p in PM])), 4),
           "rho_B1": round(float(np.mean([spearmanr(p, y).correlation for p in PB])), 4),
           "auc_top20_model": round(float(np.mean([roc_auc_score(top, p) for p in PM])), 4),
           "auc_top20_B1": round(float(np.mean([roc_auc_score(top, p) for p in PB])), 4)}
    E = te[["k5"]].copy()
    E["eM"] = np.mean([np.abs(y - p) for p in PM], axis=0); E["eB1"] = np.mean([np.abs(y - p) for p in PB], axis=0); E["eB0"] = np.abs(y)
    out["boot_vsB1"] = boot_ci(E, "eM", "eB1"); out["boot_vsB0"] = boot_ci(E, "eM", "eB0")
    return out


# ------------------------------------------------------------------ dev
def run_dev():
    res = {"mode": "dev", "run_at": time.strftime("%Y-%m-%d %H:%M")}
    for g, flow in [("G2", "X"), ("G1", "M")]:
        D = derive(build(flow, 2018, verbose=False))
        feats = SPEC[g]["features"]; b1 = SPEC[g]["baseline_B1_features"]; sp = model_spec(g)
        per, PP = [], []
        for v in SPEC["dev"]["eval_origins"]:
            tr, te = D[D.t0 <= v - 3], D[D.t0 == v]
            PM, PB = fit_multi(tr, te, feats, sp), fit_multi(tr, te, b1, {})
            s = score(te, PM, PB); s.pop("boot_vsB1"); s.pop("boot_vsB0"); s["v"] = v; per.append(s)
            e = te[["k5"]].copy(); y = te.y.values
            e["eM"] = np.mean([np.abs(y - p) for p in PM], axis=0); e["eB1"] = np.mean([np.abs(y - p) for p in PB], axis=0); e["eB0"] = np.abs(y)
            PP.append(e)
        P = pd.DataFrame(per); E = pd.concat(PP)
        ref = np.array(SPEC[g]["dev_reference_mae"])
        res[g] = {"mean_skill_vsB1": round(float(P.skill_vsB1.mean()), 4), "mean_skill_vsB0": round(float(P.skill_vsB0.mean()), 4),
                  "B1_mean_skill_vsB0": round(float(P.B1_skill_vsB0.mean()), 4),
                  "win_vsB1": round(float((P.mae_model < P.mae_B1).mean()), 2), "rho": round(float(P.rho_model.mean()), 4),
                  "rho_B1": round(float(P.rho_B1.mean()), 4), "auc_top20": round(float(P.auc_top20_model.mean()), 4),
                  "auc_top20_B1": round(float(P.auc_top20_B1.mean()), 4),
                  "boot_pooled_vsB1": boot_ci(E, "eM", "eB1"), "boot_pooled_vsB0": boot_ci(E, "eM", "eB0"),
                  "reproduces_frozen_mae": bool(np.allclose(P.mae_model.values, ref, atol=5e-4)),
                  "per_origin": P.to_dict("records")}
        print(g, json.dumps({k: v for k, v in res[g].items() if k != "per_origin"}, ensure_ascii=False), flush=True)
    json.dump(res, open(os.path.join(HERE, "dev_result_r2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ prereg (결과 미사용)
def run_prereg():
    out = {}
    for g, flow in [("G2", "X"), ("G1", "M")]:
        Dtr = derive(build(flow, 2018, verbose=False))
        Dp = derive(build(flow, 2021, t0_min=2021, trade_max=2021, require_future=False, verbose=False))
        assert Dp.y.isna().all()
        feats = SPEC[g]["features"]; sp = model_spec(g)
        tr = Dtr[Dtr.t0 <= 2018]
        P = fit_multi(tr, Dp, feats, sp).mean(axis=0)
        B = fit_multi(tr, Dp, SPEC[g]["baseline_B1_features"], {}).mean(axis=0)
        o = Dp[["hs6", "k5", "t0", "lvl", "mom3"]].copy(); o["score_model"] = P; o["score_B1"] = B
        o["risk_rank"] = o.score_model.rank(method="first").astype(int)   # 1 = 가장 낮은 예측 상대성장(둔화 위험)
        o["warn_bottom20"] = o.score_model <= o.score_model.quantile(0.2)
        f = os.path.join(HERE, f"prereg_t2021_{flow}.csv"); o.sort_values("risk_rank").to_csv(f, index=False, encoding="utf-8-sig")
        out[g] = {"file": f, "n": int(len(o)), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(), "created": time.strftime("%Y-%m-%d %H:%M")}
        print(g, out[g], flush=True)
    json.dump(out, open(os.path.join(HERE, "prereg_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ final
def fetch_kr_export_final():
    """🔒 최종시험 때 1회: 한국(410) HS6 대세계 수출 2022–24. H5(HS2017 변환) 먼저, 실패 시 HS(보고분류)."""
    if os.path.exists(KR_EXPORT_FINAL):
        return json.load(open(KR_EXPORT_FINAL + ".meta.json", encoding="utf-8"))
    codes = sorted(load_map().hs6.unique()); chunks = [codes[i:i + 250] for i in range(0, len(codes), 250)]
    rows = []; used = None; maxn = 0
    for cl in ["H5", "HS"]:
        rows = []; ok = True
        for y in [2022, 2023, 2024]:
            for cs in chunks:
                u = f"https://comtradeapi.un.org/public/v1/preview/C/A/{cl}?reporterCode=410&flowCode=X&partnerCode=0&period={y}&cmdCode={','.join(cs)}"
                d = None
                for a in range(4):
                    try:
                        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180) as r:
                            j = json.loads(r.read().decode("utf-8"))
                        if j.get("error"):
                            raise RuntimeError(j["error"])
                        d = j.get("data", []); break
                    except urllib.error.HTTPError as e:
                        if e.code == 403:
                            raise SystemExit("403 Quota — 중단, 다음 날 재실행")
                        time.sleep(10 * 2 ** a)
                    except Exception:
                        time.sleep(10 * 2 ** a)
                time.sleep(3)
                if not d:
                    ok = False; break
                maxn = max(maxn, len(d))   # 감사 추가: 미리보기 500행 상한 절단 여부 기록
                for x in d:
                    if x.get("motCode", 0) != 0 or x.get("customsCode") != "C00" or x.get("partner2Code", 0) != 0:
                        continue
                    rows.append((y, "X", str(x["cmdCode"]).zfill(6), x["primaryValue"], x.get("classificationCode")))
            if not ok:
                break
        if ok and rows:
            used = cl; break
    if used is None:
        raise SystemExit("2022–24 수출 수집 실패")
    pd.DataFrame(rows, columns=["year", "flow", "hs6", "value", "cls"]).to_csv(KR_EXPORT_FINAL, index=False)
    meta = {"classification_requested": used, "collected_at": time.strftime("%Y-%m-%d %H:%M"), "rows": len(rows), "max_rows_per_call": maxn}
    json.dump(meta, open(KR_EXPORT_FINAL + ".meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    return meta


def run_final():
    outp = os.path.join(HERE, "final_result.json")
    if os.path.exists(outp):
        raise SystemExit("final_result.json 이 이미 있다 — 최종시험은 한 번만.")
    res = {"mode": "final", "F": 2022, "origin": 2021, "run_at": time.strftime("%Y-%m-%d %H:%M")}
    res["kr_export_fetch"] = fetch_kr_export_final()
    for g, flow in [("G2", "X"), ("G1", "M")]:
        D = derive(build(flow, 2021, trade_max=2024, allow_final=True, verbose=False))
        tr, te = D[D.t0 <= 2018], D[D.t0 == 2021]
        assert tr.t0.max() + 3 < 2022
        PM, PB = fit_multi(tr, te, SPEC[g]["features"], model_spec(g)), fit_multi(tr, te, SPEC[g]["baseline_B1_features"], {})
        res[g] = score(te, PM, PB)
        o = te[["hs6", "k5", "y"]].copy(); o["score_model"] = PM.mean(0); o["score_B1"] = PB.mean(0)
        o.to_csv(os.path.join(HERE, f"final_predictions_{flow}.csv"), index=False, encoding="utf-8-sig")
        pr = os.path.join(HERE, f"prereg_t2021_{flow}.csv")
        if os.path.exists(pr):   # 사전등록 점수로 본 성과(참고)
            q = pd.read_csv(pr, dtype={"hs6": str}).merge(te[["hs6", "y"]], on="hs6")
            res[g]["prereg_overlap_n"] = int(len(q)); res[g]["prereg_rho"] = round(float(spearmanr(q.score_model, q.y).correlation), 4)
        print(g, json.dumps(res[g], ensure_ascii=False), flush=True)
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ future
def build_pat_panel_2024():
    import duckdb
    f = os.path.join(CACHE, "pat_panel_2024.parquet")
    if os.path.exists(f):
        return
    META = os.path.join(HERE, "..", "_shared", "meta", "*.parquet").replace("\\", "/")
    con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
    q = open(os.path.join(HERE, "build_panel.py"), encoding="utf-8").read()
    q = q[q.index('q = f"""') + 8:q.index('"""\ndf = con.sql')].replace("dy <= 2023 and ay <= 2023", "dy <= 2024 and ay <= 2024").replace("{META}", META)
    con.sql(q).df().to_parquet(f, index=False)


def run_future():
    if not os.path.exists(os.path.join(HERE, "final_result.json")):
        raise SystemExit("future 는 최종시험(final) 이후에만")
    build_pat_panel_2024()
    for g, flow in [("G2", "X"), ("G1", "M")]:
        Dtr = derive(build(flow, 2021, trade_max=2024, allow_final=True, verbose=False, pat_file="pat_panel_2024.parquet", kosis_max=2024))
        Dp = derive(build(flow, 2024, t0_min=2024, trade_max=2024, allow_final=True, require_future=False, verbose=False,
                          pat_file="pat_panel_2024.parquet", kosis_max=2024))
        P = fit_multi(Dtr[Dtr.t0 <= 2021], Dp, SPEC[g]["features"], model_spec(g)).mean(0)
        o = Dp[["hs6", "k5"]].copy(); o["score_model"] = P; o["risk_rank"] = o.score_model.rank(method="first").astype(int)
        o.sort_values("risk_rank").to_csv(os.path.join(HERE, f"future_t2024_{flow}.csv"), index=False, encoding="utf-8-sig")
        print(g, len(o), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["dev", "prereg", "final", "future"])
    a = ap.parse_args()
    {"dev": run_dev, "prereg": run_prereg, "final": run_final, "future": run_future}[a.mode]()
