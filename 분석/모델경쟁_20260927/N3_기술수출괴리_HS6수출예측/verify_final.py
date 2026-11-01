# -*- coding: utf-8 -*-
"""N3 최종시험 독립 재계산 (final_run_r2.py 실행 후에만).
1) 레이블을 원자료(R2global + R2N3 2022–24 / hs6_import_yearly)에서 n3_build 없이 직접 계산해 final_predictions_*.csv 의 y·표본과 대조
2) 2022–24 수집값을 다른 트랙이 따로 수집한 N1final 파일과 대조(읽기만)
3) 예측 파일로 MAE·skill 재계산(시드 평균 예측 기준)
4) 캐시된 개발 패널(panel_*.parquet, t0≤2018)로 학습해 XGB를 직접 다시 적합 → 최종 MAE 재현 확인
5) 사후 진단(모형 변경 아님): 괴리 제거·특허+괴리 제거판의 최종시험 MAE, 조기경보 하위 20% precision
출력: verify_final.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from common_n3 import HERE, EXT, FUS, CACHE, load_map
from n3_build import build
from n3_lib import derive

SPEC = json.load(open(os.path.join(HERE, "final_spec_r2.json"), encoding="utf-8"))
P = dict(SPEC["G2"]["params"])
out = {}
m = load_map(); hs_map = set(m.hs6)

# ---------- 1) 독립 레이블
a = pd.read_csv(os.path.join(EXT, "R2global_kr_export_hs6.csv"), encoding="utf-8-sig", dtype={"hs6": str})
a = a[a.year <= 2021]; a["hs6"] = a.hs6.str.zfill(6)
b = pd.read_csv(os.path.join(EXT, "R2N3_kr_export_hs6_2022_2024.csv"), dtype={"hs6": str}); b["hs6"] = b.hs6.str.zfill(6)
ex = pd.concat([a[["year", "hs6", "value"]], b[["year", "hs6", "value"]]]).pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")
im = pd.read_csv(os.path.join(FUS, "hs6_import_yearly.csv"), encoding="utf-8-sig", dtype={"hs6": str}); im["hs6"] = im.hs6.str.zfill(6)
im = im[im.partner == "WLD"].pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")


def lab(W):
    yrs = list(range(2016, 2025)); s = W.reindex(columns=yrs)
    s = s[(s > 0).all(axis=1) & s.index.isin(hs_map)]
    yr = np.log(s[[2022, 2023, 2024]].sum(1) / s[[2019, 2020, 2021]].sum(1))
    return (yr - yr.median()).rename("y_ind")


for g, flow, W in [("G2", "X", ex), ("G1", "M", im)]:
    pr = pd.read_csv(os.path.join(HERE, f"final_predictions_{flow}.csv"), dtype={"hs6": str})
    L = lab(W)
    q = pr.merge(L, left_on="hs6", right_index=True, how="outer", indicator=True)
    both = q[q._merge == "both"]
    y = pr.y.values
    r = {"n_pred_file": int(len(pr)), "n_independent": int(len(L)), "n_match": int(len(both)),
         "max_abs_y_diff": float((both.y - both.y_ind).abs().max()),
         "mae_model_meanpred": round(float(np.abs(y - pr.score_model).mean()), 4),
         "mae_B1_meanpred": round(float(np.abs(y - pr.score_B1).mean()), 4),
         "mae_B0": round(float(np.abs(y).mean()), 4)}
    r["skill_vsB1_meanpred"] = round(1 - r["mae_model_meanpred"] / r["mae_B1_meanpred"], 4)
    r["skill_vsB0_meanpred"] = round(1 - r["mae_model_meanpred"] / r["mae_B0"], 4)
    # 조기경보: 예측 하위 20% 중 실제 하위 20% 비율
    bot = pr.y <= pr.y.quantile(0.2)
    for c in ["score_model", "score_B1"]:
        w = pr[c] <= pr[c].quantile(0.2)
        r[f"precision_bottom20_{c}"] = round(float(bot[w].mean()), 4)
        r[f"mean_y_warned_{c}"] = round(float(pr.y[w].mean()), 4)
    # KSIC별 개선 비율
    e = pr.assign(eM=np.abs(y - pr.score_model), eB=np.abs(y - pr.score_B1)).groupby("k5")[["eM", "eB"]].mean()
    r["share_ksic_improved"] = round(float((e.eM < e.eB).mean()), 4); r["n_ksic"] = int(len(e))
    out[g] = r

# ---------- 2) 다른 트랙 수집본과 대조
n1 = os.path.join(EXT, "N1final_kr_export_hs6_2019_2024.csv")
if os.path.exists(n1):
    c = pd.read_csv(n1, dtype={"hs6": str}); c["hs6"] = c.hs6.astype(str).str.zfill(6)
    vcol = "value" if "value" in c else [x for x in c.columns if "val" in x.lower()][0]
    c = c[c.year >= 2022].groupby(["year", "hs6"])[vcol].sum().rename("n1")
    j = b.set_index(["year", "hs6"]).value.to_frame().join(c, how="outer")
    out["crosscheck_N1"] = {"n_both": int(j.dropna().shape[0]), "n_only_N3": int(j.n1.isna().sum()), "n_only_N1": int(j.value.isna().sum()),
                            "max_rel_diff": float((j.dropna().value / j.dropna().n1 - 1).abs().max())}

# ---------- 4) 직접 재적합 + 5) 사후 진단
for g, flow in [("G2", "X"), ("G1", "M")]:
    tr = derive(pd.read_parquet(os.path.join(CACHE, f"panel_{flow}.parquet")))
    tr = tr[tr.t0 <= 2018]
    D = derive(build(flow, 2021, trade_max=2024, allow_final=True, verbose=False))
    trF = D[D.t0 <= 2018]; te = D[D.t0 == 2021]
    feats = SPEC[g]["features"]; b1 = SPEC[g]["baseline_B1_features"]
    same = bool(len(trF) == len(tr) and np.allclose(trF[feats].values.astype(float), tr[feats].values.astype(float), equal_nan=True)
                and np.allclose(trF.y.values, tr.y.values))
    gap = [f for f in feats if f.startswith("gap_")]
    pat = [f for f in feats if f.startswith(("pl_", "p_"))]
    sets = {"frozen": feats, "B1": b1, "minus_gap": [f for f in feats if f not in gap], "minus_pat_gap": [f for f in feats if f not in gap + pat]}
    res = {"train_rows_identical_to_dev_cache": same, "n_train": int(len(tr)), "n_test": int(len(te))}
    y = te.y.values
    for k, fs in sets.items():
        maes = []
        for s in SPEC["seeds"]:
            mdl = xgb.XGBRegressor(random_state=s, **P); mdl.fit(tr[fs], tr.y.values)
            maes.append(np.abs(y - mdl.predict(te[fs])).mean())
        res["mae_" + k] = round(float(np.mean(maes)), 4)
    for k in ["frozen", "minus_gap", "minus_pat_gap"]:
        res["skill_vsB1_" + k] = round(1 - res["mae_" + k] / res["mae_B1"], 4)
    out[g]["refit"] = res
    print(g, json.dumps(out[g], ensure_ascii=False), flush=True)

json.dump(out, open(os.path.join(HERE, "verify_final.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out.get("crosscheck_N1"), ensure_ascii=False))
