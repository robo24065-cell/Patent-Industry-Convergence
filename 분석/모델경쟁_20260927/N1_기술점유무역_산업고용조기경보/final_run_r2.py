# -*- coding: utf-8 -*-
"""N1 동결 실행 스크립트 (final_spec_r2.json).
  --mode dev    : 개발 원점 전진 검증 재현(v=2011..2018, 학습 t0 ≤ v−3) + 기준선·참고모형·특허 증분·군집 부트스트랩 CI → R2/final_dev_result.json
  --mode final  : 🔒 최종시험 1회. 학습 t0 ≤ 2018(결과 ≤2021 < F=2022) → t0=2021 예측, 11차 종사자 2021→2024 순위와 비교 → final_result.json
                  (이미 final_result.json 이 있으면 거부 — 1회성). 모듈 D는 2022–24 수출 파일이 있을 때만 평가.
  --mode future : 최종시험 이후 전망(t0=2024 → 2027). N1_MODE=future 패널(결과 ≤2024 레이블로 학습). 최종시험 전 실행 금지.
개발 단계에서는 dev 만 실행한다."""
import argparse, json, os, sys, subprocess
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SPEC = json.load(open(os.path.join(HERE, "final_spec_r2.json"), encoding="utf-8"))
F = SPEC["features"]; ALPHA = SPEC["model"]["alpha"]
FD = SPEC["module_D"]["features"]; ALPHA_D = SPEC["module_D"]["alpha"]

def rank_within(D, cols):
    D = D.copy()
    for c in cols: D[c] = D.groupby("t0")[c].rank(pct=True).fillna(0.5)
    return D

def fit_predict(train, test, feats, tg, alpha):
    Q = rank_within(pd.concat([train, test]), feats + [tg])
    tr, te = Q.loc[train.index], Q.loc[test.index]
    tr = tr[train[tg].notna().values]
    m = Ridge(alpha=alpha).fit(tr[feats], tr[tg])
    return m.predict(te[feats]), m

def auc_top3(y, p):
    lab = (pd.Series(y).rank(pct=True) >= 2/3).astype(int)
    return roc_auc_score(lab, p)

def boot_one(ks, y, p, pb=None, n=1000, seed=0):
    """단일 원점: 산업 부트스트랩 CI(ρ, Δρ vs 비교 예측)."""
    rng = np.random.default_rng(seed); idx = np.arange(len(ks)); r, d = [], []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        a = spearmanr(y[s], p[s])[0]; r.append(a)
        if pb is not None: d.append(a - spearmanr(y[s], pb[s])[0])
    out = {"rho_ci": np.nanpercentile(r, [2.5, 97.5]).round(3).tolist()}
    if pb is not None: out["d_ci"] = np.nanpercentile(d, [2.5, 97.5]).round(3).tolist()
    return out

def ensure_panel(mode="dev"):
    name = "panel_n1.parquet" if mode == "dev" else "panel_n1_future.parquet"
    path = os.path.join(HERE, "cache", name)
    if not os.path.exists(path):
        env = dict(os.environ, N1_MODE=mode, PYTHONIOENCODING="utf-8")
        subprocess.run([sys.executable, os.path.join(HERE, "b01_panel.py")], check=True, env=env, cwd=HERE)
    return pd.read_parquet(path)

# ------------------------------------------------------------------ dev
def run_dev():
    import engine as E
    P = E.P; o = E.preds(F, alpha=ALPHA); o0 = E.b0()
    FT = [c for c in F if not E.is_patent(c)]; oT = E.preds(FT, alpha=ALPHA)
    oB1 = E.preds(SPEC["reference_models"]["B1_trade_only"]); oM0 = E.preds(SPEC["reference_models"]["M0_patent_only"])
    s, s0, sT = E.score(o), E.score(o0), E.score(oT); a, a0 = E.auc(o), E.auc(o0)
    bc = E.boot(o, o0, n=1000); bi = E.boot(o, oT, n=1000)
    res = {"mode": "dev", "champion": SPEC["champion_id"], "features": F, "alpha": ALPHA,
           "per_origin_rho": s.round(4).to_dict(), "rho_mean": round(s.mean(), 4), "rho_ci": bc["ci"], "pos_origins": int((s > 0).sum()),
           "B0_per_origin": s0.round(4).to_dict(), "B0_rho": round(s0.mean(), 4), "skill": round(s.mean() - s0.mean(), 4), "skill_ci": bc["dci"],
           "skill_win": round(((s - s0) > 0).mean(), 3),
           "auc_top3": round(a.mean(), 4), "auc_B0": round(a0.mean(), 4), "auc_skill": round((a.mean() - a0.mean())/(1 - a0.mean()), 4),
           "ref_B1_trade_only": round(E.score(oB1).mean(), 4), "ref_M0_patent_only": round(E.score(oM0).mean(), 4),
           "same_model_no_patent": round(sT.mean(), 4), "patent_increment": round((s - sT).mean(), 4), "patent_increment_win": round(((s - sT) > 0).mean(), 3),
           "patent_increment_ci": bi["dci"], "patent_increment_p_le0": round(bi["p_le0"], 3)}
    # 모듈 D
    import sD_export as D
    oD = D.pD(FD, ALPHA_D); oDB = D.oB; sD, sDB = E.score(oD), E.score(oDB); bD = E.boot(oD, oDB, n=1000)
    res["module_D"] = {"features": FD, "per_origin_rho": sD.round(4).to_dict(), "rho_mean": round(sD.mean(), 4), "rho_ci": bD["ci"],
                       "B0_rho": round(sDB.mean(), 4), "skill": round(sD.mean() - sDB.mean(), 4), "skill_ci": bD["dci"], "auc_top3": round(E.auc(oD).mean(), 4)}
    json.dump(res, open(os.path.join(HERE, "R2", "final_dev_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ final (🔒 1회)
def load_final_target():
    """🔒 최종시험 전용: 11차 종사자 2021·2024 (체인 없음). --mode final 에서만 호출."""
    from n1lib import BASE
    d = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
    d = d[(d.항목코드 == "T02") & (d.기준 == "11차") & (d.연도.isin([2021, 2024]))].copy()
    d.loc[d.마스킹 == 1, "값"] = np.nan
    w = d.pivot_table(index="세세분류코드", columns="연도", values="값"); w = w.where(w > 0)
    y = np.log(w[2024] / w[2021]); return (y - y.median()).rename("y")

def run_final():
    out = os.path.join(HERE, "final_result.json")
    if os.path.exists(out): sys.exit("final_result.json 이 이미 있다 — 최종시험은 1회만 실행한다.")
    P = ensure_panel("dev"); P = P[P.t0 != 2010]
    train = P[(P.t0 >= 2008) & (P.t0 <= 2018)].dropna(subset=["y_emp"]); test = P[P.t0 == 2021].copy()
    assert train.t0.max() + 3 <= 2021 < 2022
    y = load_final_target(); test = test.join(y, on="ksic"); test = test[test.y.notna()]
    p, m = fit_predict(train, test, F, "y_emp", ALPHA)
    FT = [c for c in F if not c.startswith(("mom_", "lvl_", "dsh_", "frgn_", "kab", "dffam"))]
    pT, _ = fit_predict(train, test, FT, "y_emp", ALPHA)
    pB0 = test.g3_emp.rank(pct=True).fillna(0.5).values
    pB1, _ = fit_predict(train, test, SPEC["reference_models"]["B1_trade_only"], "y_emp", 5.0)
    pM0, _ = fit_predict(train, test, SPEC["reference_models"]["M0_patent_only"], "y_emp", 5.0)
    yy = test.y.values; ks = test.ksic.values
    rho, rho0 = spearmanr(yy, p)[0], spearmanr(yy, pB0)[0]; a, a0 = auc_top3(yy, p), auc_top3(yy, pB0)
    bc = boot_one(ks, yy, p, pB0); bi = boot_one(ks, yy, p, pT, seed=1)
    res = {"mode": "final", "t0": 2021, "F": 2022, "n_industries": int(len(test)), "rho": round(rho, 4), "rho_ci": bc["rho_ci"],
           "B0_rho": round(rho0, 4), "d_rho_vs_B0": round(rho - rho0, 4), "d_rho_ci": bc["d_ci"],
           "auc_top3": round(a, 4), "auc_B0": round(a0, 4), "auc_skill": round((a - a0)/(1 - a0), 4),
           "ref_B1_trade_only_rho": round(spearmanr(yy, pB1)[0], 4), "ref_M0_patent_only_rho": round(spearmanr(yy, pM0)[0], 4),
           "same_model_no_patent_rho": round(spearmanr(yy, pT)[0], 4), "patent_increment": round(rho - spearmanr(yy, pT)[0], 4), "patent_increment_ci": bi["d_ci"],
           "coef": dict(zip(F, m.coef_.round(4).tolist()))}
    from n1lib import ksic_names
    R = pd.DataFrame({"ksic": ks, "name": pd.Series(ks).map(ksic_names()).values, "pred": p, "rank_pred": pd.Series(p).rank(ascending=False).values,
                      "y_2021_2024": yy, "rank_true": pd.Series(yy).rank(ascending=False).values})
    R.sort_values("rank_pred").to_csv(os.path.join(HERE, "final_ranking_2021_2024.csv"), index=False, encoding="utf-8-sig")
    res["module_D"] = run_final_D(P)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False, indent=1))

def run_final_D(P):
    """모듈 D 최종시험: 2022–24 수출(최종시험 때 1회 수집한 _shared/ext/N1final_kr_export_hs6_2019_2024.csv)이 있을 때만."""
    from n1lib import B, load_map, kr_export_ksic, cn_share
    f = os.path.join(B, "_shared", "ext", "N1final_kr_export_hs6_2019_2024.csv")
    if not os.path.exists(f): return {"status": "skipped — 2022–24 수출 미수집(c02_fetch_final_exports.py 를 최종시험 때 1회 실행)"}
    import sD_export as D
    PD = D.PD; train = PD[(PD.t0 <= 2018)].dropna(subset=["yD"])
    x = pd.read_csv(f, dtype={"hs6": str}); x["hs6"] = x.hs6.str.zfill(6)
    mp = load_map()
    # HS2022 개정 처리: 2019–21 과 2022–24 양쪽에 모두 보고된 HS6 코드만 써서 분자·분모를 같은 품목 바구니로 맞춘다(개정으로 사라진 코드는 제외, 커버리지 보고)
    pre, post = set(x[x.year <= 2021].hs6), set(x[x.year >= 2022].hs6); common = pre & post & set(mp.hs6)
    cov = {"common_codes": len(common), "map_codes": int(mp.hs6.nunique()),
           "value_share_2019_21": round(x[(x.year <= 2021) & x.hs6.isin(common)].value.sum() / x[(x.year <= 2021) & x.hs6.isin(mp.hs6)].value.sum(), 3)}
    k = x[x.hs6.isin(common)].merge(mp, on="hs6").assign(v=lambda z: z.value*z.w).pivot_table(index="k5", columns="year", values="v", aggfunc="sum")
    K0 = kr_export_ksic()
    now = k[[2019, 2020, 2021]].sum(1); past = K0[[2016, 2017, 2018]].sum(1).reindex(now.index); fut = k[[2022, 2023, 2024]].sum(1)
    now_all = K0[[2019, 2020, 2021]].sum(1).reindex(now.index)   # 피처(momX, xlvD)는 개발과 같은 전체 품목 기준
    te = pd.DataFrame({"ksic": now.index, "t0": 2021, "momX": np.log(now_all/past).values, "xlvD": np.log(now_all).values})
    te["yD"] = np.log(fut.reindex(now.index)/now).values; te["yD"] -= te.yD.median()
    te = te.merge(P[P.t0 == 2021].drop(columns=[c for c in P.columns if c.startswith("y_")]), on=["ksic", "t0"], how="left")
    te = te[te.ksic.isin(cn_share().ksic.unique()) & te.yD.notna()]; te.index = range(10**6, 10**6 + len(te))
    p, _ = fit_predict(train, te, FD, "yD", ALPHA_D); yy = te.yD.values; pb = te.momX.rank(pct=True).fillna(.5).values
    b = boot_one(te.ksic.values, yy, p, pb)
    return {"rho": round(spearmanr(yy, p)[0], 4), "rho_ci": b["rho_ci"], "B0_rho": round(spearmanr(yy, pb)[0], 4), "d_ci": b["d_ci"],
            "auc_top3": round(auc_top3(yy, p), 4), "hs_map_value_coverage_2022_24": cov, "n": int(len(te))}

# ------------------------------------------------------------------ future
def run_future():
    """최종시험 이후 전망: 결과 ≤2024 레이블(t0 ≤ 2021, 11차 체인)로 학습, t0=2024 → 2027 순위 예측. 최종시험 전 실행 금지."""
    if not os.path.exists(os.path.join(HERE, "final_result.json")): sys.exit("최종시험(final) 이후에만 실행한다.")
    P = ensure_panel("future"); P = P[P.t0 != 2010]
    train = P[(P.t0 >= 2008) & (P.t0 <= 2021)].dropna(subset=["y_emp"]); test = P[P.t0 == 2024].copy()
    p, m = fit_predict(train, test, F, "y_emp", ALPHA)
    from n1lib import ksic_names
    R = pd.DataFrame({"ksic": test.ksic.values, "name": test.ksic.map(ksic_names()).values, "pred": p}).sort_values("pred", ascending=False)
    R["rank"] = range(1, len(R) + 1); R.to_csv(os.path.join(HERE, "future_forecast_2024_2027.csv"), index=False, encoding="utf-8-sig")
    json.dump({"mode": "future", "t0": 2024, "train_t0_max": int(train.t0.max()), "coef": dict(zip(F, m.coef_.round(4).tolist()))},
              open(os.path.join(HERE, "future_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["dev", "final", "future"], default="dev"); a = ap.parse_args()
    {"dev": run_dev, "final": run_final, "future": run_future}[a.mode]()
