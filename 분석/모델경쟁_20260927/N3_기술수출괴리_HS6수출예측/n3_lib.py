# -*- coding: utf-8 -*-
"""N3 평가 엔진 — 전진 검증(walk-forward) + 챔피언/도전자 판정.
개발 평가 원점 v = 2010..2018, 각 v의 학습 = t0 ≤ v−3 (결과 구간 ≤ v). 🔒 t0=2021 미사용.
채택 판정은 고정 3시드(0,1,2)의 원점별 MAE 평균으로 한다(개발 첫 단계에 고정).
skill_vsB1(v) = 1 − MAE_모형(v)/MAE_B1(v), 평균 skill = 9개 원점 평균. 채택: 챔피언 대비 +0.003 이상 & 원점 60%(6/9) 이상 우세.
"""
import os, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from common_n3 import CACHE, R2

DEV_V = list(range(2010, 2019))
SEEDS = (0, 1, 2)
BASE_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                   objective="reg:absoluteerror", nthread=2)
OWN = {"X": ["lvl", "mom3", "mom1", "vol", "imp_ratio", "imp_mom", "k_mom3"],
       "M": ["lvl", "mom3", "mom1", "vol", "cn_sh", "cn_sh_chg", "k_mom3"]}
PAT_LVL = ["pl_KR", "pl_US", "pl_JP", "pl_EP", "pl_CN", "pl_KRdom", "pl_KRin", "pl_KRout", "pl_CNout", "pl_FOR",
           "p_domshare", "p_outshare", "p_krshare5"]


def derive(D):
    """원점 내 상대화 피처(같은 원점 t0의 품목 분포만 사용 — 원점 시점 정보)."""
    for c in ["mom3", "mom1", "k_mom3", "imp_mom", "mom_prev"]:
        if c in D:
            D[c + "_rel"] = D[c] - D.groupby("t0")[c].transform("median")
    for c in ["gap_kr_h", "gap_kr_lr", "gap_cn"]:
        if c in D:
            D[c + "_abs"] = D[c].abs()
    for c in ["gap_kr", "gap_kr_lr", "gap_cn", "gap_jp", "gap_kr5", "gap_kr_h", "p_outshare", "p_krshare5", "p_domshare"]:
        if c in D:
            D[c + "_rk"] = D.groupby("t0")[c].rank(pct=True)
    return D


def load_panel(flow, variant=""):
    return derive(pd.read_parquet(os.path.join(CACHE, f"panel_{flow}{variant}.parquet")))


def _fit_predict(tr, te, feats, spec, seed):
    kind = spec.get("model", "xgb")
    p = dict(BASE_PARAMS); p.update(spec.get("params", {}))
    w = None
    if spec.get("recency"):  # 최근 원점 가중
        w = np.power(spec["recency"], (tr.t0.max() - tr.t0).values)
    y = tr.y.values
    if spec.get("target_clip"):
        c = spec["target_clip"]; y = np.clip(y, -c, c)
    if kind == "xgb":
        mono = spec.get("mono")
        if mono:
            p["monotone_constraints"] = "(" + ",".join(str(mono.get(f, 0)) for f in feats) + ")"
        mdl = xgb.XGBRegressor(random_state=seed, **p)
        mdl.fit(tr[feats], y, sample_weight=w)
        return mdl.predict(te[feats])
    if kind == "lin_then_xgb":
        # 1단계: 특허 피처 선형 LAD(중앙값 회귀, 표준화) → 2단계: 잔차에 XGB(전 피처)
        from sklearn.linear_model import QuantileRegressor
        lf = spec["lin_feats"]
        mu, sd = tr[lf].mean(), tr[lf].std().replace(0, 1)
        Xl = ((tr[lf] - mu) / sd).fillna(0).values; Xt = ((te[lf] - mu) / sd).fillna(0).values
        q = QuantileRegressor(quantile=0.5, alpha=spec.get("alpha", 0.01), solver="highs").fit(Xl, y)
        r = y - q.predict(Xl)
        mdl = xgb.XGBRegressor(random_state=seed, **p); mdl.fit(tr[feats], r, sample_weight=w)
        return q.predict(Xt) + mdl.predict(te[feats])
    if kind == "blend":
        # XGB 두 개(피처 집합 A·B)의 예측 평균
        a = _fit_predict(tr, te, feats, {**spec, "model": "xgb"}, seed)
        b = _fit_predict(tr, te, spec["feats_b"], {**spec, "model": "xgb", "params": {**spec.get("params", {}), **spec.get("params_b", {"max_depth": 2, "n_estimators": 500})}}, seed)
        wa = spec.get("wa", 0.5)
        return wa * a + (1 - wa) * b
    raise ValueError(kind)


def evaluate(D, feats, spec=None, seeds=SEEDS, dev_v=DEV_V, horizon=3, eval_mask=None, return_preds=False):
    spec = spec or {}
    res, preds = [], []
    for v in dev_v:
        tr = D[(D.t0 <= v - horizon) & (D.t0 >= spec.get("min_train_t0", 0))]
        te = D[D.t0 == v]
        if eval_mask is not None:
            te = te[eval_mask(te)]
        top = (te.y >= te.y.quantile(0.8)).astype(int)
        maes, rhos, aucs, P = [], [], [], []
        for s in seeds:
            p = _fit_predict(tr, te, feats, spec, s)
            P.append(p); maes.append(np.abs(te.y.values - p).mean())
            rhos.append(spearmanr(p, te.y).correlation); aucs.append(roc_auc_score(top, p))
        res.append({"v": v, "n": len(te), "mae": float(np.mean(maes)), "mae_B0": float(np.abs(te.y).mean()),
                    "rho": float(np.mean(rhos)), "auc": float(np.mean(aucs)), "mae_seeds": [float(x) for x in maes]})
        if return_preds:
            pp = te[["hs6", "t0", "k5", "y"]].copy(); pp["pred"] = np.mean(P, axis=0); preds.append(pp)
    R = pd.DataFrame(res)
    return (R, pd.concat(preds)) if return_preds else R


# ---------------------------------------------------------------- 챔피언/도전자 기록
def _paths(flow):
    tag = "" if flow == "X" else "_G1"
    return (os.path.join(R2, f"experiments{tag}.csv"), os.path.join(R2, f"champion{tag}.json"),
            os.path.join(R2, f"b1{tag}.json"))


def get_b1(D, flow):
    _, _, bp = _paths(flow)
    if os.path.exists(bp):
        return pd.DataFrame(json.load(open(bp, encoding="utf-8")))
    R = evaluate(D, OWN[flow], {})
    R.to_json(bp, orient="records", force_ascii=False)
    return R


def summarize(R, B1):
    sk = 1 - R.mae.values / B1.mae.values
    return {"skill_vsB1": float(sk.mean()), "skill_vsB0": float((1 - R.mae / R.mae_B0).mean()),
            "rho": float(R.rho.mean()), "auc": float(R.auc.mean()), "mae": float(R.mae.mean()),
            "win_vsB1": float((R.mae.values < B1.mae.values).mean()), "skills": sk}


def record(flow, exp_id, parent, change, feats, spec, R, B1, note="", decide=True, force_decision=None):
    ep, cp, _ = _paths(flow)
    S = summarize(R, B1)
    champ = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None
    d, win, dec = np.nan, np.nan, "BASE"
    if champ is not None:
        cm = np.array(champ["mae"])
        csk = 1 - cm / B1.mae.values
        d = float(S["skills"].mean() - csk.mean())
        win = float((R.mae.values < cm).mean())
        dec = "ACCEPT" if (d >= 0.003 and win >= 0.6) else "ROLLBACK"
    if force_decision:
        dec = force_decision
    row = {"exp_id": exp_id, "parent": parent, "change": change, "decision": dec,
           "mean_skill_vsB1": round(S["skill_vsB1"], 4), "d_vs_champ": None if np.isnan(d) else round(d, 4),
           "win_vs_champ": None if np.isnan(win) else round(win, 2),
           "skill_vsB0": round(S["skill_vsB0"], 4), "mae": round(S["mae"], 4), "rho": round(S["rho"], 4),
           "auc_top20": round(S["auc"], 4), "win_vsB1": round(S["win_vsB1"], 2)}
    for v, sk, m in zip(R.v, S["skills"], R.mae):
        row[f"skill_{v}"] = round(float(sk), 4)
    for v, m in zip(R.v, R.mae):
        row[f"mae_{v}"] = round(float(m), 4)
    row["n_feats"] = len(feats); row["note"] = note
    df = pd.DataFrame([row])
    if os.path.exists(ep):
        old = pd.read_csv(ep, encoding="utf-8-sig")
        old = old[old.exp_id != exp_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(ep, index=False, encoding="utf-8-sig")
    if dec in ("ACCEPT", "START"):
        json.dump({"exp_id": exp_id, "feats": feats, "spec": spec, "mae": [float(x) for x in R.mae]},
                  open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{flow}] {exp_id} {dec} skill={S['skill_vsB1']:.4f} d={row['d_vs_champ']} win={row['win_vs_champ']} "
          f"vsB0={S['skill_vsB0']:.4f} rho={S['rho']:.3f} auc={S['auc']:.3f}", flush=True)
    return row


def champion(flow):
    _, cp, _ = _paths(flow)
    return json.load(open(cp, encoding="utf-8"))
