# -*- coding: utf-8 -*-
"""_ideas_R4 공용: 경로·봉인 상수·평가(AUC·skill)·고정 하이퍼파라미터.
🔒 봉인: 이 폴더 스크립트는 날짜 >= 2023-01-01 인 등록원부 사건(이전·실시권·소멸·납부)과 연도 >= 2023 NTIS 과제를 읽지 않는다.
   (I6 최종시험 결과창 2023-01-01~2024-12-31, I7 최종시험 결과창 과제연도 2023–2024, F=2023)
   개발·학습 원점은 결과창이 2022-12-31 이전에 끝나는 것만 쓴다."""
import os, re
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
R3I = os.path.join(MC, r"_ideas_R3")
OUT = os.path.join(MC, r"_ideas_R4")
CACHE = os.path.join(OUT, "cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
os.makedirs(CACHE, exist_ok=True)
SEAL_DATE = "20230101"   # 이 날짜 이상 사건은 읽지 않음
SEAL_YEAR = 2023         # NTIS 과제연도 이 값 이상 읽지 않음


def ro(path=REG):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p)


def strat_auc(y, p, g):
    """기관 내 AUC: 같은 군집(기관) 안의 양성-음성 쌍만 비교(쌍 수 가중 평균)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(g)})
    num = den = 0.0
    for _, s in d.groupby("g"):
        npos = s.y.sum(); nneg = len(s) - npos
        if npos == 0 or nneg == 0:
            continue
        r = s.p.rank().values
        a = (r[s.y.values == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        num += a * npos * nneg; den += npos * nneg
    return num / den if den > 0 else np.nan


def skill(a, b):
    return (a - b) / (1 - b)


XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def fit_pred(Xtr, ytr, Xte, model="xgb", seed=0):
    if model == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def boot_ci_skill(y, pm, pb, groups=None, n=300, seed=0, strat=False):
    """원점 하나 안에서 부트스트랩 skill 95% CI. groups가 있으면 군집(기관 등) 부트스트랩."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    if groups is None:
        idx_sets = None
    else:
        g = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(lambda s: s.values)
        idx_sets = list(g.values)
    out = []
    for _ in range(n):
        if idx_sets is None:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(idx_sets), len(idx_sets))
            i = np.concatenate([idx_sets[k] for k in pick])
        if y[i].min() == y[i].max():
            continue
        if strat:
            # 재표본된 군집마다 새 id를 줘서 같은 기관이 두 번 뽑히면 별개 군집으로 취급
            gid = np.concatenate([np.full(len(idx_sets[k]), j) for j, k in enumerate(pick)])
            out.append(skill(strat_auc(y[i], pm[i], gid), strat_auc(y[i], pb[i], gid)))
        else:
            out.append(skill(auc(y[i], pm[i]), auc(y[i], pb[i])))
    return np.percentile(out, [2.5, 97.5])


def run_steps(D, ycol, dev, base, steps, group_col=None, min_train=None, models=("xgb", "logit"), size=None, metric="pooled", n_boot=200):
    """사전 등록 순차 실험. 원점 v 평가, 학습 t0 ≤ v-2. 채택: 평균 Δskill ≥ +0.003 AND 원점 60% 이상 우세."""
    allr, allo = [], []
    for model in models:
        preds = {}

        def score(feats, tag):
            res = {}
            for v in dev:
                tr = D[(D.t0 <= v - 2)] if min_train is None else D[(D.t0 <= v - 2) & (D.t0 >= min_train)]
                te = D[D.t0 == v]
                p = fit_pred(tr[feats].astype(float), tr[ycol], te[feats].astype(float), model=model)
                preds[(tag, v)] = p
                res[v] = strat_auc(te[ycol], p, te[group_col]) if metric == "strat" else auc(te[ycol], p)
            return res
        rows = []
        if size:
            rs = score([size], "B_size")
            rows.append(dict(model=model, exp="B_size", change=f"규모 1변수({size})", **{f"auc_{v}": rs[v] for v in dev}))
        champ_feats = list(base); champ = score(champ_feats, "M0"); champ_tag = "M0"; b0 = dict(champ)
        rows.append(dict(model=model, exp="M0", change="기준선", **{f"auc_{v}": champ[v] for v in dev}, decision="BASE"))
        for name, add in steps:
            feats = champ_feats + add; r = score(feats, name)
            sk = np.mean([skill(r[v], champ[v]) for v in dev]); win = np.mean([r[v] > champ[v] for v in dev])
            dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
            rows.append(dict(model=model, exp=name, change="+" + ",".join(add), **{f"auc_{v}": r[v] for v in dev},
                             d_skill_vs_champ=round(sk, 4), win=win, decision=dec))
            if dec == "ACCEPT":
                champ_feats, champ, champ_tag = feats, r, name
        out = []
        for v in dev:
            te = D[D.t0 == v]; yv = te[ycol].values
            g = te[group_col].values if group_col else None
            if champ_tag != "M0":
                lo, hi = boot_ci_skill(yv, preds[(champ_tag, v)], preds[("M0", v)], groups=g, n=n_boot, strat=(metric == "strat"))
            else:
                lo = hi = np.nan
            p = preds[(champ_tag, v)]; pb = preds[("M0", v)]; k = max(1, int(0.01 * len(te)))
            out.append(dict(model=model, champ=champ_tag, v=v, n=len(te), pos=int(yv.sum()), base_rate=yv.mean(),
                            auc_M0=b0[v], auc_champ=champ[v], pooled_M0=auc(yv, pb), pooled_champ=auc(yv, p), skill=skill(champ[v], b0[v]), ci_lo=lo, ci_hi=hi,
                            top1pct_prec=yv[np.argsort(-p)[:k]].mean(), top1pct_prec_M0=yv[np.argsort(-pb)[:k]].mean()))
        allr.append(pd.DataFrame(rows)); allo.append(pd.DataFrame(out))
    return pd.concat(allr), pd.concat(allo), preds
