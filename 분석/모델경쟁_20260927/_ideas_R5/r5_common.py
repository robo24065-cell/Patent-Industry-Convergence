# -*- coding: utf-8 -*-
"""_ideas_R5 공용: 경로·봉인 상수·평가(AUC·skill·상위 k 적중)·고정 하이퍼파라미터 (r4_common 계승).
🔒 봉인(아이디어별, 빌더에 assert):
  J1 정부R&D 과제 단위 50개 산업 특허 성과: 최종 t=2022 → 출원 2022–23 (F=2022). 개발 레이블은 출원연도 ≤ 2021만 읽는다.
  J2 특허 분쟁(무효·권리범위 심판) 표적: 최종 t0=2022 → 예고등록 2023–24 (F=2023). 등록원부 사건은 날짜 < 2023-01-01 캐시만.
  J3 특허기업 피합병: 최종 t0=2022 → 법인합병 이전등록 2023–24 (F=2023). 등록원부 소유자 이력 < 2023-01-01 캐시만."""
import os
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
R3I = os.path.join(MC, r"_ideas_R3")
R4I = os.path.join(MC, r"_ideas_R4")
R4C = os.path.join(R4I, "cache")
OUT = os.path.join(MC, r"_ideas_R5")
CACHE = os.path.join(OUT, "cache")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
os.makedirs(CACHE, exist_ok=True)
SEAL_DATE = "20230101"
SEAL_DATE_I = 20230101
J1_MAX_APPL_YEAR = 2021


def ro(path=REG):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def auc(y, p, w=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def skill(a, b):
    return (a - b) / (1 - b)


XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def fit_pred(Xtr, ytr, Xte, model="xgb", seed=0, w=None):
    if model == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr, logisticregression__sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def topk(y, p, frac=None, k=None):
    y = np.asarray(y); o = np.argsort(-np.asarray(p), kind="stable")
    if k is None:
        k = max(1, int(round(frac * len(y))))
    return y[o[:k]].mean(), int(y[o[:k]].sum()), k


def boot_ci_skill(y, pm, pb, groups=None, n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    # 군집 부트스트랩을 가중치로 구현: 군집별 뽑힌 횟수(다항분포)를 행 가중치로 쓴다(빠름)
    if groups is None:
        codes = np.arange(len(y)); G = len(y)
    else:
        codes, uniq = pd.factorize(np.asarray(groups)); G = len(uniq)
    out = []
    for _ in range(n):
        cnt = rng.multinomial(G, np.full(G, 1.0 / G))
        w = cnt[codes].astype(float)
        m = w > 0
        if y[m].min() == y[m].max():
            continue
        out.append(skill(auc(y[m], pm[m], w[m]), auc(y[m], pb[m], w[m])))
    return np.percentile(out, [2.5, 97.5])


def run_steps(D, ycol, dev, base, steps, gap=2, group_col=None, models=("xgb", "logit"), neg_frac=None, n_boot=200, min_train=None, seed=0):
    """사전 등록 순차 실험. 원점 v 평가, 학습 t0 ≤ v-gap(결과창이 v 이전에 끝남). 채택: 평균 Δskill ≥ +0.003 AND 원점 60% 이상 우세.
    neg_frac: 학습 음성 표본추출 비율(가중치 1/neg_frac 보정). 평가는 원점 전체."""
    rng = np.random.default_rng(seed)
    if neg_frac:
        keep = (D[ycol].values == 1) | (rng.random(len(D)) < neg_frac)
        wfull = np.where(D[ycol].values == 1, 1.0, 1.0 / neg_frac)
    allr, allo, P = [], [], {}
    for model in models:
        preds = {}

        def score(feats, tag):
            res = {}
            for v in dev:
                msk = (D.t0 <= v - gap) if min_train is None else ((D.t0 <= v - gap) & (D.t0 >= min_train))
                if neg_frac:
                    msk = msk & keep
                tr = D[msk]; te = D[D.t0 == v]
                w = wfull[msk.values] if neg_frac else None
                p = fit_pred(tr[feats].astype(float), tr[ycol], te[feats].astype(float), model=model, w=w)
                preds[(tag, v)] = p
                res[v] = auc(te[ycol], p)
            return res
        rows = []
        champ_feats = list(base); champ = score(champ_feats, "M0"); champ_tag = "M0"; b0 = dict(champ)
        rows.append(dict(model=model, exp="M0", change="기준선", **{f"auc_{v}": champ[v] for v in dev}, decision="BASE"))
        for name, add in steps:
            feats = champ_feats + add; r = score(feats, name)
            sk = np.mean([skill(r[v], champ[v]) for v in dev]); win = np.mean([r[v] > champ[v] for v in dev])
            dec = "ACCEPT" if (sk >= 0.003 and win >= 0.6) else "ROLLBACK"
            rows.append(dict(model=model, exp=name, change="+" + ",".join(add), **{f"auc_{v}": r[v] for v in dev},
                             d_skill_vs_champ=round(sk, 4), win=win, decision=dec))
            print(model, name, round(sk, 4), win, dec, flush=True)
            if dec == "ACCEPT":
                champ_feats, champ, champ_tag = feats, r, name
        out = []
        for v in dev:
            te = D[D.t0 == v]; yv = te[ycol].values
            g = te[group_col].values if group_col else None
            p = preds[(champ_tag, v)]; pb = preds[("M0", v)]
            lo, hi = (boot_ci_skill(yv, p, pb, groups=g, n=n_boot) if champ_tag != "M0" else (np.nan, np.nan))
            t1, h1, k1 = topk(yv, p, 0.01); t1b, h1b, _ = topk(yv, pb, 0.01)
            t5, _, _ = topk(yv, p, 0.05); t5b, _, _ = topk(yv, pb, 0.05)
            _, h100, _ = topk(yv, p, k=100); _, h100b, _ = topk(yv, pb, k=100)
            r10 = topk(yv, p, 0.10)[1] / max(1, yv.sum()); r10b = topk(yv, pb, 0.10)[1] / max(1, yv.sum())
            out.append(dict(model=model, champ=champ_tag, v=v, n=len(te), pos=int(yv.sum()), base_rate=yv.mean(),
                            auc_M0=b0[v], auc_champ=champ[v], skill=skill(champ[v], b0[v]), ci_lo=lo, ci_hi=hi,
                            top1pct=t1, top1pct_M0=t1b, top5pct=t5, top5pct_M0=t5b, hit100=h100, hit100_M0=h100b,
                            recall_top10pct=r10, recall_top10pct_M0=r10b))
        allr.append(pd.DataFrame(rows)); allo.append(pd.DataFrame(out)); P[model] = (preds, champ_tag, champ_feats)
    return pd.concat(allr), pd.concat(allo), P


def norm_name(s):
    import re
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = re.sub(r"\(주\)|㈜|주식회사|유한회사|\(유\)|유한책임회사|\(사\)|사단법인|재단법인|\(재\)|株式会社|co\.?,?\s*ltd\.?|inc\.?|corp(oration)?\.?|limited", "", s, flags=re.I)
    return re.sub(r"[\s\.,\-·()\[\]]", "", s).lower()
