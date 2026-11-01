# -*- coding: utf-8 -*-
"""_ideas_R3 공용: 경로, 상호 정규화, 평가(AUC·skill), 고정 하이퍼파라미터 XGB.
봉인: 이 폴더의 스크립트는 NPS 2024-12 이후 스냅샷을 읽지 않는다(I2·I3 최종시험 결과창 2024–25).
      I1 최종시험(t0=2021, 결과 출원 2022–23)의 결과변수는 계산하지 않는다(출원일 ≥2022 레이블 생성 금지)."""
import os, re
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
OUT = os.path.join(MC, r"_ideas_R3")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

# 개발에 쓸 수 있는 NPS 스냅샷(연말). 2024-12 이후는 봉인.
SNAPS_OK = ["2015-12", "2016-12", "2017-12", "2018-12", "2019-12", "2020-12", "2021-12", "2022-12", "2023-12"]
SEAL_I1_OUTCOME_YEAR = 2022   # I1: 출원일 연도 >= 2022 레이블 금지
SEAL_I23_SNAP = "2024-12"     # I2·I3: 이 스냅샷 이후 금지

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p)


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
    else:  # 표준화 로지스틱(결측=중앙값)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]


def boot_ci_skill(y, pm, pb, n=300, seed=0):
    """원점 하나 안에서 단위 부트스트랩으로 skill의 95% CI"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(skill(auc(y[i], pm[i]), auc(y[i], pb[i])))
    return np.percentile(out, [2.5, 97.5])
