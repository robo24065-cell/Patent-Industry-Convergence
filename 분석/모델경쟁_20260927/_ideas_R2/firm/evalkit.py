# -*- coding: utf-8 -*-
"""전진검증 공용 도구. 원점 v 평가 시 학습 = t0 ≤ v − h (결과가 v 시점에 이미 관측된 원점만)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import xgboost as xgb


def logit_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def xgb_model(**kw):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=10, reg_lambda=5, nthread=2, eval_metric="logloss", verbosity=0)
    p.update(kw)
    return xgb.XGBClassifier(**p)


def walk(df, y, feats, dev, h, model="logit", min_train_t0=None, **kw):
    """원점별 AUC 반환 dict {v: auc}, 예측 dict."""
    out, preds = {}, {}
    d = df[df[y].notna()]
    for v in dev:
        tr = d[(d.t0 <= v - h)]
        if min_train_t0 is not None:
            tr = tr[tr.t0 >= min_train_t0]
        te = d[d.t0 == v]
        if len(tr) == 0 or te[y].nunique() < 2:
            continue
        m = logit_model() if model == "logit" else xgb_model(**kw)
        m.fit(tr[feats], tr[y])
        p = m.predict_proba(te[feats])[:, 1]
        out[v] = roc_auc_score(te[y], p)
        preds[v] = pd.Series(p, index=te.index)
    return out, preds


def skill(a_m, a_b):
    return {v: (a_m[v] - a_b[v]) / (1 - a_b[v]) for v in a_m if v in a_b}


def summarize(name, a_m, a_b):
    s = skill(a_m, a_b)
    return {"exp": name, **{f"auc_{v}": round(a_m[v], 4) for v in a_m}, **{f"base_{v}": round(a_b[v], 4) for v in a_b},
            "mean_auc": round(np.mean(list(a_m.values())), 4), "mean_base": round(np.mean(list(a_b.values())), 4),
            "mean_skill": round(np.mean(list(s.values())), 4)}
