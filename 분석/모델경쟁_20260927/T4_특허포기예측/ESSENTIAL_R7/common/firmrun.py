# -*- coding: utf-8 -*-
"""R7 공용: 기업 단위 트랙(N4·N7) 피처 변형 재학습 러너. 동결 챔피언 모형(predict)·원점 규칙 그대로, 피처 목록만 바꾼다."""
import os, json, time
import numpy as np, pandas as pd


def skill(a, b):
    return (a - b) / (1 - b)


def run_variant(D, feats, origins, predict, auc, seeds, model, out_json, base_preds, train_min=2015, ycol="y", extra=None):
    """D: dev 패널(원점별 행), feats: 피처 목록. base_preds: {v: M0 예측 배열}. 결과 json 저장."""
    t = time.time()
    res = {}
    for v in origins:
        tr = D[(D.t0 >= train_min) & (D.t0 <= v - 2)]; te = D[D.t0 == v]
        p = predict(tr[feats].astype(float), tr[ycol], te[feats].astype(float), model=model, seeds=seeds)
        y = te[ycol].values
        a = float(auc(y, p)); a0 = float(auc(y, base_preds[v]))
        k = max(1, int(round(0.01 * len(y))))
        res[str(v)] = dict(auc=a, auc_M0=a0, skill=float(skill(a, a0)), n=int(len(y)), pos=int(y.sum()),
                           top1pct=float(y[np.argsort(-p)[:k]].mean()))
    o = dict(feats=feats, n_feats=len(feats), by_origin=res,
             mean_auc=float(np.mean([r["auc"] for r in res.values()])),
             mean_skill=float(np.mean([r["skill"] for r in res.values()])), sec=round(time.time() - t))
    if extra: o.update(extra)
    json.dump(o, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return o
