# -*- coding: utf-8 -*-
"""N9 R7 필수성 시험 실행기 — 동결 챔피언 C10 설정 그대로(XGB 고정 하이퍼파라미터, 음성 30% 표본 시드 0, 학습 t 2010..v-2), 피처만 바꿈.
개발 원점 2014–2020(7개). 기준선 M0_L2(동결 정의)로 skill 계산.
사용: python r7_n9_run.py VAR [VAR ...]   → results_dev.csv에 원점별 행 추가(같은 변형은 덮어씀)"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); N9 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, N9)
from n9_lib import CACHE, DEV_ORIGINS, GAP, NEG_FRAC, fit_pred, auc, skill, topk, log

SPEC = json.load(open(os.path.join(N9, "final_spec_r6.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; M0L2 = SPEC["references"]["M0_L2"]["feats"]
P1 = ["org_p50_L2", "org_p50_rate_L2", "org_p50_share_L2", "pre_p50_L2", "chain_p50_L2"]
P2 = ["pt_max", "pt_top", "pt_margin", "pt_ent"]
P3 = ["org_krapp", "org_krapp3", "org_krapp_topk"]
LABSUP = ["ksic_prior_b1", "ksic_prior_top", "txt_clf"]
DEF = ["b1_tfidf_score", "b1_tfidf_assigned", "b2_dict_score", "b2_dict_assigned", "b1_top", "top_ksic"]
drop = lambda L: [f for f in FEATS if f not in L]
VARS = {
    "FULL": dict(feats=FEATS),
    "R_P1": dict(feats=drop(P1)), "R_P2": dict(feats=drop(P2)), "R_P3": dict(feats=drop(P3)),
    "R_DIRECT": dict(feats=drop(P1 + P2 + P3)),
    "R_LABSUP": dict(feats=drop(LABSUP)), "R_DEF": dict(feats=drop(DEF)),
    "R_ALLKSIC": dict(feats=drop(P1 + P2 + P3 + LABSUP + DEF)),
    "S_P1": dict(feats=FEATS, sub={c: "alt" for c in P1}), "S_P1b": dict(feats=FEATS, sub={c: "altb" for c in P1}),
    "S_P2": dict(feats=FEATS, sub={c: "alt" for c in P2}), "S_P3": dict(feats=FEATS, sub={c: "alt" for c in P3}),
    "S_DIRECT": dict(feats=FEATS, sub={c: "alt" for c in P1 + P2 + P3}),
    "S_DIRECTb": dict(feats=FEATS, sub={**{c: "altb" for c in P1}, **{c: "alt" for c in P2 + P3}}),
}


def load():
    D = pd.read_parquet(f"{CACHE}/panel_dev.parquet").reset_index(drop=True)
    A = pd.read_parquet(os.path.join(HERE, "alt_features_dev.parquet")); B = pd.read_parquet(os.path.join(HERE, "alt_p1b_dev.parquet"))
    assert (A.pjt_no.values == D.pjt_no.values).all() and (A.t0.values == D.t0.values).all() and (B.pjt_no.values == D.pjt_no.values).all()
    for c in A.columns[2:]:
        D[c] = A[c].values
    for c in B.columns[2:]:
        D[c] = B[c].values
    return D


def with_txt(D, v):
    tc = pd.read_parquet(f"{CACHE}/txtclf_dev_{v}.parquet")
    X = D.copy(); X["txt_clf"] = np.nan; X.loc[tc.row_id.values, "txt_clf"] = tc.txt_clf.values
    return X


def predict(X, v, feats):
    rng = np.random.default_rng(0)
    keep = (X.y.values == 1) | (rng.random(len(X)) < NEG_FRAC)
    w = np.where(X.y.values == 1, 1.0, 1.0 / NEG_FRAC)
    msk = (X.t0 >= 2010).values & (X.t0 <= v - GAP).values & keep
    tr = X[msk]; te = X[X.t0 == v]
    return fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float), w=w[msk])


def main(names):
    D = load(); fp = os.path.join(HERE, "results_dev.csv")
    old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
    m0p = os.path.join(HERE, "m0l2_dev.parquet")
    M0 = pd.read_parquet(m0p) if os.path.exists(m0p) else None
    rows = []
    if M0 is None:
        mm = []
        for v in DEV_ORIGINS:
            X = with_txt(D, v); mm.append(pd.DataFrame({"t0": v, "p": predict(X, v, M0L2)}))
        M0 = pd.concat(mm, ignore_index=True); M0.to_parquet(m0p, index=False); log("M0_L2 saved")
    for nm in names:
        spec = VARS[nm]
        for v in DEV_ORIGINS:
            X = with_txt(D, v)
            for c, tag in spec.get("sub", {}).items():
                X[c] = X[f"{c}__{tag}"].values
            te = X[X.t0 == v]; y = te.y.values
            p = predict(X, v, spec["feats"]); pb = M0[M0.t0 == v].p.values
            a, a0 = auc(y, p), auc(y, pb)
            rows.append(dict(variant=nm, origin=v, n_feats=len(spec["feats"]), auc=a, auc_M0L2=a0, skill=skill(a, a0),
                             top1=topk(y, p, 0.01)[0], top10=topk(y, p, 0.10)[0], hit100=topk(y, p, k=100)[1]))
            log(nm, v, round(a, 4), round(skill(a, a0), 4))
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.variant.isin(names)]
    pd.concat([old, new], ignore_index=True).to_csv(fp, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(sys.argv[1:])
