# -*- coding: utf-8 -*-
"""N9 단계 C: 동결 챔피언 C10 설정(XGB 고정 하이퍼파라미터, 음성 30% 표본 시드 0, 학습 t 2010..v-2)으로 팔별 재학습·예측.
사용: python n9_c_run.py OUTCOME ARM[,ARM..] [ORIGIN ...]
  OUTCOME: y_frozen(동결 결과, 순환 참고) | y_all | y_G | y_H | y_C | y_A | y_B
  ARM: PROV | CW | TXT | NONE | NONE_ALL | M0
산출: preds/n9_{OUTCOME}_{ARM}_{v}.parquet (p) + preds/n9_{OUTCOME}_meta_{v}.parquet (y, org)"""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r9_common import *
sys.path.insert(0, N9)
from n9_lib import fit_pred, NEG_FRAC

GAP = 2
SPEC = json.load(open(os.path.join(N9, "final_spec_r6.json"), encoding="utf-8"))
FEATS = SPEC["feats"]; M0L2 = SPEC["references"]["M0_L2"]["feats"]
P1 = ["org_p50_L2", "org_p50_rate_L2", "org_p50_share_L2", "pre_p50_L2", "chain_p50_L2"]
P2 = ["pt_max", "pt_top", "pt_margin", "pt_ent"]
P3 = ["org_krapp", "org_krapp3", "org_krapp_topk"]
DEF = ["b1_tfidf_score", "b1_tfidf_assigned", "b2_dict_score", "b2_dict_assigned", "b1_top", "top_ksic"]
KP = ["ksic_prior_b1", "ksic_prior_top"]
drop = lambda L: [f for f in FEATS if f not in L]
ARMS = {
    "PROV": dict(feats=FEATS, sub={}),
    "CW": dict(feats=FEATS, sub={**{c: "altb" for c in P1}, **{c: "alt" for c in P2 + P3}}),
    "TXT": dict(feats=FEATS, sub={**{c: "altb" for c in P1}, "org_krapp": "alt", "org_krapp3": "alt", "org_krapp_topk": "txt", **{c: "txt" for c in P2}}),
    "NONE": dict(feats=drop(P1 + P2 + P3), sub={}),
    "NONE_ALL": dict(feats=drop(P1 + P2 + P3 + DEF + KP), sub={}),
    "M0": dict(feats=M0L2, sub={}),
}

outcome = sys.argv[1]; arms = sys.argv[2].split(","); origins = [int(x) for x in sys.argv[3:]] or list(range(2014, 2021))
cols = sorted(set(FEATS + M0L2 + ["pjt_no", "t0", "org"]) - {"txt_clf"})
D = pd.read_parquet(os.path.join(N9, "cache", "panel_dev.parquet"), columns=cols)
Y = pd.read_parquet(os.path.join(CACHE, "n9_outcomes_panel.parquet"))
assert (Y.pjt_no.values == D.pjt_no.values).all()
D["y"] = Y[outcome].values.astype(int)
A = pd.read_parquet(os.path.join(N9E, "alt_features_dev.parquet")); B = pd.read_parquet(os.path.join(N9E, "alt_p1b_dev.parquet"))
T = pd.read_parquet(os.path.join(CACHE, "n9_txt_feats.parquet"))
for S in (A, B, T):
    assert (S.pjt_no.values == D.pjt_no.values).all() and (S.t0.values == D.t0.values).all()
need = set()
for a in arms:
    need |= {f"{c}__{t}" for c, t in ARMS[a]["sub"].items()}
for c in need:
    src = A if c in A.columns else (B if c in B.columns else T)
    D[c] = src[c].values.astype("float32")
del A, B, T
if outcome != "y_frozen":
    L = pd.read_parquet(os.path.join(CACHE, f"n9_lab_{outcome}.parquet"))
    for c in KP:
        D[c] = L[c].values
for c in cols:
    if c not in ("pjt_no", "org") and D[c].dtype == "float64":
        D[c] = D[c].astype("float32")
log("loaded", outcome, arms, D.shape, "base", round(D.y.mean(), 4))

for v in origins:
    tcp = os.path.join(N9, "cache", f"txtclf_dev_{v}.parquet") if outcome == "y_frozen" else os.path.join(CACHE, f"n9_txtclf_{outcome}_{v}.parquet")
    tc = pd.read_parquet(tcp)
    txt = np.full(len(D), np.nan, dtype="float32"); txt[tc.row_id.values] = tc.txt_clf.values
    D["txt_clf"] = txt
    rng = np.random.default_rng(0)
    keep = (D.y.values == 1) | (rng.random(len(D)) < NEG_FRAC)
    w = np.where(D.y.values == 1, 1.0, 1.0 / NEG_FRAC)
    msk = (D.t0.values >= 2010) & (D.t0.values <= v - GAP) & keep
    te = (D.t0.values == v)
    mp = os.path.join(PRED, f"n9_{outcome}_meta_{v}.parquet")
    if not os.path.exists(mp):
        pd.DataFrame({"y": D.y.values[te], "org": D.org.values[te]}).to_parquet(mp, index=False)
    for a in arms:
        sp = ARMS[a]; fe = sp["feats"]
        Xtr = D.loc[msk, fe].copy(); Xte = D.loc[te, fe].copy()
        for c, tg in sp["sub"].items():
            Xtr[c] = D.loc[msk, f"{c}__{tg}"].values; Xte[c] = D.loc[te, f"{c}__{tg}"].values
        p = fit_pred(Xtr.astype(float), D.y.values[msk], Xte.astype(float), w=w[msk])
        pd.DataFrame({"p": p.astype("float64")}).to_parquet(os.path.join(PRED, f"n9_{outcome}_{a}_{v}.parquet"), index=False)
        log(outcome, a, v, "auc", round(wauc(p, D.y.values[te]), 4))
log("end")
