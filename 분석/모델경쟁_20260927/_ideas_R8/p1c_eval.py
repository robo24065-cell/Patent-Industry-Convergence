# -*- coding: utf-8 -*-
"""P1 개발 원점 평가 + 7라운드식 필수성(대체·KR만·재정의). 최종시험 단위(t≥2019)는 결과가 없고 여기서 쓰지 않는다."""
import sys
sys.dont_write_bytecode = True
from r8_common import *
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

P = pd.read_parquet(p("cache", "p1_panel.parquet"))
P = P[P.t <= 2018].reset_index(drop=True)
assert (P.y >= 0).all()
cols = P.columns
pro = [c for c in cols if c.startswith("prov__")]
SETS = {
    "M0_prov_KR6": ["prov__KR_shP", "prov__KR_dsh", "prov__KR_lnn", "prov__KR_lng", "prov__g_grow", "prov__KR_shT"],
    "NOLAB_ipc_only": [f"prov__{cc}_{x}" for cc in ["KR", "US", "JP", "CN", "EP"] for x in ("lng", "lngPP")] + ["prov__g_grow"],
    "FULL_prov5c": pro,
    "KRONLY_prov": [c for c in pro if c.startswith("prov__KR_")] + ["prov__g_grow", "prov__n_lab_g"],
    "TXT_sub": [c for c in cols if c.startswith("txt__")],
    "CW_sub": [c for c in cols if c.startswith("cw__")],
}
EVAL_T = list(range(2010, 2019))


def fit_pred(name, tr, te, X):
    if name.startswith("M0"):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        m.fit(tr[X].fillna(0), tr.y)
        return m.predict_proba(te[X].fillna(0))[:, 1]
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=2, tree_method="hist", random_state=0, eval_metric="logloss")
    m.fit(tr[X], tr.y)
    return m.predict_proba(te[X])[:, 1]


rows, preds = [], []
for t in EVAL_T:
    tr = P[P.t <= t - 3]; te = P[P.t == t]
    rec = {"t": t, "n": len(te), "pos": int(te.y.sum()), "n_train": len(tr)}
    for nm, X in SETS.items():
        s = fit_pred(nm, tr, te, X)
        rec[nm] = auc(te.y, s); rec[nm + "_top1"] = topk_hit(te.y.values, s, 0.01)
        w = te.W.values == 1
        rec[nm + "_W"] = auc(te.y.values[w], s[w])
        if nm in ("FULL_prov5c", "TXT_sub", "CW_sub"):
            preds.append(pd.DataFrame({"t": t, "mg": te.mg.values, "k": te.k.values, "m": nm, "s": s}))
    rows.append(rec); log(t, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float) and not k.endswith("top1")})
R = pd.DataFrame(rows)
R.to_csv(p("p1_by_origin.csv"), index=False)
pd.concat(preds).to_parquet(p("cache", "p1_preds.parquet"), index=False)

summ = {}
for nm in SETS:
    summ[nm] = {"auc": float(R[nm].mean()), "auc_W": float(R[nm + "_W"].mean()), "top1": float(R[nm + "_top1"].mean()),
                "skill_vs_M0": float(skill(R[nm], R["M0_prov_KR6"]).mean())}
comp = {}
for a, b in (("FULL_prov5c", "TXT_sub"), ("FULL_prov5c", "CW_sub"), ("FULL_prov5c", "KRONLY_prov"),
             ("FULL_prov5c", "M0_prov_KR6"), ("TXT_sub", "CW_sub"), ("FULL_prov5c", "NOLAB_ipc_only")):
    for suf in ("", "_W"):
        d = (R[a + suf] - R[b + suf]).values
        comp[f"{a}-{b}{suf}"] = {"mean": float(np.nanmean(d)), "pos_origins": f"{int((d > 0).sum())}/{int((~np.isnan(d)).sum())}",
                                 "ci": boot_ci(d)}
# 재정의 시험: 같은 '이동 사건'을 대체 라벨로 정의하면?
red = {}
for a in ("y_cw", "y_txt"):
    yy, za = P.y.values == 1, P[a].values == 1
    red[a] = {"rate": float(za.mean()), "rate_prov": float(yy.mean()), "recall_of_prov": float((yy & za).sum() / yy.sum()),
              "precision": float((yy & za).sum() / max(za.sum(), 1)), "jaccard": float((yy & za).sum() / (yy | za).sum())}
    w = P.W.values == 1
    red[a]["recall_of_prov_W"] = float((yy & za & w).sum() / (yy & w).sum())
red["prov_events_in_W_share"] = float(((P.y == 1) & (P.W == 1)).sum() / (P.y == 1).sum())
red["units_W_share"] = float(P.W.mean())
out = {"summary": summ, "compare": comp, "redefinition": red, "n_units_dev": int(len(P)), "pos": int(P.y.sum()),
       "base_rate": float(P.y.mean()), "eval_origins": EVAL_T}
json.dump(out, open(p("p1_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
log("end")
