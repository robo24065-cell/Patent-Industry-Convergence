# -*- coding: utf-8 -*-
"""[2] T1 라벨 무관 결과 A 평가 — R8 a_eval.py와 같은 설계(학습 원점 c-6..c-4, 레이블 first_ky≤c, 음성 30% 역가중, 같은 시드·XGB 설정, 평가 전수).
팔: NONE, CW27, P50(재학습, R8 재현 점검), TXT50, FREE50, P50TXT, P50FREE.
사용: python c_eval.py 2008 [...] → res/C_c<c>.json, res/C_pred_c<c>.parquet"""
from r9_common import *
import xgboost as xgb

BASEF = ["f_n", "f_rec3", "f_nsc", "f_age", "s_n", "s_grow", "s_nfirm", "s_newrate",
         "p_sec", "p_relmean", "p_relmax", "p_cls", "p_sect", "p_cf"]
CTX = ["dot", "cos", "adopt", "entry", "qmain", "mommain", "sindmom"]
cx = lambda a: [f"{a}_{x}" for x in CTX]
ARMS = {"NONE": BASEF, "CW27": BASEF + cx("CW27"), "P50": BASEF + cx("P50"),
        "TXT50": BASEF + cx("TXT50"), "FREE50": BASEF + cx("FREE50"),
        "P50TXT": BASEF + cx("P50") + cx("TXT50"), "P50FREE": BASEF + cx("P50") + cx("FREE50")}
PRM = dict(n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=20, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=5, tree_method="hist", nthread=2, eval_metric="auc")
NEG = 0.3
NEED = ["applicantCd", "sc", "t", "first_ky"] + BASEF + cx("CW27") + cx("P50")


def load(t):
    a = pd.read_parquet(os.path.join(NC8, "cache", f"A_o{t}.parquet"), columns=NEED)
    x = pd.read_parquet(os.path.join(CACHE, f"X_o{t}.parquet"))
    assert (a.applicantCd.values == x.applicantCd.values).all() and (a.sc.values == x.sc.values).all()
    return pd.concat([a, x.drop(columns=["applicantCd", "sc"])], axis=1)


def topk_hit(y, s, k):
    o = np.argsort(-np.asarray(s))[:k]
    return float(np.asarray(y)[o].mean())


def auc_w(y, s):
    y = np.asarray(y, dtype=np.int8); s = np.asarray(s, dtype=np.float64)
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    _, idx = np.unique(s, return_index=True)
    wp = np.add.reduceat((y == 1).astype(float), idx); wn = np.add.reduceat((y == 0).astype(float), idx)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def run(c):
    tr = []
    for t in (c - 6, c - 5, c - 4):
        if t < 2002:
            continue
        d = load(t)
        d["y"] = (d.first_ky <= c).astype(np.int8)
        rng = np.random.default_rng(1000 + t)
        d = d[(d.y == 1) | (rng.random(len(d)) < NEG)]
        d["w"] = np.where(d.y == 1, 1.0, 1 / NEG)
        tr.append(d)
    tr = pd.concat(tr, ignore_index=True)
    ev = load(c); ev["y"] = ev.first_ky.notna().astype(np.int8)
    out = {"c": c, "n_train": len(tr), "pos_train": int(tr.y.sum()), "n_eval": len(ev), "pos_eval": int(ev.y.sum()), "arms": {}}
    pred = pd.DataFrame({"applicantCd": ev.applicantCd, "sc": ev.sc, "y": ev.y, "far": (ev.p_cls == 0).astype(np.int8)})
    far = pred.far.values == 1
    for arm, cols in ARMS.items():
        s = np.zeros(len(ev))
        for seed in (0, 1):
            m = xgb.XGBClassifier(random_state=seed, **PRM)
            m.fit(tr[cols], tr.y, sample_weight=tr.w)
            s += m.predict_proba(ev[cols])[:, 1] / 2
        pred[arm] = s.astype(np.float32)
        out["arms"][arm] = {"auc": auc_w(ev.y, s), "auc_far": auc_w(ev.y[far], s[far]), "top1000": topk_hit(ev.y, s, 1000)}
        log(c, arm, out["arms"][arm])
    json.dump(out, open(os.path.join(RES, f"C_c{c}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pred.to_parquet(os.path.join(RES, f"C_pred_c{c}.parquet"), index=False)


if __name__ == "__main__":
    for c in map(int, sys.argv[1:]):
        run(c)
