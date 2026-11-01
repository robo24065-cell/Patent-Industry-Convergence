# -*- coding: utf-8 -*-
"""(1) 결과 A 평가: 평가 원점 c(2008–2012, 개발 원점)마다 학습 원점 t∈{c-6,c-5,c-4}(창 끝 ≤ c),
학습 레이블은 c 연말까지 가시(first_ky ≤ c)한 출원만. 평가 레이블은 최종 등록 기준.
팔(arm): NONE(라벨 무관), CW27(+공식 연계표 산업맥락), P27(+제공 라벨을 연계표 그룹으로 접은 산업맥락), P50(+제공 50), P50CW(둘 다).
사용: python a_eval.py 2008 [2009 ...] → res/A_c<c>.json, res/A_pred_c<c>.parquet
"""
from nc_common import *
import xgboost as xgb

RES = os.path.join(OUT, "res"); os.makedirs(RES, exist_ok=True)
BASEF = ["f_n", "f_rec3", "f_nsc", "f_age", "s_n", "s_grow", "s_nfirm", "s_newrate",
         "p_sec", "p_relmean", "p_relmax", "p_cls", "p_sect", "p_cf"]
CTX = ["dot", "cos", "adopt", "entry", "qmain", "mommain", "sindmom"]
ARMS = {"NONE": BASEF,
        "CW27": BASEF + [f"CW27_{x}" for x in CTX],
        "P27": BASEF + [f"P27_{x}" for x in CTX],
        "P50": BASEF + [f"P50_{x}" for x in CTX],
        "P50CW": BASEF + [f"P50_{x}" for x in CTX] + [f"CW27_{x}" for x in CTX]}
PRM = dict(n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=20, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=5, tree_method="hist", nthread=2, eval_metric="auc")
NEG = 0.3


def run(c):
    tr = []
    for t in (c - 6, c - 5, c - 4):
        f = os.path.join(CACHE, f"A_o{t}.parquet")
        if t < 2002 or not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        d["y"] = (d.first_ky <= c).astype(np.int8)
        rng = np.random.default_rng(1000 + t)
        d = d[(d.y == 1) | (rng.random(len(d)) < NEG)]
        d["w"] = np.where(d.y == 1, 1.0, 1 / NEG)
        tr.append(d)
    tr = pd.concat(tr, ignore_index=True)
    ev = pd.read_parquet(os.path.join(CACHE, f"A_o{c}.parquet"))
    ev["y"] = ev.first_ky.notna().astype(np.int8)
    out = {"c": c, "train_t": sorted(tr.t.unique().tolist()), "n_train": len(tr), "pos_train": int(tr.y.sum()),
           "n_eval": len(ev), "pos_eval": int(ev.y.sum()), "arms": {}}
    pred = pd.DataFrame({"applicantCd": ev.applicantCd, "sc": ev.sc, "y": ev.y, "far": (ev.p_cls == 0).astype(np.int8)})
    for arm, cols in ARMS.items():
        s = np.zeros(len(ev))
        for seed in (0, 1):
            m = xgb.XGBClassifier(random_state=seed, **PRM)
            m.fit(tr[cols], tr.y, sample_weight=tr.w)
            s += m.predict_proba(ev[cols])[:, 1] / 2
        pred[arm] = s
        far = pred.far.values == 1
        out["arms"][arm] = {"auc": auc_w(ev.y, s), "auc_far": auc_w(ev.y[far], s[far]), "top1000": topk_hit(ev.y, s, 1000)}
        log(c, arm, out["arms"][arm])
    cl = pred.applicantCd.values
    out["delta"] = {}
    for a, b in (("P50", "CW27"), ("P27", "CW27"), ("CW27", "NONE"), ("P50", "NONE"), ("P27", "NONE"), ("P50CW", "CW27"), ("P50", "P27")):
        lo, hi, pr = cluster_boot_delta(pred.y.values, pred[a].values, pred[b].values, cl, B=200, seed=c)
        out["delta"][f"{a}-{b}"] = {"d": out["arms"][a]["auc"] - out["arms"][b]["auc"], "ci": [lo, hi], "p_gt0": pr}
        log(c, a, "-", b, out["delta"][f"{a}-{b}"])
    out["n_eval_far"] = int(pred.far.sum()); out["pos_eval_far"] = int(pred.y[pred.far == 1].sum())
    json.dump(out, open(os.path.join(RES, f"A_c{c}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pred.to_parquet(os.path.join(RES, f"A_pred_c{c}.parquet"), index=False)


if __name__ == "__main__":
    for c in map(int, sys.argv[1:]):
        run(c)
