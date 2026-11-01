# -*- coding: utf-8 -*-
"""(3) 2×2 평가. 결과 D∈{P,C} × 피처 L∈{P,C}. 평가 원점 c 2008–2012, 학습 원점 c-6..c-4(레이블은 c까지 가시분).
모든 팔에 공통 기반 = 법인 일반(f_*) + '결과 정의 D'의 그룹 통계(gN·ggrow·gnew). 팔별로 라벨 체계 L의 법인-그룹 피처를 더한다.
변형 nodirect: L의 직접 보유 피처(m, n)를 뺀 구조 피처만(co·en·fit·mmax·ent·nm).
사용: python c_eval.py 2008 ... → res/C_c<c>.json
"""
from nc_common import *
import xgboost as xgb

RES = os.path.join(OUT, "res"); os.makedirs(RES, exist_ok=True)
FB = ["f_n", "f_nsc", "f_rec3", "f_age"]
PAIR = ["m", "n", "co", "en", "fit", "mmax", "ent", "nm"]
PAIR_ND = ["co", "en", "fit", "mmax", "ent", "nm"]
PRM = dict(n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=20, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=5, tree_method="hist", nthread=2)
NEG = 0.5


def arms(D):
    b = FB + [f"{D}_gN", f"{D}_ggrow", f"{D}_gnew"]
    return {"NONE": b,
            "F_P": b + [f"P_{x}" for x in PAIR], "F_C": b + [f"C_{x}" for x in PAIR],
            "F_PC": b + [f"P_{x}" for x in PAIR] + [f"C_{x}" for x in PAIR],
            "F_P_nd": b + [f"P_{x}" for x in PAIR_ND], "F_C_nd": b + [f"C_{x}" for x in PAIR_ND]}


def run(c):
    out = {"c": c}
    for D in ("P", "C"):
        tr = []
        for t in (c - 6, c - 5, c - 4):
            f = os.path.join(CACHE, f"C_o{t}_{D}.parquet")
            if t < 2002 or not os.path.exists(f):
                continue
            d = pd.read_parquet(f); d["y"] = (d.first_ky <= c).astype(np.int8)
            rng = np.random.default_rng(2000 + t)
            d = d[(d.y == 1) | (rng.random(len(d)) < NEG)]; d["w"] = np.where(d.y == 1, 1.0, 1 / NEG)
            tr.append(d)
        tr = pd.concat(tr, ignore_index=True)
        ev = pd.read_parquet(os.path.join(CACHE, f"C_o{c}_{D}.parquet")); ev["y"] = ev.first_ky.notna().astype(np.int8)
        o = {"n_train": len(tr), "pos_train": int(tr.y.sum()), "n_eval": len(ev), "pos_eval": int(ev.y.sum()), "arms": {}}
        S = {}
        for arm, cols in arms(D).items():
            s = np.zeros(len(ev))
            for seed in (0, 1):
                m = xgb.XGBClassifier(random_state=seed, **PRM); m.fit(tr[cols], tr.y, sample_weight=tr.w)
                s += m.predict_proba(ev[cols])[:, 1] / 2
            S[arm] = s
            o["arms"][arm] = {"auc": auc_w(ev.y, s), "top500": topk_hit(ev.y, s, 500)}
            log(c, "Y_" + D, arm, o["arms"][arm])
        o["delta"] = {}
        for a, b in (("F_P", "F_C"), ("F_P_nd", "F_C_nd"), ("F_P", "NONE"), ("F_C", "NONE"), ("F_PC", "F_C"), ("F_PC", "F_P")):
            lo, hi, pr = cluster_boot_delta(ev.y.values, S[a], S[b], ev.applicantCd.values, B=300, seed=c)
            o["delta"][f"{a}-{b}"] = {"d": o["arms"][a]["auc"] - o["arms"][b]["auc"], "ci": [lo, hi], "p_gt0": pr}
            log(c, "Y_" + D, a, "-", b, o["delta"][f"{a}-{b}"])
        out["Y_" + D] = o
    json.dump(out, open(os.path.join(RES, f"C_c{c}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    for c in map(int, sys.argv[1:]):
        run(c)
