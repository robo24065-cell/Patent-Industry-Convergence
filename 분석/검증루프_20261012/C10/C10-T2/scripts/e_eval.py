# -*- coding: utf-8 -*-
r"""[0][1][3] 평가 — 원본 TXT_R9 c_eval.py 복사·수정. 학습 seed 쌍(p → random_state 2p, 2p+1)·음성 표집 seed(k) 변동, 새 팔(Z) 지원.
사용: python -B e_eval.py <c> <k> <pairs: 0,1> <arms: P50,FREE50,TXT50> [z]
출력: cache/pred_c<c>_k<k>_<arms>_p<pairs>.parquet (열 <arm>_p<p>, float32), res/auc/auc_c<c>_k<k>_<arms>_p<pairs>.json"""
import sys
sys.dont_write_bytecode = True
from c10_common import *
import xgboost as xgb

c, k = int(sys.argv[1]), int(sys.argv[2])
pairs = [int(x) for x in sys.argv[3].split(",")]
arms = sys.argv[4].split(",")
z = len(sys.argv) > 5 and sys.argv[5] == "z"
tag = f"c{c}_k{k}_{'-'.join(arms)}_p{''.join(map(str, pairs))}"
os.makedirs(os.path.join(RES, "auc"), exist_ok=True)
tr = train_set(c, k, z)
ev = load(c, z); ev["y"] = ev.first_ky.notna().astype(np.int8)
far = (ev.p_cls == 0).values
out = {"c": c, "k": k, "neg_seed_base": (k + 1) * 1000, "pairs": pairs, "n_train": len(tr), "pos_train": int(tr.y.sum()),
       "n_eval": len(ev), "pos_eval": int(ev.y.sum()), "xgb": xgb.__version__, "res": {}}
pred = {}
for arm in arms:
    cols = ARMS[arm]
    for p in pairs:
        s = np.zeros(len(ev))
        indiv = []
        for seed in (2 * p, 2 * p + 1):
            m = xgb.XGBClassifier(random_state=seed, **PRM)
            m.fit(tr[cols], tr.y, sample_weight=tr.w)
            pp = m.predict_proba(ev[cols])[:, 1]
            indiv.append(auc_w(ev.y, pp))
            s += pp / 2
        pred[f"{arm}_p{p}"] = s.astype(np.float32)
        r = {"auc": auc_w(ev.y, s), "auc_far": auc_w(ev.y[far], s[far]), "top1000": topk_hit(ev.y, s, 1000), "auc_single": indiv}
        out["res"][f"{arm}_p{p}"] = r
        log(tag, arm, p, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in r.items() if kk != "auc_single"})
pd.DataFrame(pred).to_parquet(os.path.join(CACHE, f"pred_{tag}.parquet"), index=False)
json.dump(out, open(os.path.join(RES, "auc", f"auc_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", tag)
