# -*- coding: utf-8 -*-
"""T4 5개국 제거 시험 = R2 E23(동결 챔피언 E03에서 제공 5개국 패밀리 7개 제거, 같은 시드·프레임)의 캐시 예측을 읽어 다시 채점(쓰기 없음)."""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.join(os.path.dirname(HERE), "R2")
sys.path.insert(0, R2); os.environ["T4R2_MODE"] = "dev"
import numpy as np, pandas as pd, hashlib
import r2lib as R
from sklearn.metrics import roc_auc_score
out = {}
for e in ["E03", "E23"]:
    s = json.load(open(os.path.join(R2, "specs", f"{e}.json"), encoding="utf-8"))["spec"]
    res = {}
    for o, _, _ in R.ORIGINS:
        d = os.path.join(R2, "runs", o, "pred")
        y = pd.read_parquet(os.path.join(R2, "runs", o, "ev.parquet"), columns=["y"]).y.values
        ws = s["ensemble"]["weights"]
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        z = sum(w * lg(np.load(os.path.join(d, R.comp_key(s, k) + ".npy"))) for k, w in ws.items() if w) / sum(w for w in ws.values() if w)
        bkey = "base_" + hashlib.md5(json.dumps(s["baseline"], sort_keys=True).encode()).hexdigest()[:10]
        a = roc_auc_score(y, z); ab = roc_auc_score(y, np.load(os.path.join(d, bkey + ".npy")))
        res[o] = dict(auc=float(a), auc_base=float(ab), skill=float((a - ab) / (1 - ab)))
    out[e] = dict(by_origin=res, mean_auc=float(np.mean([r["auc"] for r in res.values()])), mean_skill=float(np.mean([r["skill"] for r in res.values()])))
out["d_skill_E23_vs_E03"] = out["E23"]["mean_skill"] - out["E03"]["mean_skill"]
out["n_origins_worse"] = int(sum(out["E23"]["by_origin"][o]["skill"] < out["E03"]["by_origin"][o]["skill"] for o in out["E03"]["by_origin"]))
json.dump(out, open(os.path.join(HERE, "res_R_FAM_E23check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
