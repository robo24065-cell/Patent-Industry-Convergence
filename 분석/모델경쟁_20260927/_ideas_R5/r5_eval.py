# -*- coding: utf-8 -*-
"""_ideas_R5 사전 등록 순차 실험. 사용: python r5_eval.py J2 xgb
묶음·순서·하이퍼파라미터는 실행 전에 아래 SPEC에 고정(결과 보고 바꾸지 않음)."""
import sys, time, json
import numpy as np, pandas as pd
from r5_common import *

from r5_eval_spec import SPEC

idea, model = sys.argv[1], sys.argv[2]
S = SPEC[idea]
t = time.time()
D = pd.read_parquet(f"{CACHE}/{S['panel']}.parquet")
if "filter" in S:
    D = D.query(S["filter"])
D = D.reset_index(drop=True)
R, O, P = run_steps(D, "y", S["dev"], S["base"], S["steps"], gap=S["gap"], group_col=S.get("group"), models=(model,),
                    neg_frac=S.get("neg_frac"), n_boot=S.get("n_boot", 100), min_train=S.get("min_train"))
R.to_csv(f"{OUT}/experiments_{idea}_{model}.csv", index=False, encoding="utf-8-sig")
O.to_csv(f"{OUT}/pilot_by_origin_{idea}_{model}.csv", index=False, encoding="utf-8-sig")
preds, champ, feats = P[model]
rows = []
for v in S["dev"]:
    te = D[D.t0 == v]
    df = pd.DataFrame({"t0": v, "y": te.y.values, "p_M0": preds[("M0", v)], "p_champ": preds[(champ, v)]})
    for name, _ in S["steps"]:
        if (name, v) in preds:
            df[f"p_{name}"] = preds[(name, v)]
    if S.get("group"):
        df["g"] = te[S["group"]].values
    rows.append(df)
pd.concat(rows).to_parquet(f"{CACHE}/preds_{idea}_{model}.parquet", index=False)
json.dump(dict(champ=champ, feats=feats), open(f"{CACHE}/champ_{idea}_{model}.json", "w", encoding="utf-8"), ensure_ascii=False)
print(R.to_string()); print(O.round(4).to_string()); print("sec", round(time.time() - t))
