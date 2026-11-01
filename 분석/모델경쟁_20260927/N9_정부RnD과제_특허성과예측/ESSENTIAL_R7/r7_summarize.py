# -*- coding: utf-8 -*-
"""results_dev.csv → 변형별 평균 AUC·skill, FULL 대비 차이, 하락 원점 수 → summary_dev.csv (트랙 공용)"""
import os, sys
import pandas as pd, numpy as np
fp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_dev.csv")
r = pd.read_csv(fp)
skcol = "skill"
full = r[r.variant == "FULL"].set_index("origin")
rows = []
for nm, g in r.groupby("variant", sort=False):
    g = g.set_index("origin"); fu = full.loc[g.index]          # 같은 원점끼리 비교(일부 원점만 돈 변형 대비)
    d_auc = g.auc - fu.auc; d_sk = g[skcol] - fu[skcol]
    rows.append(dict(variant=nm, n_origins=len(g), mean_auc=g.auc.mean(), mean_skill=g[skcol].mean(), d_auc=d_auc.mean(), d_skill=d_sk.mean(),
                     n_drop=int((d_auc < 0).sum()), min_d_auc=d_auc.min(), max_d_auc=d_auc.max(),
                     skill_retained=g[skcol].mean() / fu[skcol].mean(), full_auc_same_origins=fu.auc.mean(), full_skill_same_origins=fu[skcol].mean(),
                     **({"mean_top1": g.top1.mean()} if "top1" in g else {}), **({"mean_hit100": g.hit100.mean()} if "hit100" in g else {})))
S = pd.DataFrame(rows)
out = fp.replace("results_", "summary_")
S.to_csv(out, index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(S.round(4).to_string(index=False))
