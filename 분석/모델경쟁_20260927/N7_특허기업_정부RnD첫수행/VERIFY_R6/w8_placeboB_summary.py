# -*- coding: utf-8 -*-
r"""위약 B 청크(cache/placeboB_*.csv) 집계 → w8_placeboB.json. 관측값은 공식 동결 skill 0.2856(orig 패널 재현과 같음)."""
import os, glob, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
OBS = json.load(open(os.path.join(HERE, "w2_fit_orig.json"), encoding="utf-8"))
obs_sk, obs_c, obs_0 = OBS["skill"]["M0"], OBS["auc"]["champ"], OBS["auc"]["M0"]
R = {}
for s in ["t0nps", "global", "t0"]:
    fs = glob.glob(os.path.join(HERE, "cache", f"placeboB_{s}_*.csv"))
    if not fs:
        continue
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("rep")
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    R[s] = dict(n_rep=len(d), skill_mean=float(d.skill.mean()), skill_95=q(d.skill), skill_ge_obs=int((d.skill >= obs_sk).sum()),
                auc_champ_mean=float(d.auc_champ.mean()), auc_champ_95=q(d.auc_champ), auc_champ_ge_obs=int((d.auc_champ >= obs_c).sum()),
                auc_M0_mean=float(d.auc_M0.mean()), auc_M0_95=q(d.auc_M0), auc_M0_ge_obs=int((d.auc_M0 >= obs_0).sum()))
    print(s, R[s])
R["observed"] = dict(skill=obs_sk, auc_champ=obs_c, auc_M0=obs_0)
json.dump(R, open(os.path.join(HERE, "w8_placeboB.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
