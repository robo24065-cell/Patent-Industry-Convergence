# -*- coding: utf-8 -*-
"""묶음 결과 취합 → seed_bundles.csv, variance_decomp.csv, verdicts.json (PREREG §4 기준 그대로)."""
import os, sys, json, glob
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from c13util import *
import numpy as np, pandas as pd

OFFICIAL_CLASS = {"N4": "강화 우위", "N6": "강화 우위", "N7": "강화 우위", "N8": "강화 우위", "N9": "강화 우위", "T4": "조건부"}
KTARGET = {"N4": 10, "N6": 10, "N7": 10, "N8": 9, "N9": 9, "T4": 8}
rows = []
for t in ["n4", "n6", "n7", "n8", "n9", "t4"]:
    for f in glob.glob(os.path.join(CACHE, t, "b*.json")):
        J = json.load(open(f, encoding="utf-8"))
        J["seeds"] = json.dumps(J["seeds"], ensure_ascii=False)
        for k in list(J):
            if isinstance(J[k], (dict, list)):
                J[k] = json.dumps(J[k], ensure_ascii=False)
        rows.append(J)
D = pd.DataFrame(rows).sort_values(["track", "bundle"]).reset_index(drop=True)
D["ci_lo_gt0"] = D.ci_lo > 0
D["strong_class"] = np.where(D.strong_ci_lo > 0, "강화 우위", "조건부")
D["official_bundle"] = D.bundle == 0
first = ["track", "bundle", "official_bundle", "seeds", "n", "pos", "auc_champ", "auc_M0", "skill", "ci_lo", "ci_hi", "ci_lo_gt0", "strong_name", "auc_strong",
         "skill_vs_strong", "strong_ci_lo", "strong_ci_hi", "strong_class", "top1_k", "top1_hits", "top1_prec", "top1_capture"]
D = D[first + [c for c in D.columns if c not in first]]
D.to_csv(os.path.join(W, "seed_bundles.csv"), index=False, encoding="utf-8-sig")
V, VER = [], {}
for t, g in D.groupby("track"):
    o = g[g.bundle == 0].iloc[0]; K = len(g)
    se = (o.ci_hi - o.ci_lo) / 3.92; se_s = (o.strong_ci_hi - o.strong_ci_lo) / 3.92
    share = float(g.ci_lo_gt0.mean())
    v_m0 = "판정 보류(K<6)" if K < 6 else ("견고" if share >= 0.9 else ("경계" if share >= 0.5 else "취약"))
    cls = sorted(g.strong_class.unique())
    v_st = "분류 불안정" if len(cls) > 1 else ("분류 유지" if cls[0] == OFFICIAL_CLASS[t] else f"분류 바뀜({cls[0]})")
    V.append(dict(track=t, K=K, K_target=KTARGET[t], skill_official=o.skill, skill_mean=g.skill.mean(), skill_sd=g.skill.std(ddof=1), skill_min=g.skill.min(), skill_max=g.skill.max(),
                  ci_lo_official=o.ci_lo, ci_lo_min=g.ci_lo.min(), ci_lo_max=g.ci_lo.max(), share_ci_lo_gt0=share, boot_se_official=se,
                  ratio_seed_sd_to_boot_se=g.skill.std(ddof=1) / se,
                  strong_official=o.skill_vs_strong, strong_mean=g.skill_vs_strong.mean(), strong_sd=g.skill_vs_strong.std(ddof=1), strong_min=g.skill_vs_strong.min(),
                  strong_max=g.skill_vs_strong.max(), strong_ci_lo_official=o.strong_ci_lo, strong_ci_lo_min=g.strong_ci_lo.min(), strong_ci_lo_max=g.strong_ci_lo.max(),
                  share_strong_ci_lo_gt0=float((g.strong_ci_lo > 0).mean()), strong_boot_se_official=se_s, ratio_strong_sd_to_boot_se=g.skill_vs_strong.std(ddof=1) / se_s,
                  top1_capture_official=o.top1_capture, top1_capture_min=g.top1_capture.min(), top1_capture_max=g.top1_capture.max(),
                  top1_prec_official=o.top1_prec, top1_prec_min=g.top1_prec.min(), top1_prec_max=g.top1_prec.max(),
                  verdict_M0=v_m0, verdict_strong=v_st, official_rank_skill=int((g.skill > o.skill).sum() + 1)))
    VER[t] = dict(K=K, verdict_M0=v_m0, verdict_strong=v_st, share_ci_lo_gt0=share)
pd.DataFrame(V).to_csv(os.path.join(W, "variance_decomp.csv"), index=False, encoding="utf-8-sig")
dump(VER, os.path.join(W, "verdicts.json"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(pd.DataFrame(V)[["track", "K", "skill_official", "skill_mean", "skill_sd", "skill_min", "skill_max", "ci_lo_min", "ci_lo_max", "share_ci_lo_gt0", "boot_se_official",
                       "ratio_seed_sd_to_boot_se", "strong_official", "strong_min", "strong_max", "strong_ci_lo_min", "strong_ci_lo_max", "verdict_M0", "verdict_strong", "official_rank_skill"]].round(4).to_string())
