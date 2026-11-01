# -*- coding: utf-8 -*-
"""P0: 묶음 0(공식 시드) 결과를 동결 결과 JSON·동결 예측과 대조. 기준 차 ≤ 1e−9. → p0_result.json (C13-T2 루트)"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from c13util import *
import numpy as np, pandas as pd

J = lambda *p: json.load(open(os.path.join(MC, *p), encoding="utf-8"))
B = lambda t: json.load(open(os.path.join(CACHE, t, "b0.json"), encoding="utf-8"))
out = {}


def cmp(trk, pairs, preds=None):
    d = {}
    for nm, mine, off in pairs:
        mine = mine if isinstance(mine, list) else [mine]; off = off if isinstance(off, list) else [off]
        d[nm] = dict(mine=mine, official=off, max_abs_diff=float(max(abs(a - b) for a, b in zip(mine, off))))
    r = dict(items=d, max_abs_diff=float(max(v["max_abs_diff"] for v in d.values())))
    if preds:
        r["pred_max_abs_diff"] = preds
    r["pass"] = bool(r["max_abs_diff"] <= 1e-9 and (not preds or max(preds.values()) <= 1e-9))
    out[trk] = r


def pdiff(trk, frozen, mapping, key=None):
    me = pd.read_parquet(os.path.join(CACHE, trk.lower(), "preds_b0.parquet"))
    if key:
        assert (me[key[0]].astype(str).values == frozen[key[1]].astype(str).values).all()
    return {k: float(np.abs(me[a].values - frozen[b].values).max()) for k, (a, b) in mapping.items()}


if os.path.exists(os.path.join(CACHE, "n4", "b0.json")):
    o = J("N4_특허행태_기업폐업조기경보", "final_result_r4.json")["results"][0]; m = B("n4")
    fz = pd.read_parquet(os.path.join(MC, "N4_특허행태_기업폐업조기경보", "final_test_pred_r4.parquet"))
    cmp("N4", [("skill", m["skill"], o["skill_vs_M0"]), ("ci", [m["ci_lo"], m["ci_hi"]], o["skill_vs_M0_ci95"]),
               ("strong_skill(M0ens)", m["skill_vs_strong"], o["skill_vs_M0ens"]), ("strong_ci", [m["strong_ci_lo"], m["strong_ci_hi"]], o["skill_vs_M0ens_ci95"]),
               ("auc", [m["auc_champ"], m["auc_M0"], m["auc_strong"]], [o["auc"]["champ"], o["auc"]["M0_xgb"], o["auc"]["M0_ens"]]),
               ("top1_prec", m["top1_prec_official_rule"], o["top1pct_precision"]["champ"])],
        pdiff("N4", fz, {"champ": ("p_champ", "p_champ"), "M0": ("p_M0", "p_M0")}, ("bizno", "bizno")))
if os.path.exists(os.path.join(CACHE, "n6", "b0.json")):
    o = J("N6_정부RnD기업_첫특허전환", "final_result_r5.json"); m = B("n6")
    cmp("N6", [("skill", m["skill"], o["skill"]), ("ci", [m["ci_lo"], m["ci_hi"]], o["skill_ci95"]),
               ("strong_skill(M0ens)", m["skill_vs_strong"], o["secondary_vs_M0_ensemble"]["skill"]),
               ("strong_ci", [m["strong_ci_lo"], m["strong_ci_hi"]], o["secondary_vs_M0_ensemble"]["skill_ci95"]),
               ("auc", [m["auc_champ"], m["auc_M0"], m["auc_strong"]], [o["auc"], o["auc_M0"], o["secondary_vs_M0_ensemble"]["auc_ref"]]),
               ("top1_prec", m["top1_prec_official_rule"], o["top1"])], m.get("max_pred_diff_vs_frozen"))
if os.path.exists(os.path.join(CACHE, "n7", "b0.json")):
    o = J("N7_특허기업_정부RnD첫수행", "final_result_r5.json")["results"][0]; m = B("n7")
    fz = pd.read_parquet(os.path.join(MC, "N7_특허기업_정부RnD첫수행", "final_test_pred_r5.parquet"))
    cmp("N7", [("skill", m["skill"], o["skill_vs_M0"]), ("ci", [m["ci_lo"], m["ci_hi"]], o["skill_vs_M0_ci95"]),
               ("strong_skill(SB)", m["skill_vs_strong"], o["skill_vs_SB"]), ("strong_ci", [m["strong_ci_lo"], m["strong_ci_hi"]], o["skill_vs_SB_ci95"]),
               ("M0ens", [m["skill_vs_M0ens"], m["M0ens_ci_lo"], m["M0ens_ci_hi"]], [o["skill_vs_M0ens"]] + o["skill_vs_M0ens_ci95"]),
               ("auc", [m["auc_champ"], m["auc_M0"], m["auc_strong"]], [o["auc"]["champ"], o["auc"]["M0"], o["auc"]["SB"]]),
               ("top1", [m["top1_prec_official_rule"], m["top1_capture_official_rule"]], [o["topk"]["champ"]["top1pct"]["precision"], o["topk"]["champ"]["top1pct"]["recall"]])],
        pdiff("N7", fz, {"champ": ("p_champ", "p_champ"), "M0": ("p_M0", "p_M0"), "SB": ("p_SB", "p_SB")}, ("bizno", "bizno")))
if os.path.exists(os.path.join(CACHE, "n8", "b0.json")):
    o = J("N8_특허분쟁표적레이더", "final_result_r6.json"); m = B("n8")
    cmp("N8", [("skill", m["skill"], o["skill_vs_M0"]), ("ci", [m["ci_lo"], m["ci_hi"]], o["skill_vs_M0_ci"]),
               ("strong_skill(SB)", m["skill_vs_strong"], o["skill_vs_SB"]), ("strong_ci", [m["strong_ci_lo"], m["strong_ci_hi"]], o["skill_vs_SB_ci"]),
               ("auc", [m["auc_champ"], m["auc_M0"], m["auc_strong"]], [o["champ_auc"], o["M0_auc"], o["SB_auc"]]),
               ("top1_capture", [m["top1_capture"], float(m["top1_hits"]), float(m["hit100"])], [o["champ_top1pct_capture"], float(o["champ_top1pct_hits"]), float(o["champ_hit100"])])],
        m.get("max_pred_diff_vs_frozen"))
if os.path.exists(os.path.join(CACHE, "n9", "b0.json")):
    o = J("N9_정부RnD과제_특허성과예측", "final_result_r6.json"); m = B("n9")
    fz = pd.read_parquet(os.path.join(MC, "N9_정부RnD과제_특허성과예측", "final_preds_r6.parquet"))
    cmp("N9", [("skill", m["skill"], o["skill"]), ("ci", [m["ci_lo"], m["ci_hi"]], o["skill_ci95"]),
               ("strong_skill(strong_L2)", m["skill_vs_strong"], o["skill_vs_strong_L2"]), ("strong_ci", [m["strong_ci_lo"], m["strong_ci_hi"]], o["skill_vs_strong_L2_ci95"]),
               ("auc", [m["auc_champ"], m["auc_M0"], m["auc_strong"]], [o["auc"], o["auc_M0L2"], o["auc_strong_L2"]]),
               ("top1", [m["top1_prec_official_rule"], float(m["hit100"])], [o["top1"], float(o["hit100"])])],
        pdiff("N9", fz, {"champ": ("p_champ", "p_champ"), "M0_L2": ("p_M0_L2", "p_M0_L2"), "strong_L2": ("p_strong_L2", "p_strong_L2")}, ("org", "org")))
if os.path.exists(os.path.join(CACHE, "t4", "b0.json")):
    o = J("T4_특허포기예측", "R2", "final_result.json"); m = B("t4")
    vr = J("T4_특허포기예측", "VERIFY_R5", "v3_results.json") if os.path.exists(os.path.join(MC, "T4_특허포기예측", "VERIFY_R5", "v3_results.json")) else None
    pairs = [("skill", m["skill"], o["skill_auc"]), ("ci(4자리 반올림 저장)", [round(m["ci_lo"], 4), round(m["ci_hi"], 4)], o["bootstrap_95ci"]["skill"]),
             ("auc", [m["auc_champ"], m["auc_M0"]], [o["model"]["auc"], o["baseline"]["auc"]])]
    if vr:
        k13 = [k for k in vr["strong_baseline"] if "납부행동" in k][0]
        pairs += [("strong_skill(13변수)", m["skill_vs_strong"], vr["strong_baseline"][k13]["skill"]),
                  ("strong_ci", [m["strong_ci_lo"], m["strong_ci_hi"]], vr["strong_baseline"][k13]["ci95"])]
    cmp("T4", pairs, m.get("max_pred_diff_vs_frozen"))
dump(out, os.path.join(W, "p0_result.json"))
for k, v in out.items():
    print(k, "PASS" if v["pass"] else "FAIL", v["max_abs_diff"], v.get("pred_max_abs_diff"))
