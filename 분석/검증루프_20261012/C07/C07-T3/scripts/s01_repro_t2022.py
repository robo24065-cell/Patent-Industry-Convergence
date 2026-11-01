# -*- coding: utf-8 -*-
"""C07-T3 s01: 재현 검증 R1 — 공식 final()과 같은 학습(t0 2008–2020)·같은 시험 피처(t0=2022)로 champ·M0·SB를 다시 적합해
final_preds_r6.parquet의 p_champ·p_M0·p_SB와 대조. 시험 레이블(2023–24)은 읽지 않는다(no_label). → ../cache/s01_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c07lib import *  # noqa
t = time.time()
TR = load_dev_panel_like_official()
TR = TR[(TR.t0 >= SPEC["final"]["train_t0"][0]) & (TR.t0 <= SPEC["final"]["train_t0"][1])].reset_index(drop=True)
print("TR", len(TR), int(TR.y.sum()), rss(), round(time.time() - t), flush=True)
R = L.extract(L.SEAL); X = L.prep(R); del R
FT = L.family_tokens(X["pt"]); EXTU = L.external_unit(); KI = L.ksic_industry()
TE = L.to32(L.build_t0(2022, X, FT, EXTU, KI, no_label=True)).reset_index(drop=True)
del X
print("TE", len(TE), rss(), round(time.time() - t), flush=True)
FP = pd.read_parquet(os.path.join(L.HERE, "final_preds_r6.parquet"))
out = dict(n_TE=len(TE), n_final=len(FP), same_set=bool(set(TE.rgst_no) == set(FP.rgst_no)))
FPi = FP.set_index("rgst_no")
for k, (feats, model) in MODELS.items():
    p = fit_score(TR, TE, feats, model)
    ref = FPi.loc[TE.rgst_no.values, f"p_{k}"].values
    d = np.abs(p - ref)
    out[k] = dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), n_diff_gt_1e5=int((d > 1e-5).sum()))
    print(k, out[k], rss(), round(time.time() - t), flush=True)
out["R1_pass"] = bool(out["same_set"] and all(out[k]["max_abs_diff"] <= 1e-5 for k in MODELS))
out["peak_rss_gb"] = rss(); out["sec"] = round(time.time() - t)
json.dump(out, open(os.path.join(W, "cache", "s01_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
