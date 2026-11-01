# -*- coding: utf-8 -*-
"""C07-T3 s02b: future()와 같은 TR(panel_dev 전체 + t0 2021·2022)로 champ·M0·SB를 적합해 t0=2024 채점.
재현 검증 R2: 재적합 챔피언 = future_scores.p. 결과(2025년 이후 사건)는 읽지 않는다. → ../cache/s02b_scores_t2024.parquet(내부용), s02b_out.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c07lib import *  # noqa
t = time.time()
COLS = list(dict.fromkeys(["rgst_no", "t0", "y", "ksic", "owner_v2", "own_disp", "own_trial"] + NEED))
D = pd.read_parquet(f"{L.CACHE}/panel_dev.parquet", columns=COLS).reset_index(drop=True)
t0s = sorted(D.t0.unique().tolist())
T2 = pd.read_parquet(os.path.join(W, "cache", "s02a_tr2122.parquet"))
TR = pd.concat([D, T2[D.columns]], ignore_index=True); del D, T2
print("TR", len(TR), int(TR.y.sum()), t0s[0], t0s[-1], rss(), round(time.time() - t), flush=True)
TE = pd.read_parquet(os.path.join(W, "cache", "s02a_te2024.parquet"))
S = pd.read_parquet(os.path.join(L.HERE, "future_scores_r6.parquet"))
out = dict(n_TR=len(TR), TR_pos=int(TR.y.sum()), dev_t0=[int(t0s[0]), int(t0s[-1])], n_TE=len(TE),
           same_order=bool((TE.rgst_no.values == S.rgst_no.values).all()), same_set=bool(set(TE.rgst_no) == set(S.rgst_no)))
P = {}
for k, (feats, model) in MODELS.items():
    P[k] = fit_score(TR, TE, feats, model)
    print(k, rss(), round(time.time() - t), flush=True)
ref = S.set_index("rgst_no").loc[TE.rgst_no.values, "p"].values
d = np.abs(P["champ"] - ref)
out["champ_vs_future"] = dict(max_abs_diff=float(d.max()), n_diff_gt_1e5=int((d > 1e-5).sum()))
out["R2_pass"] = bool(out["same_set"] and d.max() <= 1e-5)
O = TE[["rgst_no", "ksic", "owner_v2", "own_disp", "own_trial", "H_pre", "H_2324only", "H_any"]].copy()
O["p_champ"] = ref; O["p_champ_refit"] = P["champ"]; O["p_M0"] = P["M0"]; O["p_SB"] = P["SB"]
O.to_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"), index=False)
out.update(sum_p={k: float(v.sum()) for k, v in P.items()}, peak_rss=rss(), sec=round(time.time() - t))
json.dump(out, open(os.path.join(W, "cache", "s02b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
