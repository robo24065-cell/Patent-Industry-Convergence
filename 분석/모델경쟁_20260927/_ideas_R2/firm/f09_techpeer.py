# -*- coding: utf-8 -*-
"""아이디어 2 재시도: '기술 동료 모멘텀' — 특허 KSIC 구성 벡터의 코사인 유사도로 이은 동료 기업들의
t0 매출성장(DART) 가중평균이 초점 기업의 t0→t0+2 성장을 예측하는가. 동료 정보는 모두 t0 시점 값.
기준선·원점 동일(f08). 결과: iter_techpeer_<Y>.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalkit import walk, skill
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
B = pd.read_parquet(os.path.join(C, "panel12.parquet")); B = B[B.has_pat == 1].copy()
S = pd.read_parquet(os.path.join(C, "ksic_share.parquet"))
ks = sorted(S.ksic.unique())
feats = []
for t0, g in B.groupby("t0"):
    s = S[(S.t0 == t0) & S.corp_code.isin(g.corp_code)].pivot_table(index="corp_code", columns="ksic", values="share", fill_value=0)
    s = s.reindex(columns=ks, fill_value=0)
    V = s.values / np.maximum(np.linalg.norm(s.values, axis=1, keepdims=True), 1e-9)
    sim = V @ V.T; np.fill_diagonal(sim, 0)
    gg = g.set_index("corp_code").reindex(s.index)
    for col, nm in [("g_rev1", "peer_g1"), ("g_rev2", "peer_g2"), ("opm", "peer_opm")]:
        x = gg[col].values; ok = ~np.isnan(x)
        w = sim[:, ok]
        num = w @ x[ok]; den = w.sum(1)
        v = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        feats.append(pd.DataFrame({"corp_code": s.index, "t0": t0, "f": nm, "v": v}))
    # 상위 20 동료만
    x = gg["g_rev1"].values; ok = ~np.isnan(x)
    w = np.where(ok[None, :], sim, 0)
    thr = -np.sort(-w, axis=1)[:, min(19, w.shape[1] - 1)][:, None]
    w20 = np.where(w >= thr, w, 0)
    v = (w20 @ np.nan_to_num(x)) / np.maximum(w20.sum(1), 1e-9)
    feats.append(pd.DataFrame({"corp_code": s.index, "t0": t0, "f": "peer20_g1", "v": np.where(w20.sum(1) > 0, v, np.nan)}))
Fp = pd.concat(feats).pivot_table(index=["corp_code", "t0"], columns="f", values="v").reset_index()
B = B.merge(Fp, on=["corp_code", "t0"], how="left")
B.to_parquet(os.path.join(C, "panel12_peer.parquet"), index=False)
DEV = list(range(2015, 2022)); H = 2
for Y in ["y_hg", "y_dist"]:
    base_f = {"y_hg": ["log_rev", "g_rev2", "opm", "log_assets", "firm_age"],
              "y_dist": ["opm", "roa", "lev", "eq_cap", "log_assets", "op_neg_m1"]}[Y]
    ab, _ = walk(B, Y, base_f, DEV, H, "logit")
    rows = [{"exp": "B0", "mean_auc": round(np.mean(list(ab.values())), 4)}]
    for nm, f in [("peer_g1", ["peer_g1"]), ("peer_g2", ["peer_g2"]), ("peer20_g1", ["peer20_g1"]), ("peer_opm", ["peer_opm"]),
                  ("peer_all", ["peer_g1", "peer_g2", "peer_opm"])]:
        am, _ = walk(B, Y, base_f + f, DEV, H, "logit")
        rows.append({"exp": nm, "mean_auc": round(np.mean(list(am.values())), 4), "skill": round(np.mean(list(skill(am, ab).values())), 4),
                     "win": round(np.mean([am[v] > ab[v] for v in DEV]), 2), **{f"auc_{v}": round(am[v], 3) for v in DEV}})
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, f"iter_techpeer_{Y}.csv"), index=False)
    print(Y); print(R.to_string())
