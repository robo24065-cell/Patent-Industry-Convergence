# -*- coding: utf-8 -*-
"""재시도 감사 + 진단(채택 판단 무관). 이전 시도가 만든 K1·K2 패널의 봉인 준수 확인과 '쉬운 과제' 점검.
1) 봉인: 패널 t0 ≤ 2020, 양성의 상대 관청 첫 공개연도 ≤ 2022, 캐시 토큰 연도 < 2023.
2) 기계적 쉬움: WO(PCT) 공개가 t0까지 보인 단위의 양성 비중·양성률, wo_vis 단독 AUC, wo_vis 층별 챔피언/M0/SB AUC."""
import os, numpy as np, pandas as pd
from r6_common import CACHE, OUT, auc, skill

rows = []
for task, oy in (("K1", "fy"), ("K2", "ky")):
    cols = ["appl_no", "cc", "t0", "y", "wo_vis", "lag_m", "reg_first", "at_comp", "at_indiv", oy]
    D = pd.read_parquet(os.path.join(CACHE, f"{task}_panel.parquet"), columns=cols)
    assert D.t0.max() <= 2020 and D.loc[D.y == 1, oy].max() <= 2022
    print(task, "봉인 OK: t0 max", D.t0.max(), "양성 상대공개 max", D.loc[D.y == 1, oy].max(), "n", len(D))
    P = pd.read_parquet(os.path.join(CACHE, f"preds_{task}_xgb.parquet"))
    M = P.merge(D[["appl_no", "cc", "t0", "wo_vis", "lag_m"]], on=["appl_no", "cc", "t0"], how="left")
    assert M.wo_vis.notna().all() and len(M) == len(P)
    for v, g in M.groupby("t0"):
        r = dict(task=task, v=v, n=len(g), pos=int(g.y.sum()), rate=g.y.mean(),
                 wo_share_units=g.wo_vis.mean(), pos_share_wo=g.loc[g.y == 1, "wo_vis"].mean(),
                 rate_wo1=g.loc[g.wo_vis == 1, "y"].mean(), rate_wo0=g.loc[g.wo_vis == 0, "y"].mean(),
                 auc_wo_only=auc(g.y, g.wo_vis), auc_wo_lag=auc(g.y, g.wo_vis * 1000 - g.lag_m.fillna(99)),
                 auc_M0=auc(g.y, g.p_M0), auc_champ=auc(g.y, g.p_champ), auc_SB=auc(g.y, g.p_SB))
        for s in (1, 0):
            h = g[g.wo_vis == s]
            r[f"n_wo{s}"] = len(h); r[f"pos_wo{s}"] = int(h.y.sum())
            for c in ("M0", "champ", "SB"):
                r[f"auc_{c}_wo{s}"] = auc(h.y, h[f"p_{c}"])
            r[f"skill_wo{s}"] = skill(r[f"auc_champ_wo{s}"], r[f"auc_M0_wo{s}"])
            r[f"skill_vs_SB_wo{s}"] = skill(r[f"auc_champ_wo{s}"], r[f"auc_SB_wo{s}"])
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "R6_diag_mechanical.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
show = ["task", "v", "pos", "rate", "wo_share_units", "pos_share_wo", "rate_wo1", "rate_wo0", "auc_wo_only", "auc_wo_lag", "auc_M0", "auc_champ",
        "auc_M0_wo1", "auc_champ_wo1", "skill_wo1", "skill_vs_SB_wo1", "auc_M0_wo0", "auc_champ_wo0", "skill_wo0", "skill_vs_SB_wo0"]
print(R[show].round(3).to_string())
print(R.groupby("task")[[c for c in show if c not in ("task", "v")]].mean().round(3).T.to_string())
# 토큰 봉인
for cc in ["KR", "US", "JP", "EP"]:
    t = pd.read_parquet(os.path.join(CACHE, f"tok_{cc}.parquet"), columns=["y_us", "y_jp", "y_ep", "y_wo", "y_kr"])
    assert (t.max() < 2023).all(), (cc, t.max())
print("토큰 연도 봉인 OK")
