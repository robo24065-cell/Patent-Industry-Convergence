# -*- coding: utf-8 -*-
"""사후 점검(채택 판단 무관) — K1P·K2P 챔피언 예측 파일로만 계산. 사용: python r6_robust.py
1) 위약: 원점×국가 층 안에서 시험 레이블을 섞었을 때 챔피언 vs M0 skill(200회) — 0 근처여야 정상.
2) 대형 출원인 제외: 원점별 단위 수 상위 5개 출원인 제외 후 skill(vs M0, vs SB).
3) 하위집단: K2P 출신 관청(US/JP/EP), K1P·K2P 출원인 유형(기업/대학/연구기관/개인).
4) KSIC 집계(트렌드 산출물 검증): 원점별 KSIC 진입률 예측(확률 평균) vs 실제, 스피어만. 비교: 직전 가용 원점(v−2)의 KSIC 실제 진입률."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from r6_common import CACHE, OUT, auc, skill
sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.default_rng(7)
rows = []
for task, model in (("K1P", "xgb"), ("K1P", "logit"), ("K2P", "xgb"), ("K2P", "logit")):
    f = os.path.join(CACHE, f"preds_{task}_{model}.parquet")
    if not os.path.exists(f):
        continue
    P = pd.read_parquet(f)
    D = pd.read_parquet(os.path.join(CACHE, f"{task}_panel.parquet"), columns=["appl_no", "cc", "t0", "atype"])
    P = P.merge(D, on=["appl_no", "cc", "t0"], how="left")
    assert P.atype.notna().all()
    full = pd.read_parquet(os.path.join(CACHE, f"{task}_panel.parquet"), columns=["t0", "ksic", "y"])
    for v, g in P.groupby("t0"):
        y = g.y.values
        base = dict(task=task, model=model, v=v)
        a_c, a_0, a_s = auc(y, g.p_champ), auc(y, g.p_M0), auc(y, g.p_SB)
        # 1 위약
        strata = g.cc.values
        sk = []
        for _ in range(200):
            ys = y.copy()
            for s in np.unique(strata):
                m = strata == s; ys[m] = rng.permutation(ys[m])
            sk.append(skill(auc(ys, g.p_champ), auc(ys, g.p_M0)))
        rows.append(dict(base, check="위약_층내섞기200", skill=np.mean(sk), lo=np.percentile(sk, 2.5), hi=np.percentile(sk, 97.5),
                         obs=skill(a_c, a_0), exceed=np.mean(np.array(sk) >= skill(a_c, a_0))))
        # 2 대형 출원인 제외
        top = g.akey.value_counts().index[:5]
        h = g[~g.akey.isin(top)]
        rows.append(dict(base, check="상위5출원인제외", n=len(h), pos=int(h.y.sum()), auc_champ=auc(h.y, h.p_champ), auc_M0=auc(h.y, h.p_M0),
                         skill=skill(auc(h.y, h.p_champ), auc(h.y, h.p_M0)), skill_vs_SB=skill(auc(h.y, h.p_champ), auc(h.y, h.p_SB)),
                         top5_share=1 - len(h) / len(g)))
        # 3 하위집단
        subs = [("유형_" + nm, g.atype == k) for k, nm in enumerate(["개인", "기업", "연구기관", "대학"])]
        if task == "K2P":
            subs += [("관청_" + c, g.cc == c) for c in ("US", "JP", "EP")]
        for nm, m in subs:
            h = g[m.values]
            if h.y.nunique() < 2 or h.y.sum() < 10:
                continue
            rows.append(dict(base, check=nm, n=len(h), pos=int(h.y.sum()), rate=h.y.mean(), auc_champ=auc(h.y, h.p_champ), auc_M0=auc(h.y, h.p_M0),
                             auc_SB=auc(h.y, h.p_SB), skill=skill(auc(h.y, h.p_champ), auc(h.y, h.p_M0)), skill_vs_SB=skill(auc(h.y, h.p_champ), auc(h.y, h.p_SB))))
        # 4 KSIC 집계
        k = g.groupby("ksic").agg(n=("y", "size"), act=("y", "mean"), pc=("p_champ", "mean"), p0=("p_M0", "mean"))
        prev = full[full.t0 == v - 2].groupby("ksic").y.mean().rename("naive")
        k = k.join(prev, how="left"); k = k[k.n >= 20]
        kk = k.dropna(subset=["naive"])
        rows.append(dict(base, check="KSIC진입률_스피어만", n=len(k), rho_champ=spearmanr(k.act, k.pc)[0], rho_M0=spearmanr(k.act, k.p0)[0],
                         rho_naive=spearmanr(kk.act, kk.naive)[0], mae_champ=(k.act - k.pc).abs().mean(), mae_naive=(kk.act - kk.naive).abs().mean()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "R6_robust.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
num = [c for c in R.columns if c not in ("task", "model", "check", "v")]
S = R.groupby(["task", "model", "check"])[num].mean()
S["wins"] = R.assign(w=R.skill > 0).groupby(["task", "model", "check"]).w.mean()
S["wins_SB"] = R.assign(w=R.get("skill_vs_SB", pd.Series(np.nan, index=R.index)) > 0).groupby(["task", "model", "check"]).w.mean()
print(S.round(3).dropna(axis=1, how="all").to_string())
S.round(4).to_csv(os.path.join(OUT, "R6_robust_summary.csv"), encoding="utf-8-sig")
