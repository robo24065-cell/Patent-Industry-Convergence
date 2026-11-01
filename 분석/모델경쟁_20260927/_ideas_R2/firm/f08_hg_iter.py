# -*- coding: utf-8 -*-
"""아이디어 2(고성장 기업) 정식화 재시도: 이론 기반 소수 피처를 기준선 로지스틱에 하나씩 추가(챔피언/도전자).
채택 = 평균 skill +0.003 이상 & 원점 60% 이상 우세. 개발 원점 v=2015..2021, 학습 t0 ≤ v−2.
대안 타깃: y_hg_rel (KSIC 노출도 동료 대비 초과성장 상위 25%) — 산업 공통 충격을 제거."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalkit import walk, skill
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
B = pd.read_parquet(os.path.join(C, "panel12.parquet")); B = B[B.has_pat == 1].copy()
B["log_alive"] = np.log1p(B.n_alive); B["log_reg3"] = np.log1p(B.n_reg3)
B["pat_int"] = np.log1p(B.n_reg3) - B.log_rev.where(B.log_rev > 0) / 3   # 매출 대비 특허 집약도(로그)
B["new_ksic_any"] = (B.n_newksic3 > 0).astype(float)
DEV = list(range(2015, 2022)); H = 2
Y = sys.argv[1] if len(sys.argv) > 1 else "y_hg"
if Y == "y_hg_rel":
    # 동료 성장: 같은 주력 KSIC 기업들의 평균 g2 (자기 제외) — 결과 구간 정보이므로 타깃 정의에만 사용
    g = B.groupby(["t0", "top_ksic"]).g2
    s, n = g.transform("sum"), g.transform("count")
    B["peer"] = np.where(n > 1, (s - B.g2) / (n - 1), np.nan)
    B["ex"] = B.g2 - B.peer
    B["y_hg_rel"] = np.nan
    for t, gg in B[B.ex.notna()].groupby("t0"):
        B.loc[gg.index, "y_hg_rel"] = (gg.ex >= gg.ex.quantile(0.75)).astype(float)
base_f = ["log_rev", "g_rev2", "opm", "log_assets", "firm_age"]
cands = [("reg_mom", ["reg_mom"]), ("log_reg3", ["log_reg3"]), ("log_alive", ["log_alive"]), ("pat_int", ["pat_int"]),
         ("newksic", ["new_ksic_any"]), ("lap_rate", ["lap_rate"]), ("tr_rate", ["tr_rate"]), ("hhi", ["hhi"]),
         ("x_pmom_foreign", ["x_pmom_foreign"]), ("x_kr_gap", ["x_kr_gap"]), ("x_g_ship", ["x_g_ship"]), ("x_g_imp", ["x_g_imp"]),
         ("x_pmom_CN", ["x_pmom_CN"]), ("x_va_per_emp", ["x_va_per_emp"]), ("fin_more", ["roa", "lev", "g_rev1", "re_assets"])]
champ = list(base_f)
ab, _ = walk(B, Y, base_f, DEV, H, "logit")
ac = dict(ab)
log = [{"exp": "B0", "change": "baseline", "mean_auc": np.mean(list(ab.values())), "d_skill": 0, "win": 0, "decision": "BASE"}]
for name, f in cands:
    am, _ = walk(B, Y, champ + f, DEV, H, "logit")
    sk_new = np.mean(list(skill(am, ab).values())); sk_ch = np.mean(list(skill(ac, ab).values()))
    win = np.mean([am[v] > ac[v] for v in DEV])
    dec = "ACCEPT" if (sk_new - sk_ch >= 0.003 and win >= 0.6) else "ROLLBACK"
    if dec == "ACCEPT":
        champ += f; ac = am
    log.append({"exp": name, "change": "+" + ",".join(f), "mean_auc": round(np.mean(list(am.values())), 4),
                "skill_vs_base": round(sk_new, 4), "d_skill": round(sk_new - sk_ch, 4), "win": round(win, 2), "decision": dec})
    print(log[-1], flush=True)
am, _ = walk(B, Y, champ, DEV, H, "xgb", n_estimators=200, max_depth=3)
log.append({"exp": "xgb_champ", "change": "xgb on champion feats", "mean_auc": round(np.mean(list(am.values())), 4),
            "skill_vs_base": round(np.mean(list(skill(am, ab).values())), 4), "decision": "INFO"})
L = pd.DataFrame(log); L.to_csv(os.path.join(HERE, f"iter_{Y}.csv"), index=False)
print(L.to_string()); print("champion:", champ, "per-origin", {v: round(ac[v], 4) for v in ac}, "base", {v: round(ab[v], 4) for v in ab})
