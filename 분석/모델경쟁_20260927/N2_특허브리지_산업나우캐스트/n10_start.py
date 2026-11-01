# -*- coding: utf-8 -*-
"""N2 1단계: 고정 기준선·참고 기준선·출발 챔피언(스냅샷판 vs 시점판) 채점 → R2/experiments.csv 첫 줄들."""
import os, json
import numpy as np, pandas as pd
import n_lib as L

Y = L.panel()
base = L.baseline()
mb = L.metrics(base, base)
L.log_row("B_M1", "", "고정 기준선 M1: OLS g ~ 광공업 출하지수 3자리 t년 성장(학습 2014..t−1)", mb, "BASE")


def direct(col):
    P = {}
    for t in L.DEV:
        te = Y[Y.t == t].dropna(subset=["g"])
        P[t] = pd.DataFrame({"ksic": te.ksic.values, "y": te.g.values, "p": te[col].fillna(0).values if col else 0.0})
    return P


for eid, col, ch in [("B0_zero", None, "참고: 0 성장"), ("B1_lastyear", "g_lag", "참고: 작년 조사 성장"), ("B2_idxdirect", "g_idx", "참고: 지수 성장 직접 대입")]:
    L.log_row(eid, "", ch, L.metrics(direct(col), base), "REF")

S0 = {"feats": ["g_idx", "gd_f"], "model": "ols", "fin": "first", "w": "snap"}
P = L.evaluate(S0); m0 = L.metrics(P, base)
L.log_row("S0_snap", "B_M1", "파일럿 M3 재현(스냅샷판): +gd(파일럿 재무 최초보고값, right_list 스냅샷 분모 w_snap)", m0, "REF",
          "파일럿 보고 0.0477과 비교용")
P0 = {"feats": ["g_idx", "gd_f"], "model": "ols", "fin": "first", "w": "pit"}
P = L.evaluate(P0); m1 = L.metrics(P, base)
win = float(np.mean([m1["mae"][t] < m0["mae"][t] for t in L.DEV]))
L.log_row("C0_pit", "B_M1", "출발 챔피언(시점판): +gd, 가중 w_pit(T=t−1년 말 보유·생존, 권리이전·소멸 이력 반영)", m1, "CHAMP",
          f"스냅샷판 대비 skill {m1['mean_skill'] - m0['mean_skill']:+.4f}, 원점 승률 {win:.2f}", m0)
L.save_state("C0_pit", P0)
ci, pp = L.boot_ci(P, base)
print("C0_pit skill", round(m1["mean_skill"], 4), "CI", ci.round(4), "P>0", pp)
print("S0_snap skill", round(m0["mean_skill"], 4))
pd.set_option("display.width", 250)
print(pd.read_csv(L.LOG, encoding="utf-8-sig")[["exp_id", "mean_mae", "mean_skill", "rho", "dir", "medae", "decision"]].to_string())
