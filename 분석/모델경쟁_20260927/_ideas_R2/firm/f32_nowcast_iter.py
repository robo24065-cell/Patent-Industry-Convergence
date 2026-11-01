# -*- coding: utf-8 -*-
"""아이디어 1(N1 산업 나우캐스트) 챔피언/도전자 반복. 지표 = 원점(연도)별 50개 KSIC MAE, skill = 1 − MAE/MAE_기준선.
기준선(고정) = M1: OLS g ~ 광공업 출하지수 성장(3자리). 출발 챔피언 = M3(+DART 특허배분 성장, w_full).
채택 = 평균 skill +0.003 이상 & 원점 60% 이상 우세. 개발 t=2016..2023, 학습 2014 ≤ t' ≤ t−1. 2024(F)는 사용하지 않음."""
import os, sys, json, sqlite3
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.stats import spearmanr
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
Y = pd.read_parquet(os.path.join(C, "nowcast_panel.parquet"))
assert Y.t.max() <= 2023
DEV = list(range(2016, 2024))
# --- 추가 신호 ---
m = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str); m["hs6"] = m.hs6.str.zfill(6)
h = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str}); h["hs6"] = h.hs6.str.zfill(6)
im = h[h.partner == "WLD"].merge(m[["k5", "hs6"]].drop_duplicates(), on="hs6").groupby(["k5", "year"]).value.sum()
Y["g_imp1"] = [np.log((im.get((k, t), np.nan) + 1) / (im.get((k, t - 1), np.nan) + 1)) for k, t in zip(Y.ksic, Y.t)]
F = pd.read_parquet(os.path.join(C, "fin.parquet"))[["corp_code", "year", "rev"]]
W = pd.read_parquet(os.path.join(C, "alloc_w.parquet"))
fr = F.merge(F.assign(year=F.year + 1).rename(columns={"rev": "rev_prev"}), on=["corp_code", "year"])
fr = fr[(fr.rev > 0) & (fr.rev_prev > 0)]; fr["gi"] = np.log(fr.rev / fr.rev_prev).clip(-1, 1)
M = fr.merge(W, left_on=["corp_code", fr.year - 1], right_on=["corp_code", "t0"])
M["a"] = M.w_full * M.rev_prev
def wmed(x):
    x = x.sort_values("gi"); c = x.a.cumsum() / x.a.sum(); return x.gi.values[np.searchsorted(c.values, 0.5)]
G = M.groupby(["ksic", "year"])
Y = Y.merge(G.apply(wmed, include_groups=False).rename("gd_med").reset_index().rename(columns={"year": "t"}), on=["ksic", "t"], how="left")
M["aw"] = M.w_full  # 기업 동일가중(매출 무시) 성장
G = M.groupby(["ksic", "year"])
Y = Y.merge(G.apply(lambda x: np.average(x.gi, weights=x.aw), include_groups=False).rename("gd_eq").reset_index().rename(columns={"year": "t"}), on=["ksic", "t"], how="left")
# DART 업종코드(스냅샷) 배분 — 비교용: induty_code 5자리 = KSIC 인 기업 매출 전액
dart = pd.read_parquet(os.path.join(C, "dart_firms.parquet"))
fi = fr.merge(dart[["corp_code", "induty_code"]], on="corp_code"); fi = fi[fi.induty_code.str.len() == 5]
gi = fi.groupby(["induty_code", "year"]).apply(lambda x: np.log((x.rev_prev * np.exp(x.gi)).sum() / x.rev_prev.sum()), include_groups=False)
gi = gi.rename("gd_induty").reset_index().rename(columns={"induty_code": "ksic", "year": "t"})
ni = fi.groupby(["induty_code", "year"]).size().rename("nf_induty").reset_index().rename(columns={"induty_code": "ksic", "year": "t"})
Y = Y.merge(gi, on=["ksic", "t"], how="left").merge(ni, on=["ksic", "t"], how="left")
# 5개국 특허 공개 모멘텀(t년) — 제공 데이터
X = pd.read_parquet(os.path.join(C, "ksic_ext.parquet"))[["ksic", "t0", "pmom_foreign", "pmom_KR"]].rename(columns={"t0": "t"})
Y = Y.merge(X, on=["ksic", "t"], how="left")
for c in ["gd_med", "gd_eq", "gd_induty"]:
    Y[c + "_f"] = Y[c].fillna(Y.g_idx)
Y["nf"] = Y.nf_w_full.fillna(0)
Y["gd_shr"] = Y.gd_w_full_f * Y.nf / (Y.nf + 10) + Y.g_idx * 10 / (Y.nf + 10)
Y["gd_x_cov"] = Y.gd_w_full_f * np.log1p(Y.cov_ratio.fillna(0).clip(0, 5))
Y["g_lag"] = Y.g_lag
Y.to_parquet(os.path.join(C, "nowcast_panel2.parquet"), index=False)
print("coverage: gd_induty present", Y[Y.t.isin(DEV)].gd_induty.notna().mean().round(3), "nf_induty median", Y.nf_induty.median())

def run(feats, model="ols", clip=None):
    mae, rho = {}, {}
    for t in DEV:
        tr = Y[(Y.t <= t - 1) & (Y.t >= 2014)].dropna(subset=["g"])
        te = Y[Y.t == t].dropna(subset=["g"])
        med = tr[feats].median()
        ytr = tr.g.clip(-clip, clip) if clip else tr.g
        mdl = LinearRegression() if model == "ols" else HuberRegressor(epsilon=1.35, max_iter=1000)
        mdl.fit(tr[feats].fillna(med), ytr)
        p = mdl.predict(te[feats].fillna(med))
        mae[t] = np.mean(np.abs(te.g - p)); rho[t] = spearmanr(te.g, p).statistic
    return mae, rho

base, rb = run(["g_idx"])
def sk(r): return np.mean([1 - r[t] / base[t] for t in DEV])
champ = {"feats": ["g_idx", "gd_w_full_f"], "model": "ols", "clip": None}
cm, cr = run(**champ)
log = [{"exp": "B_M1", "change": "baseline idx OLS", "mean_mae": np.mean(list(base.values())), "skill": 0, "rho": np.mean(list(rb.values())), "decision": "BASE"},
       {"exp": "C0_M3", "change": "+gd_w_full (start champion)", "mean_mae": np.mean(list(cm.values())), "skill": sk(cm), "rho": np.mean(list(cr.values())), "decision": "CHAMP"}]
def trial(name, change, **spec):
    global champ, cm, cr
    s = {**champ, **spec}
    mm, rr = run(**s)
    d = sk(mm) - sk(cm); win = np.mean([mm[t] < cm[t] for t in DEV])
    dec = "ACCEPT" if (d >= 0.003 and win >= 0.6) else "ROLLBACK"
    log.append({"exp": name, "change": change, "mean_mae": round(np.mean(list(mm.values())), 4), "skill": round(sk(mm), 4),
                "d_skill": round(d, 4), "win": round(win, 2), "rho": round(np.mean(list(rr.values())), 3), "decision": dec,
                **{f"mae_{t}": round(mm[t], 4) for t in DEV}})
    print(log[-1], flush=True)
    if dec == "ACCEPT":
        champ, cm, cr = s, mm, rr
f0 = lambda: champ["feats"]
trial("E1", "+g_imp1(수입 t년 성장)", feats=f0() + ["g_imp1"])
trial("E2", "Huber 회귀", model="huber")
trial("E3", "학습 타깃 ±0.5 절단", clip=0.5)
trial("E4", "gd → 계층 수축(nf/(nf+10))", feats=[c if c != "gd_w_full_f" else "gd_shr" for c in f0()])
trial("E5", "+gd 가중중앙값", feats=f0() + ["gd_med_f"])
trial("E6", "+gd 기업동일가중", feats=f0() + ["gd_eq_f"])
trial("E7", "w_full → w_share", feats=[c if c != "gd_w_full_f" else "gd_w_share_f" for c in f0()])
trial("E8", "+g_lag(작년 조사 성장)", feats=f0() + ["g_lag"])
trial("E9", "+gd×log(1+커버율)", feats=f0() + ["gd_x_cov"])
trial("E10", "+5개국 특허 공개모멘텀", feats=f0() + ["pmom_foreign", "pmom_KR"])
trial("E11", "특허배분 → DART 업종코드 배분(스냅샷, 비교)", feats=[c if c not in ("gd_w_full_f", "gd_shr") else "gd_induty_f" for c in f0()])
trial("E12", "+DART 업종코드 배분 병행", feats=f0() + ["gd_induty_f"])
L = pd.DataFrame(log); L.to_csv(os.path.join(HERE, "iter_N1_nowcast.csv"), index=False)
pd.set_option("display.width", 250)
print(L[["exp", "change", "mean_mae", "skill", "d_skill", "win", "rho", "decision"]].to_string())
print("final champion", champ)
