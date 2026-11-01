# -*- coding: utf-8 -*-
"""N2 최종시험(t=2024) 독립 재계산 — final_run_r2.py --mode final 실행 뒤에만 실행.
(1) 레이블: 광업제조업조사 원본 CSV에서 11차 T04 2023·2024 출하액을 직접 읽어 log 성장 재계산 → final_predictions_2024.csv 의 y 대조
(2) M1: 광공업생산지수 원본에서 출하지수 3자리(없으면 2자리) 연간합 성장을 직접 계산, numpy 최소제곱(학습 2014..2023)으로 재적합 → M1 예측 대조
(3) 챔피언: n_lib 피처표(동일 정보집합)로 statsmodels QuantReg(q=0.5) 재적합 → 예측 대조 (sklearn highs 와 독립 구현)
(4) MAE·skill·KSIC 부트스트랩 CI를 별도 코드로 재계산 → final_result.json 대조
출력: final_verify.json"""
import os, sys, json
import numpy as np, pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
assert os.path.exists(os.path.join(HERE, "final_result.json")), "최종시험 이후에만"
R = json.load(open(os.path.join(HERE, "final_result.json"), encoding="utf-8"))
FP = pd.read_csv(os.path.join(HERE, "final_predictions_2024.csv"), dtype={"ksic": str})
ch = FP[FP.model == "champion"].set_index("ksic"); m1 = FP[FP.model == "M1"].set_index("ksic")
out = {}

# (1) 레이블
k = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
k = k[(k.항목코드 == "T04") & (k.기준 == "11차") & k.연도.isin([2023, 2024])]
lv = k.pivot_table(index="세세분류코드", columns="연도", values="값")
y = np.log(lv[2024] / lv[2023]).dropna()
y = y[(lv[2023] > 0) & (lv[2024] > 0)]
com = ch.index.intersection(y.index)
out["label_n_raw"] = int(len(y)); out["label_n_pred"] = int(len(ch))
out["label_max_abs_diff"] = float(np.max(np.abs(ch.loc[com, "y"] - y.loc[com])))

# (2) M1 독립
pi = pd.read_csv(os.path.join(BASE, r"공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv"), dtype=str)
pi = pi[pi.항목명 == "출하지수(원지수)"].copy()
pi["v"] = pd.to_numeric(pi.값, errors="coerce"); pi["ym"] = pi.연월.astype(int)
pi = pi[pi.ym <= 202412]; pi["yr"] = pi.ym // 100
A = pi.groupby(["산업코드", "yr"]).v.agg(["sum", "count"]); A = A[A["count"] == 12]["sum"]


def gidx(code, t):
    for L in (3, 2):
        c = "C" + code[:L]
        if (c, t) in A.index and (c, t - 1) in A.index and A[(c, t - 1)] > 0:
            return np.log(A[(c, t)] / A[(c, t - 1)])
    return np.nan


kk = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
kk = kk[(kk.항목코드 == "T04") & (kk.연도 <= 2023)]
rows = []
for (ks, bs), g in kk.groupby(["세세분류코드", "기준"]):
    s = g.set_index("연도").값.sort_index()
    for t in s.index:
        if 2014 <= t <= 2023 and t - 1 in s.index and s[t - 1] > 0 and s[t] > 0 and ((bs == "10차" and t <= 2020) or (bs == "11차" and t >= 2021)):
            rows.append((ks, t, np.log(s[t] / s[t - 1])))
TR = pd.DataFrame(rows, columns=["ksic", "t", "g"]); TR["x"] = [gidx(a, b) for a, b in zip(TR.ksic, TR.t)]
TR = TR.dropna()
X = np.c_[np.ones(len(TR)), TR.x]; beta = np.linalg.lstsq(X, TR.g.values, rcond=None)[0]
xt = np.array([gidx(a, 2024) for a in m1.index])
xt = np.where(np.isnan(xt), np.nanmedian(TR.x), xt)
p_m1 = beta[0] + beta[1] * xt
out["M1_n_train"] = int(len(TR)); out["M1_beta"] = [round(float(b), 5) for b in beta]
out["M1_pred_max_abs_diff"] = float(np.max(np.abs(p_m1 - m1.p.values)))

# (3) 챔피언 독립 적합 (피처표는 n_lib, 적합은 statsmodels)
import n_lib as L, n04_panel
L._cache["fin_rt2.parquet"] = pd.read_parquet(os.path.join(HERE, "cache", "fin_rt2_2024.parquet"))
L.PANEL_OVERRIDE = n04_panel.build(maxy=2024)
spec = L.load_state()["spec"]; Yb = L.build(spec); f = spec["feats"]
tr = Yb[(Yb.t >= 2014) & (Yb.t <= 2023)].dropna(subset=["g"]); te = Yb[Yb.t == 2024].set_index("ksic").loc[ch.index]
med = tr[f].median()
res = sm.QuantReg(tr.g.values, sm.add_constant(tr[f].fillna(med).values)).fit(q=0.5, max_iter=5000)
p_ch = res.predict(sm.add_constant(te[f].fillna(med).values, has_constant="add"))
out["champ_n_train"] = int(len(tr)); out["champ_coef_statsmodels"] = [round(float(c), 5) for c in res.params]
out["champ_pred_max_abs_diff_vs_sklearn"] = float(np.max(np.abs(p_ch - ch.p.values)))
out["feature_missing_2024"] = {c: int(te[c].isna().sum()) for c in f}
out["gd_nfirms_2024_median"] = float(te.nf_gd.median())

# (4) 지표 재계산
yv = ch.y.values
mae_ch = float(np.mean(np.abs(yv - ch.p.values))); mae_m1 = float(np.mean(np.abs(m1.y.values - m1.p.values)))
mae_ch_ind = float(np.mean(np.abs(y.loc[com].values - p_ch))); mae_m1_ind = float(np.mean(np.abs(y.loc[com].values - p_m1)))
out["mae_champ"] = round(mae_ch, 5); out["mae_M1"] = round(mae_m1, 5); out["skill"] = round(1 - mae_ch / mae_m1, 4)
out["skill_fully_independent(labels+M1+QuantReg)"] = round(1 - mae_ch_ind / mae_m1_ind, 4)
rng = np.random.default_rng(12345); e1 = np.abs(yv - ch.p.values); e0 = np.abs(m1.y.values - m1.p.values)
bs = [1 - e1[s].sum() / e0[s].sum() for s in (rng.integers(0, len(e1), len(e1)) for _ in range(5000))]
out["boot_ci_5000_seed12345"] = np.percentile(bs, [2.5, 50, 97.5]).round(4).tolist()
out["P_gt0"] = float(np.mean(np.array(bs) > 0))
out["win_share_ksic"] = round(float(np.mean(e1 < e0)), 3)
out["medae_champ"] = round(float(np.median(e1)), 5); out["medae_M1"] = round(float(np.median(e0)), 5)
out["dir_vs_median"] = round(float(np.mean(np.sign(yv - np.median(yv)) == np.sign(ch.p.values - np.median(ch.p.values)))), 3)
out["matches_final_result"] = bool(abs(out["skill"] - R["skill"]) < 1e-3 and abs(out["mae_champ"] - R["mae_model"]) < 1e-4)
# 오차 기여 상위 산업
d = pd.DataFrame({"ksic": ch.index, "y": yv, "p": ch.p.values, "pM1": m1.p.values}); d["gain"] = e0 - e1
out["top_gain"] = d.nlargest(5, "gain")[["ksic", "y", "p", "pM1", "gain"]].round(3).to_dict("records")
out["top_loss"] = d.nsmallest(5, "gain")[["ksic", "y", "p", "pM1", "gain"]].round(3).to_dict("records")
out["mae_excl_top1_abs_y"] = None
i = np.argmax(np.abs(yv)); msk = np.arange(len(yv)) != i
out["skill_excl_largest_|y|"] = round(1 - e1[msk].mean() / e0[msk].mean(), 4)
json.dump(out, open(os.path.join(HERE, "final_verify.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
