# -*- coding: utf-8 -*-
"""R5 레드팀 3단계(N5): 재현 대조, 위약(층 내 레이블 섞기), 민감도(대형 출원인 제외·강화 기준선·하위집단), 상위 k 적중, 보정."""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
from n5lib import cluster_boot_ci, skill
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
R = {}
F = pd.read_parquet(os.path.join(TR, "final_preds_r4.parquet"))          # 동결 예측(읽기만)
P = pd.read_parquet(os.path.join(HERE, "cache", "repro_preds.parquet"))  # 재현 예측
S = pd.read_parquet(os.path.join(HERE, "cache", "sens_preds.parquet"))
te = pd.read_parquet(os.path.join(HERE, "cache", "panel_final.parquet"))
assert (P.appl_no.values == te.appl_no.values).all() and (S.appl_no.values == te.appl_no.values).all()
m = F.merge(P, on="appl_no", suffixes=("_frz", "_rep"))
R["repro"] = dict(n_frozen=len(F), n_repro=len(P), n_common=len(m), pos_frozen=int(F.y.sum()), pos_repro=int(P.y.sum()),
                  label_mismatch=int((m.y_frz != m.y_rep).sum()),
                  max_abs_pred_diff={k: float((m[f"p_{k}_frz"] - m[f"p_{k}_rep"]).abs().max()) for k in ["champ", "M0", "M1_pilot", "champ_logit", "M0_logit"]})
y = te.y.values; g = te.app.values
pc, p0 = P.p_champ.values, P.p_M0.values
A = lambda yy, pp: float(roc_auc_score(yy, pp))
R["repro"]["auc_champ"] = A(y, pc); R["repro"]["auc_M0"] = A(y, p0); R["repro"]["skill"] = skill(A(y, pc), A(y, p0))
# ---------------- 위약: 층(출원연도 × 출원인 규모 4분위 × 등록여부) 안에서 레이블 섞기
te["size_q"] = pd.qcut(te.app_logn.rank(method="first"), 4, labels=False)
strata = te.ay.astype(str) + "_" + te.size_q.astype(str) + "_" + te.reg_t0.astype(int).astype(str)
sk_pl, a1_pl, a0_pl = [], [], []
idx_by = pd.Series(np.arange(len(y))).groupby(strata.values).apply(np.array)
for s in range(200):
    rng = np.random.default_rng(1000 + s); yp = y.copy()
    for ix in idx_by:
        yp[ix] = rng.permutation(y[ix])
    a1, a0 = A(yp, pc), A(yp, p0); a1_pl.append(a1); a0_pl.append(a0); sk_pl.append(skill(a1, a0))
# 완전 무작위 섞기도 병기
sk_full = []
for s in range(200):
    yp = np.random.default_rng(5000 + s).permutation(y); sk_full.append(skill(A(yp, pc), A(yp, p0)))
R["placebo"] = dict(strata="출원연도×출원인규모4분위×t0등록여부", n_strata=int(strata.nunique()), seeds=200,
                    skill_mean=float(np.mean(sk_pl)), skill_p2_5=float(np.percentile(sk_pl, 2.5)), skill_p97_5=float(np.percentile(sk_pl, 97.5)),
                    auc_champ_mean=float(np.mean(a1_pl)), auc_M0_mean=float(np.mean(a0_pl)),
                    p_value_ge_obs=float(np.mean(np.array(sk_pl) >= R["repro"]["skill"])),
                    full_shuffle_skill_mean=float(np.mean(sk_full)), full_shuffle_p2_5=float(np.percentile(sk_full, 2.5)), full_shuffle_p97_5=float(np.percentile(sk_full, 97.5)))
# ---------------- 강화 기준선
def vs(base_key, src, n=1000):
    pb = src[f"p_{base_key}"].values
    (lo, hi), _ = cluster_boot_ci(y, pc, pb, g, n=n, seed=55)
    return dict(auc_base=A(y, pb), skill=skill(A(y, pc), A(y, pb)), ci95=[float(lo), float(hi)])
R["strong_baseline"] = {"M0_logit": vs("M0_logit", P), "M0plus_xgb(M0+sg_rate,sg_logn,app_rec_rate,mg_rec_rate)": vs("M0plus", S),
                        "M0_M2p_xgb(M0+fam_declared,app_recent_dec,mg_recent_dec)": vs("M0_M2p", S)}
# ---------------- 묶음 제거·NTIS 실시간 근사(재학습, 시험 1회 값 — 공식 아님)
R["ablation_final"] = {k: dict(auc=A(y, S[f"p_{k}"].values), skill_vs_M0=skill(A(y, S[f"p_{k}"].values), A(y, p0)))
                       for k in ["champ_noNTIS", "champ_noRoS", "champ_noTitle", "champ_ntis_rt_mean", "champ_ntis_zero_test"]}
# ---------------- 대형 출원인 제외
cnt = te.app.value_counts(); posc = te.groupby("app").y.sum().sort_values(ascending=False)
def sub(mask, label):
    yy = y[mask]
    if yy.sum() < 5 or yy.sum() == len(yy):
        return dict(label=label, n=int(mask.sum()), pos=int(yy.sum()), note="양성 부족")
    a1, a0 = A(yy, pc[mask]), A(yy, p0[mask])
    return dict(label=label, n=int(mask.sum()), pos=int(yy.sum()), auc_champ=a1, auc_M0=a0, skill=skill(a1, a0))
ex = []
for kk in [1, 3, 5, 10]:
    top = set(cnt.index[:kk]); ex.append(sub(~te.app.isin(top).values, f"단위 수 상위 {kk} 출원인 제외"))
for kk in [1, 3, 5]:
    top = set(posc.index[:kk]); ex.append(sub(~te.app.isin(top).values, f"양성 수 상위 {kk} 출원인 제외"))
R["exclude_large"] = ex
R["top_apps_share"] = dict(top1_units=float(cnt.iloc[0] / len(te)), top5_units=float(cnt.iloc[:5].sum() / len(te)),
                           top1_pos=float(posc.iloc[0] / y.sum()), top5_pos=float(posc.iloc[:5].sum() / y.sum()))
# ---------------- 하위집단
prior_dec = (te.app_rate > 0) & (te.app.map(te.groupby("app").app_logn.first()) >= 0)
has_prior = te.app_logn.values > 0
subs = [sub((te.app_rate.values > np.nanmedian(te.app_rate.values)), "출원인 과거 선언률 > 중앙값"),
        sub((te.app_rate.values <= np.nanmedian(te.app_rate.values)), "출원인 과거 선언률 ≤ 중앙값"),
        sub(te.pct.values == 1, "PCT 국내단계"), sub(te.pct.values == 0, "비PCT"),
        sub(te.reg_t0.values == 1, "t0까지 등록"), sub(te.reg_t0.values == 0, "t0 미등록"),
        sub(te.ay.values <= 2019, "출원 2017–2019"), sub(te.ay.values >= 2020, "출원 2020–2022"),
        sub(te.ntis_gov.values == 1, "NTIS 정부기원"), sub(te.ntis_gov.values == 0, "비정부기원")]
for q in range(4):
    subs.append(sub(te.size_q.values == q, f"출원인 규모 4분위 Q{q+1}"))
R["subgroups"] = subs
# ---------------- 상위 k 적중
def topk(p, k):
    o = np.argsort(-p)[:k]; return int(y[o].sum())
K = [50, 100, 200, 500, int(0.01 * len(y)), int(0.05 * len(y)), int(0.10 * len(y))]
R["topk"] = [dict(k=k, champ_hits=topk(pc, k), M0_hits=topk(p0, k), M0plus_hits=topk(S.p_M0plus.values, k),
                  champ_precision=topk(pc, k) / k, M0_precision=topk(p0, k) / k, champ_recall=topk(pc, k) / y.sum(),
                  champ_lift=(topk(pc, k) / k) / y.mean()) for k in K]
R["pr_auc"] = dict(champ=float(average_precision_score(y, pc)), M0=float(average_precision_score(y, p0)), M0plus=float(average_precision_score(y, S.p_M0plus.values)), base_rate=float(y.mean()))
# ---------------- 보정
def calib(p):
    d = pd.DataFrame({"p": p, "y": y}); d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
    t = d.groupby("dec").agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    from sklearn.linear_model import LogisticRegression
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    return dict(brier=float(brier_score_loss(y, p)), mean_pred=float(p.mean()), obs_rate=float(y.mean()),
                calib_slope=float(lr.coef_[0][0]), calib_intercept=float(lr.intercept_[0]), deciles=t.round(5).to_dict("records"))
R["calibration"] = {"champ": calib(pc), "M0": calib(p0)}
json.dump(R, open(os.path.join(HERE, "v3_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ["repro", "placebo", "strong_baseline", "ablation_final", "exclude_large", "top_apps_share", "pr_auc"]}, ensure_ascii=False, indent=1, default=float))
for s in R["subgroups"]: print(s)
for s in R["topk"]: print(s)
print({k: {kk: v for kk, v in R["calibration"][k].items() if kk != "deciles"} for k in R["calibration"]})
for d in R["calibration"]["champ"]["deciles"]: print(d)
