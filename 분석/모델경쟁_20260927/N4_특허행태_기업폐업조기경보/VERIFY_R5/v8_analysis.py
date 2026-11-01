# -*- coding: utf-8 -*-
"""R5 레드팀(N4) 분석: 재현 대조, 위약, 강화 기준선, 대형 출원인 제외, 하위집단, 상위 k, 보정,
누수 영향(NPS 상호 별칭 보정 재학습, 모집단 IPW 근사). 모형은 고치지 않는다."""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import n4lib
n4lib.CACHE = os.path.join(HERE, "cache")
from n4lib import cluster_boot, skill, load_app_firm, load_registry, load_nts
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

A = lambda y, p, w=None: float(roc_auc_score(y, p, sample_weight=w))
C = os.path.join(HERE, "cache")
R = {}
F = pd.read_parquet(os.path.join(TR, "final_test_pred_r4.parquet"))   # 동결(읽기만)
M = pd.read_parquet(os.path.join(C, "preds_main.parquet"))
S1 = pd.read_parquet(os.path.join(C, "preds_sens1.parquet"))
S2 = pd.read_parquet(os.path.join(C, "preds_sens2.parquet"))
XO = pd.read_parquet(os.path.join(C, "preds_xgbonly.parquet"))
D = pd.read_parquet(os.path.join(C, "final_panel.parquet"))
te = D[D.t0 == 2023].reset_index(drop=True)
for X in (M, S1, S2, XO):
    assert (X.applicantCd.values == te.applicantCd.values).all()
y = te.y.values.astype(int); pc = M.p_champ.values; p0 = M.p_M0.values; bz = te.bizno.values
n = len(y)

# ---------------- 재현
mm = F.merge(M, on="applicantCd", suffixes=("_f", "_r"))
R["repro"] = dict(n_frozen=len(F), n_repro=len(M), n_common=len(mm), pos_frozen=int(F.y.sum()), pos_repro=int(y.sum()),
                  label_mismatch=int((mm.y_f != mm.y_r).sum()),
                  max_abs_diff_champ=float((mm.p_champ_f - mm.p_champ_r).abs().max()),
                  max_abs_diff_M0=float((mm.p_M0_f - mm.p_M0_r).abs().max()),
                  auc_champ=A(y, pc), auc_M0=A(y, p0), skill=skill(A(y, pc), A(y, p0)))
print("repro", R["repro"], flush=True)

# 주 KSIC(원점까지 공개된 출원 최다), 규모
AF = load_app_firm()
AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna() & (AF.known_date <= 20231231)]
mk = (AF.groupby(["applicantCd", "ksic"]).appl_no.nunique().reset_index()
      .sort_values("appl_no", ascending=False).drop_duplicates("applicantCd").set_index("applicantCd").ksic)
te["mksic"] = te.applicantCd.map(mk).fillna("NA").astype(str)
te["ksic2"] = te.mksic.str[:2]
sz = pd.cut(te.log_emp, [-1, np.log1p(9), np.log1p(49), np.log1p(299), 99], labels=["1-9인", "10-49인", "50-299인", "300인+"]).astype(str)
te["size"] = np.where(te.nps_present == 0, "NPS없음", sz)

# ---------------- 위약
strata = te.ksic2 + "_" + te["size"]
idx = pd.Series(np.arange(n)).groupby(strata.values).apply(np.array)
pl = []
for s in range(200):
    rng = np.random.default_rng(100 + s); yp = y.copy()
    for ix in idx:
        yp[ix] = rng.permutation(y[ix])
    pl.append((skill(A(yp, pc), A(yp, p0)), A(yp, pc), A(yp, p0)))
pl = np.array(pl)
full = []
for s in range(200):
    yp = np.random.default_rng(900 + s).permutation(y); full.append(skill(A(yp, pc), A(yp, p0)))
full = np.array(full)
R["placebo"] = dict(strata="주KSIC 2자리 × 규모(NPS 가입자 구간/NPS없음)", n_strata=int(strata.nunique()), seeds=200,
                    skill_mean=float(pl[:, 0].mean()), p2_5=float(np.percentile(pl[:, 0], 2.5)), p97_5=float(np.percentile(pl[:, 0], 97.5)),
                    auc_champ_mean=float(pl[:, 1].mean()), auc_M0_mean=float(pl[:, 2].mean()),
                    p_ge_obs=float((pl[:, 0] >= R["repro"]["skill"]).mean()),
                    full_mean=float(full.mean()), full_p2_5=float(np.percentile(full, 2.5)), full_p97_5=float(np.percentile(full, 97.5)))
print("placebo", R["placebo"], flush=True)


# ---------------- 강화 기준선
def vs(pb, nb=500, seed=7):
    ci = cluster_boot(y, pc, pb, bz, n=nb, seed=seed)
    return dict(auc_base=A(y, pb), skill=skill(A(y, pc), A(y, pb)), ci95=ci["skill_ci"])


R["strong_baseline"] = {"M0_xgb(공식)": vs(p0), "M0_ens": vs(M.p_M0_ens.values),
                        "M0+NPS탈퇴이력(C01)_ens": vs(S1.p_M0C01_ens.values),
                        "외부데이터 전부(M0+C01+NTIS+DART)_ens": vs(S1.p_EXT_ens.values)}
R["ablation"] = {k: dict(auc=A(y, S2[f"p_{k}"].values), skill_vs_M0=skill(A(y, S2[f"p_{k}"].values), A(y, p0)))
                 for k in ["champ_noC11", "champ_noC04", "champ_noDART"]}
print("strong", R["strong_baseline"], flush=True)


# ---------------- 대형 출원인 제외 / 하위집단
def sub(mask, label, pm=None, pb=None):
    pm = pc if pm is None else pm; pb = p0 if pb is None else pb
    yy = y[mask]
    if yy.sum() < 10:
        return dict(label=label, n=int(mask.sum()), pos=int(yy.sum()), note="양성 부족")
    a1, a0 = A(yy, pm[mask]), A(yy, pb[mask])
    return dict(label=label, n=int(mask.sum()), pos=int(yy.sum()), rate=float(yy.mean()), auc_champ=a1, auc_M0=a0, skill=skill(a1, a0))


na = te.n_app.values
R["exclude_large"] = [sub(na < np.quantile(na, q), f"출원 수 상위 {int(round((1 - q) * 100))}% 제외") for q in (0.99, 0.95, 0.90)]
R["exclude_large"].append(sub(~((te.nps_present == 1) & (te.log_emp >= np.log1p(300))).values, "가입자 300인 이상 제외"))
subs = [sub((te.nps_present == 1).values, "NPS 가입"), sub((te.nps_present == 0).values, "NPS 미가입")]
for s_ in ["1-9인", "10-49인", "50-299인", "300인+"]:
    subs.append(sub((te["size"] == s_).values, f"NPS {s_}"))
for k2 in sorted(te.ksic2.unique()):
    if (te.ksic2 == k2).sum() >= 1500:
        subs.append(sub((te.ksic2 == k2).values, f"주KSIC {k2}xxx"))
subs.append(sub((te.n_app <= 2).values, "출원 1–2건"))
subs.append(sub((te.n_app >= 10).values, "출원 10건 이상"))
R["subgroups"] = subs


# ---------------- 상위 k
def topk(p, k):
    return int(y[np.argsort(-p)[:k]].sum())


R["topk"] = [dict(k=k, champ_hits=topk(pc, k), M0_hits=topk(p0, k), EXT_hits=topk(S1.p_EXT_ens.values, k),
                  champ_prec=topk(pc, k) / k, M0_prec=topk(p0, k) / k, champ_recall=topk(pc, k) / y.sum(), lift=topk(pc, k) / k / y.mean())
             for k in [100, int(0.01 * n), 500, int(0.05 * n), int(0.10 * n), int(0.20 * n)]]
R["pr_auc"] = dict(champ=float(average_precision_score(y, pc)), M0=float(average_precision_score(y, p0)),
                   EXT=float(average_precision_score(y, S1.p_EXT_ens.values)), base=float(y.mean()))


# ---------------- 보정 (동결 챔피언 점수는 순위 평균 → 확률 아님. XGB 단독 성분과 M0로 본다)
def calib(p):
    d = pd.DataFrame({"p": p, "y": y}); d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
    t = d.groupby("dec").agg(mean_pred=("p", "mean"), obs=("y", "mean")).reset_index()
    from sklearn.linear_model import LogisticRegression
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(q / (1 - q)).reshape(-1, 1), y)
    return dict(brier=float(brier_score_loss(y, p)), mean_pred=float(p.mean()), obs=float(y.mean()),
                slope=float(lr.coef_[0][0]), intercept=float(lr.intercept_[0]), deciles=t.round(4).to_dict("records"))


d = pd.DataFrame({"p": pc, "y": y}); d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
R["calibration"] = {"champ_rank_decile_obs": d.groupby("dec").y.mean().round(4).tolist(),
                    "champ_xgb_component": calib(XO.p_champ_xgb.values), "M0_xgb": calib(p0),
                    "auc_champ_xgb_only": A(y, XO.p_champ_xgb.values)}

# ---------------- 누수 영향 1: NPS 연결키의 2026 상호(사후 목록)
hid = pd.read_parquet(os.path.join(C, "nps_hidden_2023.parquet"))
h = te.applicantCd.isin(set(hid.applicantCd)).values
L = dict(hidden_n=int(h.sum()), hidden_pos=int(y[h].sum()), hidden_rate=float(y[h].mean()),
         other_nps0_rate=float(y[(te.nps_present.values == 0) & ~h].mean()),
         frozen_excl_hidden=sub(~h, "숨은 NPS 가입 기업 제외(동결 예측)"),
         frozen_nonnps_excl_hidden=sub((te.nps_present.values == 0) & ~h, "NPS 미가입 중 숨은 가입 제외"),
         frozen_hidden_only=sub(h, "숨은 NPS 가입 기업만"))
MA = pd.read_parquet(os.path.join(C, "preds_alias_main.parquet"))
DA = pd.read_parquet(os.path.join(C, "final_panel_alias.parquet")); ta = DA[DA.t0 == 2023].reset_index(drop=True)
assert (MA.applicantCd.values == ta.applicantCd.values).all() and (ta.y.values == te.y.values).all()
pca, p0a = MA.p_champ.values, MA.p_M0.values
cia = cluster_boot(y, pca, p0a, bz, n=500, seed=11)
m0a = ta.nps_present.values == 0
k1 = int(0.01 * n)
L["alias_refit"] = dict(auc_champ=A(y, pca), auc_M0=A(y, p0a), auc_M0_ens=A(y, MA.p_M0_ens.values),
                        skill=skill(A(y, pca), A(y, p0a)), ci95=cia["skill_ci"],
                        skill_vs_M0ens=skill(A(y, pca), A(y, MA.p_M0_ens.values)), nps_present_share=float(ta.nps_present.mean()),
                        nonnps=dict(n=int(m0a.sum()), rate=float(y[m0a].mean()), auc_M0=A(y[m0a], p0a[m0a]), auc_champ=A(y[m0a], pca[m0a])),
                        top1pct=dict(champ=topk(pca, k1) / k1, M0=topk(p0a, k1) / k1))
if os.path.exists(os.path.join(C, "preds_alias_sens1.parquet")):
    SA = pd.read_parquet(os.path.join(C, "preds_alias_sens1.parquet"))
    L["alias_refit"]["auc_EXT_ens"] = A(y, SA.p_EXT_ens.values)
    L["alias_refit"]["skill_vs_EXT_ens"] = skill(A(y, pca), A(y, SA.p_EXT_ens.values))
    L["alias_refit"]["auc_M0C01_ens"] = A(y, SA.p_M0C01_ens.values)
R["leak_nps_name"] = L
print("leak_nps", L, flush=True)

# ---------------- 누수 영향 2: 모집단(사후 사업자번호 목록) — 원점 시점 특허 활동 셀별 연결률로 IPW 근사
Rg = load_registry(); ns = load_nts(); end = 20231231
p = Rg["pt"][Rg["pt"].rgstDate <= end]
firms = set(AF.applicantCd) & set(p.applicantCd)
g = AF[AF.applicantCd.isin(firms)].groupby("applicantCd")
f = pd.DataFrame({"last": g.known_date.max(), "n": g.appl_no.nunique()})
f["yb"] = pd.cut(2023 - f["last"] // 10000, [-1, 0, 2, 5, 10, 100]).astype(str)
f["nb"] = pd.cut(f.n, [0, 1, 3, 10, 1e9]).astype(str)
live = p[~(p.ext_d <= end)].groupby("applicantCd").rgst_no.nunique()
f["lv"] = (f.index.map(live).fillna(0) > 0).astype(str)
f["linked"] = f.index.isin(set(ns.applicantCd))
f["cell"] = f.yb + f.nb + f.lv
prop = f.groupby("cell").linked.mean()
w = 1 / te.applicantCd.map(f.cell).map(prop).astype(float).clip(lower=0.05).values
R["leak_population_ipw"] = dict(n_cells=int(prop.size), w_mean=float(w.mean()), w_max=float(w.max()),
                                auc_champ_w=A(y, pc, w), auc_M0_w=A(y, p0, w), skill_w=skill(A(y, pc, w), A(y, p0, w)),
                                rate_w=float((w * y).sum() / w.sum()))
print("ipw", R["leak_population_ipw"], flush=True)
json.dump(R, open(os.path.join(HERE, "v8_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
for s in R["subgroups"] + R["exclude_large"]:
    print(s)
for s in R["topk"]:
    print(s)
print(R["pr_auc"])
print({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "deciles"}) for k, v in R["calibration"].items()})
print(R["calibration"]["champ_xgb_component"]["deciles"])
