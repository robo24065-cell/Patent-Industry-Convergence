# -*- coding: utf-8 -*-
r"""🔒 최종시험 사후 검증(모형은 고치지 않는다): 독립 재계산 대조 + 위약 + 학습 원점 피처 동일성.
- 독립 레이블: ntis_project_raw를 pyarrow로 직접 읽어(n7lib 미사용) 사업자번호별 첫 과제연도 → y_ind(2023–24), y1_ind(2023)
- 독립 AUC: 순위합(Mann–Whitney) 직접 계산, 상위 k 적중률 직접 계산
- 위약 A: 최종시험 레이블을 층(NPS 가입 × 출원 규모 5분위) 안에서 1회 섞어 skill
- 위약 B: 학습 레이블을 층(t0 × NPS 가입) 안에서 1회 섞어 챔피언(ens)·M0 재학습 → 실제 시험 레이블로 AUC
- 동일성: final 모드 패널의 학습 원점(2015–2020) 피처·레이블이 dev 모드 패널과 같은지(미래 자료가 과거 피처를 바꾸지 않았는지)
산출: final_verify_r5.json"""
import os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import rankdata
from n7lib import OUT, CACHE, EXT, predict, rank01

SPEC = json.load(open(os.path.join(OUT, "final_spec_r5.json"), encoding="utf-8"))
FE, M0F = SPEC["features"], SPEC["baseline_M0"]
P = pd.read_parquet(os.path.join(OUT, "final_test_pred_r5.parquet"))
R = json.load(open(os.path.join(OUT, "final_result_r5.json"), encoding="utf-8"))["results"][0]
V = {}


def auc_rs(y, p):
    y = np.asarray(y).astype(int); r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sk(a, b):
    return (a - b) / (1 - b)


def top(y, p, fr):
    k = max(1, int(round(fr * len(y)))); o = np.argsort(-np.asarray(p), kind="mergesort")[:k]
    return float(np.asarray(y)[o].mean())


# ---- 1. 독립 레이블 ----
t = pq.read_table(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["year", "corp_rgst_no"]).to_pandas()
t = t[(t.year >= 2010) & (t.year <= 2024) & t.corp_rgst_no.notna()]
t["bz"] = t.corp_rgst_no.astype(str).str.replace(r"\D", "", regex=True)
fy = t[t.bz.str.len() > 0].groupby("bz").year.min()
bz = P.bizno.astype(str).str.replace(r"\D", "", regex=True)
f = bz.map(fy).astype(float)
V["indep_label"] = dict(n=int(len(P)), any_first_le_2022=int((f <= 2022).sum()),
                        y_match=float((f.between(2023, 2024).astype(float).values == P.y.values).mean()),
                        y1_match=float(((f == 2023).astype(float).values == P.y1.values).mean()),
                        pos_indep=int(f.between(2023, 2024).sum()), pos_frozen=int(P.y.sum()))
y = f.between(2023, 2024).astype(int).values
# ---- 2. 독립 AUC·skill·top-k ----
a = {k: auc_rs(y, P[c]) for k, c in [("champ", "p_champ"), ("M0", "p_M0"), ("SB", "p_SB")]}
V["indep_metrics"] = dict(auc=a, skill_vs_M0=sk(a["champ"], a["M0"]), skill_vs_SB=sk(a["champ"], a["SB"]),
                          top={m: {f"{int(fr*100)}pct": top(y, P[c], fr) for fr in (0.01, 0.05, 0.10)} for m, c in [("champ", "p_champ"), ("M0", "p_M0")]},
                          frozen_auc=R["auc"], diff_champ=a["champ"] - R["auc"]["champ"], diff_M0=a["M0"] - R["auc"]["M0"])
# ---- 3. 패널 로드(위약·동일성) ----
FP = pd.read_parquet(os.path.join(CACHE, "final_panel.parquet"))
DP = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"))
te = FP[FP.t0 == 2022].reset_index(drop=True)
assert (te.applicantCd.values == P.applicantCd.values).all()
# 동일성
cols = [c for c in FE + ["y", "y1"] if c in DP.columns]
m = FP[FP.t0 <= 2020].merge(DP[DP.t0 <= 2020], on=["applicantCd", "t0"], suffixes=("_f", "_d"), how="outer", indicator=True)
eq = {}
for c in cols:
    x1, x2 = m[c + "_f"].astype(float), m[c + "_d"].astype(float)
    eq[c] = float(((x1 == x2) | (x1.isna() & x2.isna())).mean())
V["train_origin_identity"] = dict(rows_final=int((FP.t0 <= 2020).sum()), rows_dev=int((DP.t0 <= 2020).sum()),
                                  both=int((m._merge == "both").sum()), cols_checked=len(cols),
                                  min_match=min(eq.values()), not_full=[k for k, v in eq.items() if v < 1])
# ---- 4. 위약 A: 시험 레이블 층 내 섞기 ----
rng = np.random.default_rng(20261004)
st = te.nps_present.astype(int).astype(str) + "_" + pd.qcut(te.log_app.rank(method="first"), 5, labels=False).astype(str)
yp = y.copy()
for g, idx in pd.Series(np.arange(len(y))).groupby(st.values):
    yp[idx.values] = rng.permutation(yp[idx.values])
ac, a0 = auc_rs(yp, P.p_champ), auc_rs(yp, P.p_M0)
V["placebo_A_test_shuffle"] = dict(strata="nps_present × log_app 5분위", auc_champ=ac, auc_M0=a0, skill=sk(ac, a0))
# ---- 5. 위약 B: 학습 레이블 층 내 섞기 후 재학습 ----
tr = FP[FP.t0 <= 2020].reset_index(drop=True)
ys = tr.y.values.copy()
for g, idx in pd.Series(np.arange(len(tr))).groupby((tr.t0.astype(str) + "_" + tr.nps_present.astype(int).astype(str)).values):
    ys[idx.values] = rng.permutation(ys[idx.values])
pc = predict(tr[FE].astype(float), ys, te[FE].astype(float), model="ens", seeds=(0, 1, 2))
p0 = predict(tr[M0F].astype(float), ys, te[M0F].astype(float), model="xgb", seeds=(0, 1, 2))
bc, b0 = auc_rs(y, pc), auc_rs(y, p0)
V["placebo_B_train_shuffle"] = dict(strata="t0 × nps_present", auc_champ=bc, auc_M0=b0, skill=sk(bc, b0))
# 사후 목록 경유 사업자번호(src=RL) 시험 표본 규모
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "src"])
V["test_rl_bizno_only_rows"] = int(P.applicantCd.isin(set(ns.loc[ns.src == "RL", "applicantCd"])).sum())
json.dump(V, open(os.path.join(OUT, "final_verify_r5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(V, ensure_ascii=False, indent=1))
