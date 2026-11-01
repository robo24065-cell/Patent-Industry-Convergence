# -*- coding: utf-8 -*-
r"""N4 최종시험 독립 재계산 대조 (final_run_r4.py 실행 후 1회).
final_run_r4.py·n4lib의 채점 함수(predict·auc·cluster_boot)를 쓰지 않고 다시 계산한다.
1) 시험 레이블: 국세청 원본(r3nts_applicant_status.parquet)의 end_dt로 직접 다시 만든다(2024-01-01~2025-12-31).
2) 모형 적합: xgboost·sklearn을 직접 호출(같은 동결 명세), 순위 평균도 직접 계산.
3) AUC: Mann–Whitney 순위 공식(scipy.stats.rankdata)으로 계산.
4) 군집 부트스트랩: 사업자번호 군집, 가중치(다항분포 횟수) 방식 1000회 — 다른 난수 구현.
→ verify_final_r4.json, final_test_pred_r4.parquet(시험 예측, 보고서 부가 분석용)"""
import os, json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

OUT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(OUT), "_shared", "ext")
SPEC = json.load(open(os.path.join(OUT, "final_spec_r4.json"), encoding="utf-8"))
FE, M0F = SPEC["features"], SPEC["baseline_M0"]
XP = SPEC["model"]["xgb_params"]; SEEDS = SPEC["model"]["seeds"]
FR = json.load(open(os.path.join(OUT, "final_result_r4.json"), encoding="utf-8"))["results"][0]

D = pd.read_parquet(os.path.join(OUT, "cache", "final_panel.parquet"))
tr = D[D.t0.between(2015, 2021)].copy(); te = D[D.t0 == 2023].copy()
assert tr.t0.max() + 2 < 2024 and (D.t0 != 2022).all()

# 1) 레이블 독립 재계산
ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "end_dt", "registered"])
ns = ns[ns.registered.fillna(False).astype(bool)].drop_duplicates("applicantCd").set_index("applicantCd")
e = te.applicantCd.map(ns.end_dt)
y_ind = ((e >= pd.Timestamp("2024-01-01")) & (e <= pd.Timestamp("2025-12-31"))).astype(int).values
closed_before = int((e <= pd.Timestamp("2023-12-31")).sum())
lab = dict(n=int(len(te)), pos_indep=int(y_ind.sum()), pos_panel=int(te.y.sum()), mismatch=int((y_ind != te.y.values).sum()),
           closed_before_t0_in_pop=closed_before)
y = y_ind


def xgb_mean(F):
    ps = []
    for s in SEEDS:
        m = XGBClassifier(random_state=s, **XP); m.fit(tr[F].astype(float), tr.y); ps.append(m.predict_proba(te[F].astype(float))[:, 1])
    return np.mean(ps, axis=0)


def logit(F):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(tr[F].astype(float), tr.y); return m.predict_proba(te[F].astype(float))[:, 1]


def pct(p):
    return rankdata(p) / len(p)   # 평균 순위 / n (pandas rank(pct=True)와 같은 정의)


def auc_mw(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


t = time.time()
px = xgb_mean(FE); pl = logit(FE); pc = 0.5 * pct(px) + 0.5 * pct(pl)
p0 = xgb_mean(M0F); p0l = logit(M0F); p0e = 0.5 * pct(p0) + 0.5 * pct(p0l)
A = dict(champ=auc_mw(y, pc), M0_xgb=auc_mw(y, p0), M0_ens=auc_mw(y, p0e), champ_xgb_only=auc_mw(y, px), champ_logit_only=auc_mw(y, pl))
sk = lambda a, b: (a - b) / (1 - b)

# 4) 군집 부트스트랩(다항 가중치 방식): 군집별 가중치 w_c ~ Multinomial(C, 1/C) → 행 가중 AUC
cl = pd.factorize(te.bizno.astype(str))[0]; C = cl.max() + 1
rng = np.random.default_rng(20231231)


def wauc(y, p, w):
    o = np.argsort(p, kind="mergesort"); ys, ps, ws = y[o], p[o], w[o]
    # 동점 처리: 같은 점수 블록 안에서는 절반 기여
    uniq, start = np.unique(ps, return_index=True)
    wneg = np.where(ys == 0, ws, 0.0); wpos = np.where(ys == 1, ws, 0.0)
    bneg = np.add.reduceat(wneg, start); bpos = np.add.reduceat(wpos, start)
    cneg = np.concatenate([[0], np.cumsum(bneg)[:-1]])
    return float((bpos * (cneg + 0.5 * bneg)).sum() / (bpos.sum() * bneg.sum()))


bs = []
for _ in range(1000):
    wc = rng.multinomial(C, np.full(C, 1.0 / C)).astype(float)
    w = wc[cl]
    a1, a0, a0e = wauc(y, pc, w), wauc(y, p0, w), wauc(y, p0e, w)
    bs.append((sk(a1, a0), sk(a1, a0e)))
bs = np.array(bs)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
k = max(1, int(0.01 * len(te)))
res = dict(run_at=time.strftime("%Y-%m-%d %H:%M"), secs=round(time.time() - t), labels=lab, n_train=int(len(tr)), n_clusters=int(C),
           auc={k_: float(v) for k_, v in A.items()},
           skill_vs_M0=float(sk(A["champ"], A["M0_xgb"])), skill_vs_M0_ci95=q(bs[:, 0]),
           skill_vs_M0ens=float(sk(A["champ"], A["M0_ens"])), skill_vs_M0ens_ci95=q(bs[:, 1]),
           top1pct=dict(champ=float(y[np.argsort(-pc)[:k]].mean()), M0=float(y[np.argsort(-p0)[:k]].mean())),
           compare_final_run=dict(auc_champ=FR["auc"]["champ"], auc_M0=FR["auc"]["M0_xgb"], auc_M0ens=FR["auc"]["M0_ens"],
                                  skill=FR["skill_vs_M0"], skill_ci=FR["skill_vs_M0_ci95"],
                                  d_auc_champ=float(A["champ"] - FR["auc"]["champ"]), d_auc_M0=float(A["M0_xgb"] - FR["auc"]["M0_xgb"]),
                                  d_skill=float(sk(A["champ"], A["M0_xgb"]) - FR["skill_vs_M0"])))
# 5) 부가: NPS 미등록 하위집단·위험 10분위 폐업률
m = te.nps_present.values == 0
res["nonnps"] = dict(n=int(m.sum()), rate=float(y[m].mean()), auc_M0=float(auc_mw(y[m], p0[m])), auc_champ=float(auc_mw(y[m], pc[m])))
dec = pd.qcut(pd.Series(pc).rank(method="first"), 10, labels=False)
res["decile_rate_champ"] = [float(v) for v in pd.Series(y).groupby(dec.values).mean().values]
res["decile_rate_M0"] = [float(v) for v in pd.Series(y).groupby(pd.qcut(pd.Series(p0).rank(method="first"), 10, labels=False).values).mean().values]
pd.DataFrame(dict(applicantCd=te.applicantCd.values, bizno=te.bizno.values, y=y, p_champ=pc, p_M0=p0, nps_present=te.nps_present.values)).to_parquet(
    os.path.join(OUT, "final_test_pred_r4.parquet"), index=False)
json.dump(res, open(os.path.join(OUT, "verify_final_r4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
