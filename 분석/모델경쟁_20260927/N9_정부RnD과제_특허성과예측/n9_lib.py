# -*- coding: utf-8 -*-
"""N9 (J1) 정부R&D 과제의 '50개 산업 특허' 성과 조기 예측 — 공용 모듈.
출처: NTIS Open API 활용(비상업·내부 분석).
🔒 봉인: 최종시험 과제연도 t=2022 → 출원 2022-01-01~2023-12-31 (F=2022).
   dev 모드는 NTIS 성과 연결을 출원연도 <= 2021만 읽는다(assert). 학습은 결과창 끝 t+1 <= 2021 (t <= 2020).
   이 트랙 폴더 밖에는 쓰지 않는다. sqlite는 쓰지 않는다(필요 시 mode=ro)."""
import os, time, re
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(MC, r"N9_정부RnD과제_특허성과예측")
CACHE = os.path.join(OUT, "cache")
KOSIS = os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv")
KSICDEF = os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json")
os.makedirs(CACHE, exist_ok=True)

F = 2022                      # 최종시험 첫 결과연도
FINAL_T = 2022                # 최종시험 과제연도
DEV_MAX_APPL_YEAR = 2021      # dev 모드에서 읽을 수 있는 성과 출원연도 상한
DEV_ORIGINS = list(range(2014, 2021))
GAP = 2                       # 원점 v 평가 시 학습 t <= v-2 (결과창 [t,t+1]이 v 이전에 끝남)
NEG_FRAC = 0.3

_t0 = time.time()
def log(*a):
    print(*a, round(time.time() - _t0), flush=True)


def norm_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = re.sub(r"\(주\)|㈜|주식회사|유한회사|\(유\)|유한책임회사|\(사\)|사단법인|재단법인|\(재\)|株式会社|co\.?,?\s*ltd\.?|inc\.?|corp(oration)?\.?|limited", "", s, flags=re.I)
    return re.sub(r"[\s\.,\-·()\[\]]", "", s).lower()


def auc(y, p, w=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def skill(a, b):
    return (a - b) / (1 - b)


XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def fit_pred(Xtr, ytr, Xte, model="xgb", seed=0, w=None, params=None):
    if model == "xgb":
        from xgboost import XGBClassifier
        pr = dict(XGB_PARAMS); pr.update(params or {})
        m = XGBClassifier(random_state=seed, **pr)
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr, logisticregression__sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def topk(y, p, frac=None, k=None):
    y = np.asarray(y); o = np.argsort(-np.asarray(p), kind="stable")
    if k is None:
        k = max(1, int(round(frac * len(y))))
    return float(y[o[:k]].mean()), int(y[o[:k]].sum()), k


def boot_ci(y, pm, pb, groups=None, n=200, seed=0):
    """군집 부트스트랩(군집별 뽑힌 횟수를 행 가중치로). skill·AUC·AUC_기준선 95% CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    if groups is None:
        codes = np.arange(len(y)); G = len(y)
    else:
        codes, uniq = pd.factorize(np.asarray(groups)); G = len(uniq)
    sk, am, ab = [], [], []
    for _ in range(n):
        cnt = rng.multinomial(G, np.full(G, 1.0 / G))
        w = cnt[codes].astype(float); m = w > 0
        if y[m].min() == y[m].max():
            continue
        a1 = auc(y[m], pm[m], w[m]); a0 = auc(y[m], pb[m], w[m])
        sk.append(skill(a1, a0)); am.append(a1); ab.append(a0)
    pc = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return dict(skill_ci=pc(sk), auc_ci=pc(am), auc_ref_ci=pc(ab))


def metrics(y, p, pb, prefix_ref="M0"):
    y = np.asarray(y)
    r = dict(n=int(len(y)), pos=int(y.sum()), base_rate=float(y.mean()), auc=float(auc(y, p)), **{f"auc_{prefix_ref}": float(auc(y, pb))})
    r["skill"] = float(skill(r["auc"], r[f"auc_{prefix_ref}"]))
    for fr in (0.01, 0.05, 0.10):
        k = f"top{int(fr * 100)}"
        r[k] = topk(y, p, fr)[0]; r[k + f"_{prefix_ref}"] = topk(y, pb, fr)[0]
        r[k + "_lift"] = r[k] / r["base_rate"]
    r["recall_top10"] = topk(y, p, 0.10)[1] / max(1, y.sum()); r[f"recall_top10_{prefix_ref}"] = topk(y, pb, 0.10)[1] / max(1, y.sum())
    r["hit100"] = topk(y, p, k=100)[1]; r[f"hit100_{prefix_ref}"] = topk(y, pb, k=100)[1]
    return r


# ---- 피처 묶음 (파일럿 IDEAS_R5 J1 정의 그대로) ----
M0_ORIG = ["log_gov", "log_tot", "nongov_sh", "sbiz_sh", "perform_grp_c", "dev_phase_c", "cont", "series_yr", "total_len", "period_m", "end_in_year", "ministry_c", "has_pre",
           "org_nproj", "org_kr_pat", "org_kr_rate", "pre_kr_pat"]
M1 = ["b1_tfidf_score", "b1_tfidf_assigned", "b2_dict_score", "b2_dict_assigned", "sixt_cd_c", "cls_L_canon_c", "econ_goal_cd_c", "tech_lifecycle_c", "apply1_cd_c"]
M2_ORIG = ["org_p50", "org_p50_rate", "org_p50_share", "pre_p50"]
LAGGED = ("org_kr_pat", "org_kr_rate", "pre_kr_pat", "org_p50", "org_p50_rate", "org_p50_share", "pre_p50")


def L2(fs):
    """보고 지연판: 이력 피처를 출원연도 <= t-2 버전으로."""
    return [f + "_L2" if f in LAGGED else f for f in fs]


M0_L2 = L2(M0_ORIG); M2_L2 = L2(M2_ORIG)
