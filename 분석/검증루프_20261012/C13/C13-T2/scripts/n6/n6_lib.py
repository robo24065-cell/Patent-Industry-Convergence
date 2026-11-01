# -*- coding: utf-8 -*-
"""N6 (I8) 정부R&D 수행기업의 '50개 산업 첫 특허' 전환 — 공용 라이브러리.
단위 = (사업자번호 bz, 원점 t0). t0-2..t0 과제연도에 NTIS 과제를 수행한 기업(중소·중견·대기업) 중
t0 말까지 공개·등록으로 알려진 제공범위(50 KSIC) KR 출원이 없는 기업.
결과 y = 출원일 t0+1..t0+2 인 제공범위 KR 출원 >= 1 (사업자번호 -> 출원인코드, 유일 상호 보조).

🔒 봉인: 최종시험 t0=2021 -> 출원 2022-01-01~2023-12-31 (F=2022).
   dev 모드는 출원일 >= 2022-01-01 인 출원을 메모리에 올리지 않는다(load_apps에서 필터, assert).
   dev 원점(학습·평가) t0 <= 2019 (결과창 끝 <= 2021-12-31).
파일럿(_ideas_R4/i8_build.py, r4_common.py)을 그대로 옮기고 도전자 피처를 덧붙였다. 파일럿 폴더에는 쓰지 않는다.
출처 표기: "NTIS Open API" 활용(비상업·내부 분석)."""
import os, re, time
import numpy as np, pandas as pd, duckdb

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
EXTF = EXT.replace("\\", "/")
R3I = os.path.join(MC, r"_ideas_R3")
OUT = os.path.join(MC, r"N6_정부RnD기업_첫특허전환")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

F_YEAR = 2022                 # 최종시험 첫 결과연도
DEV_SEAL_APPL = 20220101      # dev: 이 출원일 이상은 읽지 않음
FINAL_T0 = 2021
DEV_ORIGINS = list(range(2014, 2020))   # 평가 원점
MIN_TRAIN = 2012
T0_ALL_DEV = list(range(2010, 2020))    # 2010·2011은 과거 전환율(목표 인코딩) 이력 전용, 학습·평가에 쓰지 않음

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


_T = time.time()


def log(*a):
    print(*a, round(time.time() - _T), flush=True)


# ---------------------------------------------------------------- 평가
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p)


def skill(a, b):
    return (a - b) / (1 - b)


def topk(y, p, frac):
    y = np.asarray(y); p = np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    return float(y[np.argsort(-p, kind="stable")[:k]].mean())


XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def fit_pred(Xtr, ytr, Xte, model="xgb", seed=0, w=None):
    if model == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(Xtr, ytr, sample_weight=w)
        return m.predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr, logisticregression__sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def rank01(p):
    return pd.Series(p).rank(pct=True).values


def predict(tr, te, feats, ycol="y", spec=None):
    """spec: model 'xgb'|'logit'|'ens'(XGB+로지스틱 순위 평균), seeds, recent_decay(최근 원점 가중)."""
    spec = spec or {}
    model = spec.get("model", "xgb"); seeds = spec.get("seeds", [0]); decay = spec.get("recent_decay")
    Xtr = tr[feats].astype(float); Xte = te[feats].astype(float); y = tr[ycol].values
    w = None
    if decay:
        w = decay ** (tr.t0.max() - tr.t0.values).astype(float)
    if model in ("xgb", "logit"):
        ps = [fit_pred(Xtr, y, Xte, model=model, seed=s, w=w) for s in (seeds if model == "xgb" else [0])]
        return np.mean(ps, axis=0) if len(ps) > 1 else ps[0]
    if model == "ens":
        px = np.mean([fit_pred(Xtr, y, Xte, model="xgb", seed=s, w=w) for s in seeds], axis=0)
        pl = fit_pred(Xtr, y, Xte, model="logit", w=w)
        a = spec.get("w_xgb", 0.5)
        return a * rank01(px) + (1 - a) * rank01(pl)
    raise ValueError(model)


def boot_ci(y, pm, pb, n=1000, seed=0, groups=None):
    """skill 95% CI. groups가 주어지면 군집(사업자번호) 부트스트랩. AUC_모형·AUC_기준선 CI도 함께."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    if groups is None:
        sets = None
    else:
        gi = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(lambda s: s.values)
        sets = list(gi.values)
    sk, am, ab = [], [], []
    for _ in range(n):
        if sets is None:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(sets), len(sets)); i = np.concatenate([sets[k] for k in pick])
        if y[i].min() == y[i].max():
            continue
        a1, a0 = auc(y[i], pm[i]), auc(y[i], pb[i])
        sk.append(skill(a1, a0)); am.append(a1); ab.append(a0)
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return dict(skill_ci=q(sk), auc_ci=q(am), auc_m0_ci=q(ab))


# ---------------------------------------------------------------- 데이터 적재
def load_ntis(year_lo=2008, year_hi=2019):
    assert year_hi <= 2023
    nt = duckdb.sql(f"""select pjt_no, year, research_agency nm, regexp_replace(corp_rgst_no, '[^0-9]', '', 'g') bz, perform_grp, dev_phase, sixt, sixt_cd,
     cls_L_canon, cls_M, gov_funds, sbiz_funds, total_funds, continuous, ministry, business_name, apply1, total_end, tech_lifecycle
     from read_parquet('{EXTF}/ntis_project_raw.parquet') where year between {year_lo} and {year_hi} and perform_grp in ('중소기업','중견기업','대기업')""").df()
    tm = pd.read_parquet(f"{EXT}/ntis_project_ksic_text_map.parquet")
    nt = nt.merge(tm, on=["pjt_no", "year"], how="left")
    nt = nt[nt.bz.str.len() == 10].copy()
    return nt


def firm_link(nt):
    """사업자번호 -> 출원인코드 목록 (파일럿과 동일: 국세청 대상표(CORP_APPLICANT 사업자번호) 1차 + 유일 상호 보조).
    파일럿은 NTIS 2010–2019 기업의 최빈 상호를 썼다 — 재현을 위해 같은 범위 상호를 쓴다."""
    ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["applicantCd", "bizno", "aname"])
    b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
    ns["nn"] = ns.aname.map(norm); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
    src = nt[nt.year >= 2010]
    firm = src.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()
    rest = nt[~nt.bz.isin(set(firm.bz))]
    if len(rest):
        f2 = rest.groupby("bz").nm.agg(lambda s: s.value_counts().index[0]).rename("nm").reset_index()
        firm = pd.concat([firm, f2], ignore_index=True)
    firm["codes"] = firm.bz.map(b2a)
    nm_code = firm.nm.map(norm).map(uq)
    firm["codes"] = [c if isinstance(c, list) else ([n] if isinstance(n, str) else []) for c, n in zip(firm.codes, nm_code)]
    firm["linked"] = firm.codes.map(len) > 0
    firm["key"] = firm.bz.str[:6] + "|" + firm.nm.map(norm)
    return firm


def load_apps(firm, mode="dev"):
    """제공 KR 출원(app_firm) -> 기업. dev: 출원일 < 2022-01-01 만 적재(봉인)."""
    af = pd.read_parquet(f"{R3I}/cache/app_firm.parquet", columns=["applicantCd", "appl_no", "appl_date", "known_date", "ksic"])
    if mode == "dev":
        af = af[af.appl_date < DEV_SEAL_APPL]
        assert af.appl_date.max() < DEV_SEAL_APPL
    elif mode == "final":
        af = af[af.appl_date <= 20231231]
    ex = firm.explode("codes").dropna(subset=["codes"])
    af = af[af.applicantCd.isin(set(ex.codes))].merge(ex[["bz", "codes"]], left_on="applicantCd", right_on="codes")
    return af
