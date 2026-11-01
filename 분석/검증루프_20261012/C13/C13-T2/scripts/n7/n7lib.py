# -*- coding: utf-8 -*-
r"""N7 (I7) 특허기업의 정부R&D 첫 수행 진입 — 공용 라이브러리.
출발점: _ideas_R4\i7_build.py·r4_common.py·r4_eval.py(파일럿), 모집단 빌더는 N4\n4lib.py(= _ideas_R3\i09, 파일럿 I2b 패널과 전부 일치 확인됨)를 옮겼다.
단위 = (국내법인 출원인코드, t0): t0까지 제공범위 KR 등록 ≥1, t0 말까지 미폐업(국세청 end_dt), NTIS 과제(2010~t0) 수행 이력 없음.
결과 y = NTIS 과제연도 t0+1..t0+2에 그 사업자번호가 수행기관(corp_rgst_no)으로 처음 등장.
🔒 봉인: 최종시험 t0=2022 → 과제연도 2023–2024 (F=2023). 학습 t0 2015–2020.
  - dev 모드: NTIS 과제연도 ≤ 2022만 읽는다(assert). 레이블은 t0 ≤ 2020만. NPS ≤ 2020-12.
  - final 모드: NTIS ≤ 2024. 레이블 t0 ≤ 2020(학습) + t0=2022(시험). t0=2021은 결과창(2022–23)이 F와 겹쳐 학습·평가 모두 제외. NPS ≤ 2022-12.
  - future 모드: NTIS ≤ 2024. 학습 t0 ≤ 2022(결과 ≤2024), 예측 t0=2024(과제연도 2025–26, 레이블 없음). NPS ≤ 2024-12.
모든 파일 쓰기는 이 트랙 폴더(OUT) 안에서만. 원본 DB는 sqlite mode=ro."""
import os, re, sqlite3, time
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
OUT = os.path.join(MC, r"N7_특허기업_정부RnD첫수행")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
REG = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
BRS = os.path.join(BASE, "수집파이프라인", "business_rights.sqlite")

F_YEAR = 2023
DEV_ORIGINS = [2017, 2018, 2019, 2020]
TRAIN_MIN_T0 = 2015
FINAL_T0 = 2022
FUTURE_T0 = 2024
NTIS_YEAR_MAX = {"dev": 2022, "final": 2024, "future": 2024}
MODE_SNAP_MAX = {"dev": "2020-12", "final": "2022-12", "future": "2024-12"}
ALL_SNAPS = ["2015-12", "2016-12", "2017-12", "2018-12", "2019-12", "2020-12", "2021-12", "2022-12", "2023-12", "2024-12"]

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def label_allowed(t0, mode):
    """결과창 = 과제연도 t0+1..t0+2."""
    if mode == "dev":
        return t0 + 2 < F_YEAR
    if mode == "final":
        return t0 + 2 < F_YEAR or t0 == FINAL_T0
    if mode == "future":
        return t0 <= FINAL_T0
    raise ValueError(mode)


def origins_for(mode):
    if mode == "dev":
        return list(range(TRAIN_MIN_T0, 2021))
    if mode == "final":
        return list(range(TRAIN_MIN_T0, 2021)) + [FINAL_T0]
    return list(range(TRAIN_MIN_T0, FINAL_T0 + 1)) + [FUTURE_T0]


def snaps_for(mode):
    return [s for s in ALL_SNAPS if s <= MODE_SNAP_MAX[mode]]


# ---------------- 평가 ----------------
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p)


def skill(a, b):
    return (a - b) / (1 - b)


def topk(y, p, fr):
    """상위 fr 비율 적중률(정밀도)과 재현율"""
    y = np.asarray(y); p = np.asarray(p)
    k = max(1, int(round(fr * len(y))))
    o = np.argsort(-p, kind="mergesort")[:k]
    return float(y[o].mean()), float(y[o].sum() / max(1, y.sum()))


XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def fit_pred(Xtr, ytr, Xte, model="xgb", seed=0, sw=None):
    if model == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(random_state=seed, **XGB_PARAMS)
        m.fit(Xtr, ytr, sample_weight=sw)
        return m.predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    # 파일럿 r4_common과 같은 로지스틱(중앙값 대치 + 결측 지시자)
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    if sw is None:
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr, logisticregression__sample_weight=np.asarray(sw))
    return m.predict_proba(Xte)[:, 1]


def rank01(p):
    return pd.Series(p).rank(pct=True).values


def predict(Xtr, ytr, Xte, model="xgb", seeds=(0,), sw=None):
    if model == "xgb":
        return np.mean([fit_pred(Xtr, ytr, Xte, "xgb", s, sw) for s in seeds], axis=0)
    if model == "logit":
        return fit_pred(Xtr, ytr, Xte, "logit", sw=sw)
    if model == "ens":
        px = np.mean([fit_pred(Xtr, ytr, Xte, "xgb", s, sw) for s in seeds], axis=0)
        pl = fit_pred(Xtr, ytr, Xte, "logit", sw=sw)
        return 0.5 * rank01(px) + 0.5 * rank01(pl)
    raise ValueError(model)


def cluster_boot(y, pm, pb, cl, n=1000, seed=0):
    """군집(사업자번호) 부트스트랩: skill·AUC 95% CI"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True)
    idx_by = pd.Series(np.arange(len(y))).groupby(inv).apply(lambda s: s.values).values
    sk, am, ab = [], [], []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        i = np.concatenate([idx_by[j] for j in pick])
        if y[i].min() == y[i].max():
            continue
        a1, a0 = auc(y[i], pm[i]), auc(y[i], pb[i])
        sk.append(skill(a1, a0)); am.append(a1); ab.append(a0)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(skill_ci=q(sk), auc_model_ci=q(am), auc_base_ci=q(ab), n_boot=len(sk))


# ---------------- 원자료 로드 ----------------
def load_registry():
    """등록원부 파생표(N4 n4lib.load_registry와 같은 코드·같은 캐시 내용. 이 폴더 cache/에 없으면 원본 DB(mode=ro)에서 만든다)."""
    fp = {k: os.path.join(CACHE, f"reg_{k}.parquet") for k in ["ap", "pt", "lapnp", "trf", "py", "rec"]}
    if all(os.path.exists(v) for v in fp.values()):
        return {k: pd.read_parquet(v) for k, v in fp.items()}
    c = ro(REG)
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    ap = pd.read_sql("select rgst_no, applicantCd from applicant", c).drop_duplicates()
    pt = pd.read_sql("select rgst_no, rgstDate, applDate from patent", c)
    pt["rgstDate"] = num(pt.rgstDate); pt["applDate"] = num(pt.applDate)
    rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right", c)
    rt["d"] = num(rt.rgstCsDate)
    ext_first = rt[rt.rgstCsName == "소멸등록"].groupby("rgst_no").d.min()
    lapnp = rt[(rt.rgstCsName == "소멸등록") & rt.rgstCsReason.isin(["등록료불납", "연차료불납"])][["rgst_no", "d"]].merge(ap, on="rgst_no")
    trf = rt[rt.rgstCsName.fillna("").str.contains("이전")][["rgst_no", "d"]].merge(ap, on="rgst_no")
    rec = rt[rt.rgstCsName.fillna("").str.contains("회복")][["rgst_no", "d"]].merge(ap, on="rgst_no")
    py = pd.read_sql("select rgst_no, payDate, statAnnl, lastAnnl, payAmount from pay", c)
    py["d"] = num(py.payDate)
    for col in ["statAnnl", "lastAnnl", "payAmount"]:
        py[col] = pd.to_numeric(py[col], errors="coerce")
    py = py.drop(columns=["payDate"]).merge(pt[["rgst_no", "rgstDate"]], on="rgst_no", how="left").merge(ap, on="rgst_no")
    pt["ext_d"] = pt.rgst_no.map(ext_first)
    pt = pt.merge(ap, on="rgst_no")
    out = dict(ap=ap, pt=pt, lapnp=lapnp, trf=trf, py=py, rec=rec)
    for k, v in out.items():
        v.to_parquet(fp[k], index=False)
    return out


def load_registry_extra():
    """owner 테이블 권리자 변경(양도·합병·분할)과 포기 말소(N4 load_registry_extra와 동일)."""
    fo, fs = os.path.join(CACHE, "reg_own.parquet"), os.path.join(CACHE, "reg_sur.parquet")
    if os.path.exists(fo) and os.path.exists(fs):
        return pd.read_parquet(fo), pd.read_parquet(fs)
    c = ro(REG)
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    ap = pd.read_sql("select rgst_no, applicantCd from applicant", c).drop_duplicates()
    ow = pd.read_sql("select rgst_no, seq, ownerRgstCsDate, ownerRgstCsName from owner where seq > 0 and ownerRgstCsName is not null", c)
    ow["d"] = num(ow.ownerRgstCsDate)
    ow["kind"] = np.where(ow.ownerRgstCsName.str.contains("합병|분할"), "merge", "assign")
    ow = ow[["rgst_no", "d", "kind"]].merge(ap, on="rgst_no")
    rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName like '%말소%'", c)
    rt["d"] = num(rt.rgstCsDate)
    sur = rt[rt.rgstCsReason.fillna("").str.contains("포기")][["rgst_no", "d"]].merge(ap, on="rgst_no")
    ow.to_parquet(fo, index=False); sur.to_parquet(fs, index=False)
    return ow, sur


def load_nts():
    ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "b_stt_cd", "end_dt", "registered"])
    ns = ns[ns.registered.fillna(False).astype(bool)].drop_duplicates("applicantCd")
    ns["end"] = pd.to_numeric(ns.end_dt.dt.strftime("%Y%m%d"), errors="coerce")
    ns["corp_type"] = pd.to_numeric(ns.bizno.str[3:5], errors="coerce")
    return ns


def load_nps(mode):
    out = {}
    C = ["name", "biz6", "stat", "ldong", "indcd", "indnm", "adpt_dt", "rrg_dt", "n_emp", "amt", "n_new", "n_loss"]
    for s in snaps_for(mode):
        fc = os.path.join(CACHE, f"nps_firm_{s}.parquet")
        if not os.path.exists(fc):
            d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s.replace('-', '')}.parquet"), columns=C)
            d["nname"] = d.name.map(norm)
            d["key"] = d.biz6.fillna("") + "|" + d.nname
            d["act"] = d.stat == "1"
            d["rrg_dt"] = d.rrg_dt.fillna("").astype(str)
            d = d.sort_values("n_emp", ascending=False)
            g = d.groupby("key").agg(n_emp=("n_emp", "sum"), n_wkpl=("name", "size"), amt=("amt", "sum"), n_new=("n_new", "sum"),
                                     n_loss=("n_loss", "sum"), active=("act", "any"), n_act_wkpl=("act", "sum"),
                                     adpt=("adpt_dt", "min"), rrg_max=("rrg_dt", "max")).reset_index()
            g["snap"] = s
            g.to_parquet(fc, index=False)
        out[s] = pd.read_parquet(fc)
    return out


def load_app_firm():
    """제공 KR 출원 → 출원인코드(등록원부 정확 코드 + CORP_APPLICANT 유일 상호). N4 load_app_firm과 같은 캐시."""
    fc = os.path.join(CACHE, "app_firm.parquet")
    if os.path.exists(fc):
        return pd.read_parquet(fc)
    raise FileNotFoundError("app_firm 캐시 없음 — N4 n4lib.load_app_firm 코드로 재생성 필요")


def a2key():
    lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
    lk = lk[lk.nps_hit]
    lk["key"] = lk.biz6 + "|" + lk.nname
    return dict(zip(lk.applicantCd, lk.key))


def load_ntis(mode, cols=("year", "corp_rgst_no", "perform_grp", "gov_funds")):
    """NTIS 과제 raw — 모드별 과제연도 상한까지만 읽는다(dev ≤ 2022)."""
    import duckdb
    ymax = NTIS_YEAR_MAX[mode]
    f = os.path.join(EXT, "ntis_project_raw.parquet").replace("\\", "/")
    sel = ", ".join(cols)
    d = duckdb.sql(f"select {sel} from read_parquet('{f}') where year <= {ymax}").df()
    assert d.year.max() <= ymax
    if mode == "dev":
        assert d.year.max() < F_YEAR, "dev 모드에서 봉인 연도 과제를 읽음"
    return d


def ntis_first_year(NT):
    """파일럿 i7_build와 같은 정의: 과제연도 2010~ 수행기관 사업자번호(숫자만)의 첫 연도"""
    nt = NT[(NT.year >= 2010) & NT.corp_rgst_no.notna()].copy()
    nt["bz"] = nt.corp_rgst_no.str.replace(r"[^0-9]", "", regex=True)
    return nt.groupby("bz").year.min()


# ---------------- 기본 패널(M0·M1) ----------------
M0 = ["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type"]
M1_ADD = ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "lapse_1y", "lapse_2y",
          "pay_1y", "pay_trend", "trf_2y", "live_ratio"]
M2_ADD = ["g_part_any", "g_part_n", "g_part_proj", "g_part_univ", "g_part_gri", "g_part_firm", "g_part_rec", "g_part_share"]
M3_ADD = ["k_rnd_g", "k_rnd_sh"]
STRONG = M0 + ["log_app", "log_live", "n_ksic"]


def build_base(t0, AF, R, ns, NPS, A2K):
    """원점 t0의 특허 법인 모집단·M0·M1 (N4 build_base와 같은 정의; 폐업 레이블은 만들지 않는다)."""
    end = t0 * 10000 + 1231; y1 = end - 10000; y2 = end - 20000
    kn = AF[AF.known_date <= end]
    g = kn.groupby("applicantCd")
    f = pd.DataFrame({"n_app": g.appl_no.nunique(), "n_ksic": g.ksic.nunique(), "last_known": g.known_date.max()})
    f = f.join(ns.set_index("applicantCd")[["end", "corp_type", "bizno"]], how="inner")
    f = f[~(f.end <= end)]
    f["log_app"] = np.log1p(f.n_app); f["yrs_since_last"] = t0 - f.last_known // 10000
    a_new = kn[kn.ay.between(t0 - 3, t0 - 2)].groupby("applicantCd").appl_no.nunique()
    a_old = kn[kn.ay.between(t0 - 5, t0 - 4)].groupby("applicantCd").appl_no.nunique()
    f["app_trend"] = np.log((a_new.reindex(f.index).fillna(0) + 1) / (a_old.reindex(f.index).fillna(0) + 1))
    f["known_1y"] = kn[kn.known_date > y1].groupby("applicantCd").appl_no.nunique().reindex(f.index).fillna(0)
    pt, lapnp, py, trf = R["pt"], R["lapnp"], R["py"], R["trf"]
    p = pt[pt.rgstDate <= end]
    f = f[f.index.isin(set(p.applicantCd))].copy()
    f["n_live"] = p[~(p.ext_d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    npv = p[(p.rgstDate <= y1) & ~(p.ext_d <= y1)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
    f["lapse_1y"] = lapnp[(lapnp.d > y1) & (lapnp.d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    f["lapse_2y"] = lapnp[(lapnp.d > y2) & (lapnp.d <= y1)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    P1 = py[(py.d > y1) & (py.d <= end)].groupby("applicantCd").size().reindex(f.index).fillna(0)
    P2 = py[(py.d > y2) & (py.d <= y1)].groupby("applicantCd").size().reindex(f.index).fillna(0)
    f["pay_1y"] = P1 / (npv + 1); f["pay_trend"] = np.log((P1 + 1) / (P2 + 1))
    f["trf_2y"] = trf[(trf.d > y2) & (trf.d <= end)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / (npv + 1)
    f["log_live"] = np.log1p(f.n_live); f["live_ratio"] = f.n_live / (f.n_app + 1)
    cur = NPS[f"{t0}-12"]; cur = cur[cur.active].set_index("key")
    prv = NPS.get(f"{t0 - 1}-12"); prv = prv[prv.active].set_index("key") if prv is not None else None
    keys = pd.Series([A2K.get(a) for a in f.index], index=f.index)
    look = lambda df, col: pd.Series([df[col].get(k, np.nan) if k else np.nan for k in keys], index=f.index)
    emp = look(cur, "n_emp")
    f["nps_present"] = emp.notna().astype(float)
    f["log_emp"] = np.log1p(emp)
    f["g_emp1"] = np.log((emp + 1) / (look(prv, "n_emp") + 1)) if prv is not None else np.nan
    ad = pd.Series([cur.adpt.get(k, None) if k else None for k in keys], index=f.index)
    f["age"] = t0 - pd.to_numeric(ad.str[:4], errors="coerce")
    f["hire_rate"] = look(cur, "n_new") / emp
    f["loss_rate"] = look(cur, "n_loss") / emp
    f["wage"] = np.log1p(look(cur, "amt") / emp)
    f["key"] = keys
    f = f.drop(columns=["end"])
    f["t0"] = t0
    return f.reset_index().rename(columns={"index": "applicantCd"})


def build_panel(mode, verbose=True):
    """모집단(수행 이력 없음) + 레이블 + M0·M1·M2·M3. 파일럿 i7_build와 같은 정의."""
    AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
    R = load_registry(); ns = load_nts(); NPS = load_nps(mode); A2K = a2key()
    NT = load_ntis(mode)
    first = ntis_first_year(NT)
    bz_r3 = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno"]).drop_duplicates("applicantCd")
    lk = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["appl_no", "Year", "ProjectYear", "perform_grp", "ProjectID"])
    lk["Year"] = pd.to_numeric(lk.Year, errors="coerce"); lk["ProjectYear"] = pd.to_numeric(lk.ProjectYear, errors="coerce")
    KY = pd.read_parquet(os.path.join(EXT, "ntis_project_ksic_year.parquet")); KY = KY[KY.method == "a_link_pit"]
    parts = []
    for t0 in origins_for(mode):
        B = build_base(t0, AF, R, ns, NPS, A2K)
        B = B.drop(columns=["bizno"]).merge(bz_r3, on="applicantCd", how="left")
        B["first_ntis"] = B.bizno.str.replace(r"[^0-9]", "", regex=True).map(first)
        B = B[~(B.first_ntis <= t0)].copy()
        if label_allowed(t0, mode):
            assert t0 + 2 <= NTIS_YEAR_MAX[mode]
            B["y"] = B.first_ntis.between(t0 + 1, t0 + 2).astype(float)
            B["y1"] = (B.first_ntis == t0 + 1).astype(float)
        else:
            B["y"] = np.nan; B["y1"] = np.nan
            B["first_ntis"] = np.nan   # 레이블 없는 원점에는 미래 정보 컬럼을 남기지 않는다
        B = B.reset_index(drop=True)
        B = pd.concat([B, f_M2(B, t0, AF, lk), f_M3(B, t0, AF, KY)], axis=1)
        parts.append(B)
        if verbose:
            log(mode, t0, len(B), "y", B.y.sum(), round(B.y.mean(), 4) if B.y.notna().any() else None, "part_any", round(B.g_part_any.mean(), 3))
    return pd.concat(parts, ignore_index=True)


def f_M2(B, t0, AF, lk):
    """NTIS 참여 신호: 자사 특허가 정부과제 성과로 등록(레코드 Year ≤ t0, 과제연도 ≤ t0) — 파일럿 정의 그대로"""
    end = t0 * 10000 + 1231
    af = AF[AF.applicantCd.isin(set(B.applicantCd)) & (AF.known_date <= end)][["applicantCd", "appl_no", "known_date"]]
    a = af.merge(lk, on="appl_no", how="inner")
    a = a[(a.Year <= t0) & (a.ProjectYear <= t0)]
    ga = a.groupby("applicantCd")
    f = pd.DataFrame({"g_part_n": ga.appl_no.nunique(), "g_part_proj": ga.ProjectID.nunique(),
                      "g_part_univ": ga.perform_grp.apply(lambda s: (s == "대학").any()),
                      "g_part_gri": ga.perform_grp.apply(lambda s: s.isin(["출연연", "국공립연"]).any()),
                      "g_part_firm": ga.perform_grp.apply(lambda s: (s == "기업").any()),
                      "g_part_rec": t0 - ga.ProjectYear.max()})
    o = B[["applicantCd"]].merge(f, left_on="applicantCd", right_index=True, how="left").drop(columns="applicantCd")
    o.index = B.index
    o["g_part_any"] = o.g_part_n.notna().astype(float)
    for c in ["g_part_n", "g_part_proj"]:
        o[c] = np.log1p(o[c].fillna(0))
    for c in ["g_part_univ", "g_part_gri", "g_part_firm"]:
        o[c] = o[c].astype(float).fillna(0)
    nkn = af.groupby("applicantCd").appl_no.nunique()
    o["g_part_share"] = np.expm1(o.g_part_n) / B.applicantCd.map(nkn).fillna(1).values
    return o[M2_ADD]


def ksic_weights(B, t0, AF):
    end = t0 * 10000 + 1231
    kn = AF[(AF.known_date <= end) & AF.applicantCd.isin(set(B.applicantCd))]
    W = kn.groupby(["applicantCd", "ksic"]).appl_no.nunique().rename("w").reset_index()
    W["w"] = W.w / W.groupby("applicantCd").w.transform("sum")
    return W


def wavg(W, s, B):
    W = W.assign(v=W.ksic.map(s)).dropna(subset=["v"])
    v = (W.w * W.v).groupby(W.applicantCd).sum() / W.groupby("applicantCd").w.sum()
    return B.applicantCd.map(v).astype(float).values


def f_M3(B, t0, AF, KY):
    """KSIC 수준 정부연구비 성장·비중(자사 특허 KSIC 구성 가중, a_link_pit avail_year ≤ t0: 연구비 ≤ t0−1, 배분표 성과 ≤ 연도−2)"""
    ky = KY[KY.avail_year <= t0]
    pv = ky.pivot_table(index="ksic", columns="year", values="gov_funds", aggfunc="sum")
    a_ = pv[[y for y in pv.columns if t0 - 2 <= y <= t0 - 1]].sum(1); b_ = pv[[y for y in pv.columns if t0 - 4 <= y <= t0 - 3]].sum(1)
    gr = np.log((a_ + 1) / (b_ + 1)); gr[b_ <= 0] = np.nan
    sh = ky[ky.year == t0 - 1].groupby("ksic").gov_share_of_year.sum()
    W = ksic_weights(B, t0, AF)
    o = pd.DataFrame(index=B.index)
    o["k_rnd_g"] = wavg(W, gr, B); o["k_rnd_sh"] = wavg(W, sh, B)
    return o
