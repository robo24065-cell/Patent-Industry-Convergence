# -*- coding: utf-8 -*-
"""N5 표준특허(SEP) 선언 레이더 — 공용 라이브러리 (4라운드).
출발점: _ideas_R3/i07_sep_build.py(파일럿). 단위·결과·기준선 정의는 파일럿 그대로 두고, 도전자 피처 후보를 같은 원점 규칙으로 추가 계산한다.
단위 = KSIC 26429 KR 출원(출원 t0-5..t0, t0-12-31까지 공개 또는 등록, t0까지 ETSI 선언 미반영)
결과 = ETSI DB 최초 반영일(first_refl) ∈ (t0, t0+2]
🔒 최종시험 t0=2022(반영 2023–24, F=2023). dev 모드에서는 t0+2 < 2023 인 원점만 레이블을 만든다(assert).
원본 DB는 읽기전용(mode=ro). 외부 호출 없음."""
import os, re, sqlite3, time
import numpy as np, pandas as pd, duckdb

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
OUT = os.path.join(MC, r"N5_표준특허선언레이더")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
F_YEAR = 2023            # 최종시험 첫 결과연도
T0_FINAL = 2022
T0_FUTURE = 2024
KS_MAIN = "26429"

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):  # 파일럿 i00_common.norm과 동일
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


CO_STOP = {"inc", "incorporated", "corp", "corporation", "co", "ltd", "limited", "llc", "oy", "ab", "ag", "sa", "sas", "gmbh", "plc", "the", "kk", "bv", "nv", "company", "holdings", "holding"}


def co_norm(s):
    s = str(s or "").lower().replace("&", " and ")
    w = [x for x in re.split(r"[^a-z0-9]+", s) if x and x not in CO_STOP]
    return " ".join(w[:3])


def sm_rate(num, den, prior, a=10):
    return (num + a * prior) / (den + a)


def tlog(t, *a):
    print(*a, round(time.time() - t), flush=True)


# ---------------------------------------------------------------- 기초 테이블
def load_kr(ks5):
    """KSIC ks5 KR 출원(제공 데이터) + 등록원부 청구항 + ETSI KR 최초 반영일."""
    p = os.path.join(CACHE, f"kr_{ks5}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    L = pd.read_parquet(os.path.join(EXT, "trade_etsi_sep_link.parquet"))
    L["refl"] = L.first_refl.dt.strftime("%Y%m%d").astype(float)
    Lkr = L[L.cc == "KR"].groupby("appl_no").refl.min()
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"),
                        columns=["출원번호", "출원연도", "등록번호", "공개번호", "공개일자", "등록일자", "출원인명", "메인IPC", "세세분류코드",
                                 "발명명칭", "우선권국가"])
    k.columns = ["appl_no", "ay", "rg", "pb", "pub", "rgd", "names", "ipc", "ksic", "title", "prio_cc"]
    k = k[k.ksic.astype(str) == ks5].copy()
    k["appl_no"] = k.appl_no.astype(str)
    for c in ["pub", "rgd"]:
        k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    k["known"] = k[["pub", "rgd"]].min(axis=1)
    k["app"] = k.names.fillna("").str.split("|").str[0].map(norm)
    k["mg"] = k.ipc.fillna("").str.split("/").str[0].str.replace(" ", "")
    k["sg"] = k.ipc.fillna("").str.replace(" ", "")
    k["refl"] = k.appl_no.map(Lkr)
    # 출원 시점에 정해지는 속성: PCT 국내단계(출원번호 일련 7xxxxxx), 외국 우선권
    k["pct"] = (k.appl_no.str[6] == "7").astype(float)
    pc = k.prio_cc.fillna("")
    k["fprio"] = pc.map(lambda s: float(any(x and x != "KR" for x in s.split("|"))))
    c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
    cl = pd.read_sql(f"select applNo appl_no, claimCount from patent where ksic='{ks5}'", c)
    c.close()
    k = k.merge(cl.drop_duplicates("appl_no").assign(claimCount=lambda d: pd.to_numeric(d.claimCount, errors="coerce")), on="appl_no", how="left")
    k = k.drop(columns=["prio_cc"])
    k.to_parquet(p, index=False)
    return k


def load_family(k, ks5):
    """제공 US·EP·JP 기록(family에 KR 토큰) ↔ KR 출원 연결쌍 (파일럿과 동일 규칙)."""
    p = os.path.join(CACHE, f"fam_{ks5}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    L = pd.read_parquet(os.path.join(EXT, "trade_etsi_sep_link.parquet"))
    L["refl"] = L.first_refl.dt.strftime("%Y%m%d").astype(float)
    Lfo = L[L.cc.isin(["US", "EP", "JP"])].copy()
    Lfo["fkey"] = Lfo.cc + Lfo.appl_no.astype(str).str.split(".").str[0]
    Lfo = Lfo.groupby("fkey").refl.min()
    tok = {}
    for c, f in [("appl_no", lambda s: "KR" + s), ("rg", lambda s: "KR" + s.str[:9]), ("pb", lambda s: "KR" + s.str[2:])]:
        s = k[c].fillna("").astype(str).str.replace(r"\D", "", regex=True)
        ok = s.str.len() >= 9
        tok.update(dict(zip(f(s[ok]), k.appl_no[ok])))
    meta = os.path.join(MC, "_shared", "meta", "*.parquet").replace("\\", "/")
    fo = duckdb.sql(f"select distinct cc, appl_no, pub_date, family from read_parquet('{meta}') where cc in ('US','EP','JP') and ksic='{ks5}' and family like '%KR%'").df()
    fo["fkey"] = fo.cc + fo.appl_no.astype(str)
    fo["fknown"] = pd.to_numeric(fo.pub_date, errors="coerce")
    fo["frefl"] = fo.fkey.map(Lfo)
    fo["tok"] = fo.family.fillna("").str.split("|")
    ex = fo[["fkey", "cc", "fknown", "frefl", "tok"]].explode("tok")
    ex = ex[ex.tok.str.startswith("KR", na=False)]
    ex["kr"] = ex.tok.map(tok)
    ex = ex.dropna(subset=["kr"]).drop_duplicates(["fkey", "kr"]).drop(columns=["tok"])
    ex.to_parquet(p, index=False)
    return ex


def load_ext(ks5):
    """단위에 붙는 외부데이터(RoS·NTIS·OpenAlex 메인그룹·ETSI 글로벌 기업 선언)."""
    p = os.path.join(CACHE, f"ext_{ks5}.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    E = {}
    # RoS: KR 출원 → family US 특허 → 논문 인용 수(confscore≥3), US 발행연도
    ros = duckdb.sql(f"""select f.appl_no, p.issue_year, p.n_cit from read_parquet('{os.path.join(EXT, 'r3ros_family_us_pairs.parquet').replace(chr(92), '/')}') f
        join read_parquet('{os.path.join(EXT, 'r3ros_patent_sci.parquet').replace(chr(92), '/')}') p on f.us_pn = p.us_pn
        where f.cc='KR' and f.ksic='{ks5}'""").df()
    E["ros"] = ros
    # NTIS 성과 특허(연결 레코드 Year, 수행주체 그룹)
    nt = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["cc", "appl_no", "ksic", "Year", "perform_grp"])
    nt = nt[(nt.cc == "KR") & (nt.ksic == ks5)].copy()
    nt["Year"] = pd.to_numeric(nt.Year, errors="coerce")
    E["ntis"] = nt[["appl_no", "Year", "perform_grp"]]
    # OpenAlex 메인그룹 연도별 논문 수(발행연도 ≤ t0로 자름)
    oa = pd.read_parquet(os.path.join(EXT, "r3oa_mg_year_ALL.parquet"), columns=["mg", "year", "n_works"])
    oa["mg"] = oa.mg.str.replace(" ", "")
    E["oa"] = oa
    # ETSI 전체 선언 특허(전 국가) — 기업별 연도별 최초 반영 수
    s = pd.read_parquet(os.path.join(EXT, "trade_etsi_sep_patents.parquet"), columns=["company", "first_refl", "g5"])
    s["co"] = s.company.map(co_norm)
    s["ry"] = s.first_refl.dt.year
    E["etsi_co_year"] = s.groupby(["co", "ry"]).size().rename("n").reset_index()
    L = pd.read_parquet(os.path.join(EXT, "trade_etsi_sep_link.parquet"), columns=["cc", "appl_no", "first_refl", "company"])
    L = L[L.cc == "KR"].copy()
    L["refl"] = L.first_refl.dt.strftime("%Y%m%d").astype(float)
    L["co"] = L.company.map(co_norm)
    E["kr_decl_co"] = L[["appl_no", "refl", "co"]]
    pd.to_pickle(E, p)
    return E


# ---------------------------------------------------------------- 원점 패널
def build_panel(t0, ks5=KS_MAIN, mode="dev", k=None, ex=None, E=None, title_feats=True):
    """원점 t0의 단위표. mode: dev(레이블, t0+2<F만) / final(t0=2022 레이블, 🔒) / nolabel(레이블 없음)."""
    if mode == "dev":
        assert t0 + 2 < F_YEAR, "dev 모드는 결과 구간이 F 이전에 끝나는 원점만"
    elif mode == "final":
        assert t0 == T0_FINAL
    elif mode == "postfinal":  # future 모드 학습용: 최종시험 채점이 끝난 뒤에만 t0=2021·2022 레이블 허용
        assert os.path.exists(os.path.join(OUT, "final_result_r4.json")), "최종시험 채점 전에는 postfinal 레이블 금지"
    k = load_kr(ks5) if k is None else k
    ex = load_family(k, ks5) if ex is None else ex
    E = load_ext(ks5) if E is None else E
    end = t0 * 10000 + 1231
    kn = k[k.known <= end]
    dec = kn.refl <= end
    prior = dec.mean()
    ga = kn.assign(d=dec).groupby("app").agg(n=("d", "size"), d=("d", "sum"))
    gm = kn.assign(d=dec).groupby("mg").agg(n=("d", "size"), d=("d", "sum"))
    gs = kn.assign(d=dec).groupby("sg").agg(n=("d", "size"), d=("d", "sum"))
    gam = kn.assign(d=dec).groupby(["app", "mg"]).agg(n=("d", "size"), d=("d", "sum"))
    rec = kn[(kn.refl > (t0 - 2) * 10000 + 1231) & (kn.refl <= end)]
    U = kn[kn.ay.between(t0 - 5, t0) & ~dec].copy()
    # ---- M0 (파일럿 기준선)
    U["app_age"] = t0 - U.ay
    U["reg_t0"] = (U.rgd <= end).astype(float)
    U["claims_t0"] = U.claimCount.where(U.rgd <= end)
    U["app_rate"] = sm_rate(U.app.map(ga.d).fillna(0), U.app.map(ga.n).fillna(0), prior)
    U["app_logn"] = np.log1p(U.app.map(ga.n).fillna(0))
    U["mg_rate"] = sm_rate(U.mg.map(gm.d).fillna(0), U.mg.map(gm.n).fillna(0), prior)
    U["mg_logn"] = np.log1p(U.mg.map(gm.n).fillna(0))
    # ---- M1 5개국 패밀리(t0까지 공개된 해외 기록)
    e = ex[ex.fknown <= end]
    for cc in ["US", "EP", "JP"]:
        U["fam_" + cc] = U.appl_no.isin(set(e.loc[e.cc == cc, "kr"])).astype(float)
    U["fam_n"] = U.appl_no.map(e.groupby("kr").fkey.nunique()).fillna(0)
    # ---- 파일럿 M2(참고)
    U["fam_declared"] = U.appl_no.isin(set(ex.loc[ex.frefl <= end, "kr"])).astype(float)
    U["app_recent_dec"] = np.log1p(U.app.map(rec.groupby("app").size()).fillna(0))
    U["mg_recent_dec"] = np.log1p(U.mg.map(rec.groupby("mg").size()).fillna(0))
    # ---- X_ros: family US 특허(발행 ≤ t0)의 논문 인용 (Reliance on Science)
    r = E["ros"][E["ros"].issue_year <= t0]
    rg = r.groupby("appl_no").agg(n_us=("n_cit", "size"), n_cit=("n_cit", "sum"), n_sci=("n_cit", lambda x: (x > 0).sum()))
    U["ros_us_issued"] = np.log1p(U.appl_no.map(rg.n_us).fillna(0))
    U["ros_ncit"] = np.log1p(U.appl_no.map(rg.n_cit).fillna(0))
    U["ros_sci"] = (U.appl_no.map(rg.n_sci).fillna(0) > 0).astype(float)
    # ---- X_ntis: 정부 R&D 성과 연결(레코드 Year ≤ t0), 수행주체
    nt = E["ntis"][E["ntis"].Year <= t0]
    U["ntis_gov"] = U.appl_no.isin(set(nt.appl_no)).astype(float)
    U["ntis_gri"] = U.appl_no.isin(set(nt.loc[nt.perform_grp.isin(["출연연", "국공립연"]), "appl_no"])).astype(float)
    U["ntis_firm"] = U.appl_no.isin(set(nt.loc[nt.perform_grp == "기업", "appl_no"])).astype(float)
    U["ntis_univ"] = U.appl_no.isin(set(nt.loc[nt.perform_grp == "대학", "appl_no"])).astype(float)
    # ---- X_oa: 메인그룹 OpenAlex 논문(발행연도 ≤ t0)
    oa = E["oa"]
    w3 = oa[oa.year.between(t0 - 2, t0)].groupby("mg").n_works.sum()
    p3 = oa[oa.year.between(t0 - 5, t0 - 3)].groupby("mg").n_works.sum()
    has = U.mg.isin(set(oa.mg))
    U["oa_lv3"] = np.log1p(U.mg.map(w3)).where(has)
    U["oa_g3"] = (np.log1p(U.mg.map(w3)) - np.log1p(U.mg.map(p3))).where(has)
    # ---- X_etsig: 출원인 → (t0까지의 KR 선언에서 배운) ETSI 기업명 → 전 세계 선언 동향
    kd = E["kr_decl_co"][E["kr_decl_co"].refl <= end].merge(k[["appl_no", "app"]], on="appl_no")
    amap = kd.groupby("app").co.agg(lambda x: list(pd.unique(x)))
    cy = E["etsi_co_year"]
    rec2 = cy[(cy.ry > t0 - 2) & (cy.ry <= t0)].groupby("co").n.sum()
    prev2 = cy[(cy.ry > t0 - 4) & (cy.ry <= t0 - 2)].groupby("co").n.sum()
    tot = cy[cy.ry <= t0].groupby("co").n.sum()
    def agg(app, ser):
        cos = amap.get(app)
        return float(sum(ser.get(c, 0) for c in cos)) if cos else 0.0
    ua = pd.Series(U.app.unique())
    m_rec = dict(zip(ua, ua.map(lambda a: agg(a, rec2))))
    m_prev = dict(zip(ua, ua.map(lambda a: agg(a, prev2))))
    m_tot = dict(zip(ua, ua.map(lambda a: agg(a, tot))))
    U["etsi_co_rec2"] = np.log1p(U.app.map(m_rec))
    U["etsi_co_tot"] = np.log1p(U.app.map(m_tot))
    U["etsi_co_mom"] = np.log1p(U.app.map(m_rec)) - np.log1p(U.app.map(m_prev))
    # ---- X_regime: 출원인 선언 시기 관행(t0까지 반영분)
    dk = kn[dec].copy()
    dk["lag"] = (dk.refl // 10000) - dk.ay
    U["app_lag_med"] = U.app.map(dk.groupby("app").lag.median())
    U["app_rec_rate"] = sm_rate(U.app.map(rec.groupby("app").size()).fillna(0), U.app.map(ga.n).fillna(0), len(rec) / max(1, len(kn)))
    U["mg_rec_rate"] = sm_rate(U.mg.map(rec.groupby("mg").size()).fillna(0), U.mg.map(gm.n).fillna(0), len(rec) / max(1, len(kn)))
    U["age_minus_lag"] = U.app_age - U.app_lag_med
    # ---- X_famstrat: 출원인의 과거 해외 패밀리 전략(t0까지 공개된 해외 기록)
    kn_f = kn.appl_no.isin(set(e.kr)).astype(float)
    gf = kn.assign(f=kn_f).groupby("app").f.agg(["sum", "size"])
    U["app_fam_share"] = sm_rate(U.app.map(gf["sum"]).fillna(0), U.app.map(gf["size"]).fillna(0), kn_f.mean())
    # ---- X_fine: IPC 서브그룹·출원인×메인그룹 선언률(계층 평활)
    U["sg_rate"] = sm_rate(U.sg.map(gs.d).fillna(0), U.sg.map(gs.n).fillna(0), U.mg_rate)
    U["sg_logn"] = np.log1p(U.sg.map(gs.n).fillna(0))
    idx = pd.MultiIndex.from_arrays([U.app, U.mg])
    U["appmg_rate"] = sm_rate(pd.Series(gam.d.reindex(idx).values, index=U.index).fillna(0),
                              pd.Series(gam.n.reindex(idx).values, index=U.index).fillna(0), U.app_rate)
    # ---- X_title: 최근(t0-3, t0] 선언 특허 제목과의 유사도(제공 데이터 텍스트, 로컬 계산)
    if title_feats:
        from sklearn.feature_extraction.text import TfidfVectorizer
        pool = kn[(kn.refl > (t0 - 3) * 10000 + 1231) & (kn.refl <= end)]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=3, sublinear_tf=True, dtype=np.float32)
        vec.fit(kn.title.fillna(""))
        P = vec.transform(pool.title.fillna(""))
        X = vec.transform(U.title.fillna(""))
        smax = np.zeros(X.shape[0], np.float32); stop = np.zeros(X.shape[0], np.float32)
        if P.shape[0] > 0:
            kk = min(10, P.shape[0])
            for i in range(0, X.shape[0], 2000):
                S = (X[i:i + 2000] @ P.T).toarray()
                smax[i:i + 2000] = S.max(1)
                stop[i:i + 2000] = -np.sort(-S, axis=1)[:, :kk].mean(1)
        U["title_sim_max"] = smax
        U["title_sim_top10"] = stop
    # ---- 결과
    if mode in ("dev", "final", "postfinal"):
        U["y"] = ((U.refl > end) & (U.refl <= (t0 + 2) * 10000 + 1231)).astype(int)
    else:
        U["y"] = -1
    U["t0"] = t0
    U["ks"] = ks5
    return U.drop(columns=["names", "rg", "pb", "title", "refl", "ipc"])


FEATS = {
    "M0": ["app_age", "reg_t0", "claims_t0", "app_rate", "app_logn", "mg_rate", "mg_logn"],
    "M1": ["fam_US", "fam_EP", "fam_JP", "fam_n"],
    "M2p": ["fam_declared", "app_recent_dec", "mg_recent_dec"],
    "ros": ["ros_us_issued", "ros_ncit", "ros_sci"],
    "ntis": ["ntis_gov", "ntis_gri", "ntis_firm", "ntis_univ"],
    "oa": ["oa_lv3", "oa_g3"],
    "etsig": ["etsi_co_rec2", "etsi_co_tot", "etsi_co_mom"],
    "regime": ["app_lag_med", "app_rec_rate", "mg_rec_rate", "age_minus_lag"],
    "famstrat": ["app_fam_share"],
    "sg": ["sg_rate", "sg_logn"],
    "appmg": ["appmg_rate"],
    "title": ["title_sim_max", "title_sim_top10"],
    "pct": ["pct", "fprio"],
}

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, n_jobs=2, tree_method="hist", eval_metric="logloss", verbosity=0)


def fit_pred(Xtr, ytr, Xte, model="xgb", seeds=(0,), w=None):
    if model == "xgb":
        from xgboost import XGBClassifier
        out = np.zeros(len(Xte))
        for s in seeds:
            m = XGBClassifier(random_state=s, **XGB_PARAMS)
            m.fit(Xtr, ytr, sample_weight=w)
            out += m.predict_proba(Xte)[:, 1]
        return out / len(seeds)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(Xtr, ytr, logisticregression__sample_weight=w)
    return m.predict_proba(Xte)[:, 1]


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return np.nan if y.min() == y.max() else roc_auc_score(y, p)


def skill(a, b):
    return (a - b) / (1 - b)


def cluster_boot_ci(y, pm, pb, g, n=500, seed=0):
    """출원인 군집 부트스트랩(군집을 복원추출)으로 skill 95% CI."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    codes, inv = np.unique(np.asarray(g), return_inverse=True)
    idx_by = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).values
    G = len(idx_by); out = []
    for _ in range(n):
        pick = rng.integers(0, G, G)
        i = np.concatenate([idx_by[j] for j in pick])
        if y[i].min() == y[i].max():
            continue
        a1 = roc_auc_score(y[i], pm[i]); a0 = roc_auc_score(y[i], pb[i])
        out.append(skill(a1, a0))
    return np.percentile(out, [2.5, 97.5]), float(np.mean(out))
