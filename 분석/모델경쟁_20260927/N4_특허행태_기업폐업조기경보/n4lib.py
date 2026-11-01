# -*- coding: utf-8 -*-
r"""N4 (I2b) 특허 행태로 보는 기업 폐업 조기경보 — 공용 라이브러리.
출발점: _ideas_R3\i00_common.py·i01_nps.py·i02_patlink.py·i09_nts_exit.py (파일럿 설계 그대로).
봉인(🔒): 최종시험 t0=2023 → 폐업 2024-01-01~2025-12-31 (F=2024).
  - dev 모드: 원점 t0 ≤ 2021(결과창 끝 ≤ 2023-12-31)만 레이블 생성. NPS 스냅샷 ≤ 2021-12만 읽는다.
  - final 모드: 학습 t0 ≤ 2021 + 시험 t0=2023. NPS ≤ 2023-12.
  - future 모드: 학습 t0 ≤ 2023(폐업 ≤2025), 예측 t0=2024(폐업 2025–26, 레이블 없음). NPS ≤ 2024-12.
모든 파일 쓰기는 이 트랙 폴더(OUT) 안에서만 한다. 원본 DB는 sqlite mode=ro."""
import os, re, sqlite3, time
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
OUT = os.path.join(MC, r"N4_특허행태_기업폐업조기경보")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
REG = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
BRS = os.path.join(BASE, "수집파이프라인", "business_rights.sqlite")

F_YEAR = 2024                     # 최종시험 첫 결과연도
DEV_ORIGINS = [2017, 2018, 2019, 2020, 2021]
TRAIN_MIN_T0 = 2015
FINAL_T0 = 2023
FUTURE_T0 = 2024   # 등록원부가 2025-08-22까지라 2025년 말 원점은 창이 잘린다 → 2024년 말 원점으로 2025–26 폐업 예측
MODE_SNAP_MAX = {"dev": "2021-12", "final": "2023-12", "future": "2024-12"}
ALL_SNAPS = ["2015-12", "2016-12", "2017-12", "2018-12", "2019-12", "2020-12", "2021-12", "2022-12", "2023-12", "2024-12", "2025-12"]

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
    """결과창 (t0, t0+2]. dev: 결과창 끝이 F 이전이어야 한다. final: 학습(≤2021)+시험 2023. future: 학습 ≤2023."""
    if mode == "dev":
        return t0 + 2 < F_YEAR
    if mode == "final":
        return t0 + 2 < F_YEAR or t0 == FINAL_T0
    if mode == "future":
        return t0 <= FINAL_T0
    raise ValueError(mode)


def snaps_for(mode):
    mx = MODE_SNAP_MAX[mode]
    return [s for s in ALL_SNAPS if s <= mx]


# ---------------- 평가 ----------------
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p)


def prauc(y, p):
    from sklearn.metrics import average_precision_score
    return average_precision_score(np.asarray(y), np.asarray(p))


def skill(a, b):
    return (a - b) / (1 - b)


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
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    if sw is None:
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr, logisticregression__sample_weight=np.asarray(sw))
    return m.predict_proba(Xte)[:, 1]


def rank01(p):
    return pd.Series(p).rank(pct=True).values


def predict(Xtr, ytr, Xte, model="xgb", seeds=(0,), sw=None):
    """model: xgb(시드 평균) | logit | ens(xgb 시드평균 순위 + 로지스틱 순위의 평균). sw: 표본 가중치(선택)"""
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
    cl = pd.Series(cl).astype(str).values
    u, inv = np.unique(cl, return_inverse=True)
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


# ---------------- 원자료 로드(캐시) ----------------
def load_registry():
    """등록원부 파생표(원점 무관, 날짜 컬럼으로 원점마다 자름). 캐시: cache/reg_*.parquet"""
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


def load_nts():
    ns = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "b_stt_cd", "end_dt", "registered"])
    ns = ns[ns.registered.fillna(False).astype(bool)].drop_duplicates("applicantCd")
    ns["end"] = pd.to_numeric(ns.end_dt.dt.strftime("%Y%m%d"), errors="coerce")
    ns["corp_type"] = pd.to_numeric(ns.bizno.str[3:5], errors="coerce")
    return ns


def load_nps(mode):
    """NPS 기업키 스냅샷(모드별 상한까지만). 캐시가 없으면 원본에서 만든다(i01_nps.py 그대로 + 탈퇴 정보)."""
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
    """i02_patlink.py 재현: 제공 KR 출원 → 출원인코드(등록원부 정확 코드 + CORP_APPLICANT 유일 상호)."""
    fc = os.path.join(CACHE, "app_firm.parquet")
    if os.path.exists(fc):
        return pd.read_parquet(fc)
    rows = []
    with open(os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("¶")
            if len(p) >= 2:
                rows.append((p[0].strip(), p[1].strip()))
    ca = pd.DataFrame(rows, columns=["applicantCd", "aname"])
    ca["n"] = ca.aname.map(norm)
    ca = ca[ca.n.str.len() >= 2]
    amb = ca.groupby("n").applicantCd.nunique()
    ca = ca[ca.n.map(amb) == 1].drop_duplicates("n")
    n2c = dict(zip(ca.n, ca.applicantCd))
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"),
                        columns=["출원번호", "출원일", "공개일자", "등록일자", "출원인명", "세세분류코드"])
    k.columns = ["appl_no", "appl_date", "pub_date", "rgst_date", "names", "ksic"]
    k["ksic"] = k.ksic.astype(str)
    k["nm"] = k.names.fillna("").str.split("|")
    k = k.explode("nm")
    k["applicantCd"] = k.nm.map(norm).map(n2c)
    kn = k.dropna(subset=["applicantCd"])[["appl_no", "appl_date", "pub_date", "rgst_date", "ksic", "applicantCd"]].copy()
    kn["src"] = "name"
    c = ro(REG)
    rp = pd.read_sql("select p.applNo appl_no, p.applDate appl_date, p.opnDate opn, p.pubDate pub, p.rgstDate rgst_date, p.ksic, a.applicantCd "
                     "from patent p join applicant a on p.rgst_no=a.rgst_no", c)
    for _c in ["opn", "pub"]:
        rp[_c] = pd.to_numeric(rp[_c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    rp["pub_date"] = rp[["opn", "pub"]].min(axis=1)
    rp = rp[["appl_no", "appl_date", "pub_date", "rgst_date", "ksic", "applicantCd"]]
    rp["src"] = "registry"
    allp = pd.concat([rp, kn], ignore_index=True)
    for col in ["appl_date", "pub_date", "rgst_date"]:
        allp[col] = pd.to_numeric(allp[col].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
    allp["known_date"] = allp[["pub_date", "rgst_date"]].min(axis=1)
    allp = allp.sort_values("src", ascending=False).drop_duplicates(["appl_no", "applicantCd", "ksic"])
    allp["ay"] = (allp.appl_date // 10000).astype("Int64")
    allp = allp[["applicantCd", "appl_no", "appl_date", "ay", "known_date", "ksic", "src"]]
    allp.to_parquet(fc, index=False)
    return allp


def a2key():
    lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
    lk = lk[lk.nps_hit]
    lk["key"] = lk.biz6 + "|" + lk.nname
    return dict(zip(lk.applicantCd, lk.key))


# ---------------- 기본 패널(M0·M1) — i09_nts_exit.py와 같은 정의 ----------------
M0 = ["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type"]
M1_ADD = ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "lapse_1y", "lapse_2y",
          "pay_1y", "pay_trend", "trf_2y", "live_ratio"]


def build_base(t0, mode, AF, R, ns, NPS, A2K):
    """원점 t0의 모집단·M0·M1·레이블. 레이블은 label_allowed일 때만 만든다(아니면 y=NaN)."""
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
    # NPS (원점 t0 연말 스냅샷, 직전 연말)
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
    if label_allowed(t0, mode):
        f["y"] = ((f.end > end) & (f.end <= (t0 + 2) * 10000 + 1231)).astype(float)
    else:
        f["y"] = np.nan
    f = f.drop(columns=["end"])
    f["t0"] = t0
    return f.reset_index().rename(columns={"index": "applicantCd"})

def load_registry_extra():
    """등록원부 권리 처분 사건(파일럿의 trf_2y는 right 테이블에 '이전' 사건이 없어 전부 0이었다 → owner 테이블로 수정).
    own: 권리자 변경(양도·합병·분할 등) 사건일, sur: 권리의 (일부)말소등록 중 포기 사건일. 캐시 cache/reg_own.parquet, reg_sur.parquet"""
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



def load_foreign_events(AF):
    """제공 5개국 메타의 US·EP·JP·CN 기록 중 우선권국 KR인 것 → family 토큰(KR 출원·등록·공개번호) → 제공 KR 출원 → 출원인코드.
    파일럿 i05_global_build.py 방식(+CN). known = 해외 공개일. 캐시 cache/foreign_events.parquet"""
    fc = os.path.join(CACHE, "foreign_events.parquet")
    if os.path.exists(fc):
        return pd.read_parquet(fc)
    import duckdb
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "등록번호", "공개번호"])
    k.columns = ["appl_no", "rg", "pb"]
    tok = []
    for c in ["appl_no", "rg", "pb"]:
        s_ = k[c].fillna("").astype(str).str.replace(r"\D", "", regex=True)
        v = ("KR" + s_.str[:9]) if c == "rg" else (("KR" + s_.str[2:]) if c == "pb" else ("KR" + s_))
        tok.append(pd.DataFrame({"tok": v, "appl_no": k.appl_no.astype(str)})[s_.str.len() >= 9])
    tok = pd.concat(tok).drop_duplicates("tok")
    tmap = dict(zip(tok.tok, tok.appl_no))
    meta = os.path.join(MC, "_shared", "meta", "*.parquet").replace("\\", "/")
    fo = duckdb.sql(f"""select distinct cc, appl_no, pub_date, prio_date, family from read_parquet('{meta}')
                     where cc in ('US','EP','JP','CN') and prio_cc like '%KR%'""").df()
    fo["fkey"] = fo.cc + fo.appl_no.astype(str)
    fo["prio"] = fo.prio_date.fillna("").str.split("|").map(lambda L: min([int(x) for x in L if x.isdigit() and len(x) == 8], default=np.nan))
    fo["known"] = pd.to_numeric(fo.pub_date, errors="coerce")
    fo["tok"] = fo.family.fillna("").str.split("|")
    ex = fo[["fkey", "prio", "known", "tok"]].explode("tok")
    ex = ex[ex.tok.str.startswith("KR", na=False)]
    ex["kr_appl"] = ex.tok.map(tmap)
    ex = ex.dropna(subset=["kr_appl"]).drop_duplicates(["fkey", "kr_appl"])
    A = AF[["appl_no", "applicantCd"]].drop_duplicates().copy(); A["appl_no"] = A.appl_no.astype(str)
    ev = ex.merge(A, left_on="kr_appl", right_on="appl_no")[["applicantCd", "fkey", "kr_appl", "prio", "known"]]
    ev.to_parquet(fc, index=False)
    return ev
