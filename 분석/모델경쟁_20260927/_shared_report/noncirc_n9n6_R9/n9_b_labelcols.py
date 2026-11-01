# -*- coding: utf-8 -*-
"""N9 단계 B: 결과로 학습하는 열을 새 결과로 다시 만든다(동결 n9_build·n9_textclf와 같은 방식, 모든 팔 공통).
ksic_prior_b1/top: 과제연도 v-5..v-3 평활 양성률((양성+1)/(n+20)), 대응 KSIC = 공개 해설서 텍스트 대응(b1_top, top_ksic).
txt_clf: 고정 벡터라이저(N9 cache/txt_vec.pkl, 2007–2013 과제 텍스트, 레이블 미사용) + SGD 로지스틱, 학습 t<=v-2, 학습행은 과제연도 묶음 5겹 OOF.
사용: python n9_b_labelcols.py OUTCOME [OUTCOME ...]   (OUTCOME = y_frozen|y_all|y_G|...)
y_frozen은 동결 캐시와 대조(복제 점검)만 한다.
산출: cache/n9_lab_{OUTCOME}.parquet (ksic_prior 2열, 패널 행 순서), cache/n9_txtclf_{OUTCOME}_{v}.parquet (row_id, txt_clf)"""
import sys, os, pickle
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r9_common import *
from sklearn.linear_model import SGDClassifier

GAP = 2; ORIG = list(range(2014, 2021))
names = sys.argv[1:]
D = pd.read_parquet(os.path.join(N9, "cache", "panel_dev.parquet"), columns=["pjt_no", "t0", "b1_top", "top_ksic", "ksic_prior_b1", "ksic_prior_top"])
Y = pd.read_parquet(os.path.join(CACHE, "n9_outcomes_panel.parquet"))
assert (Y.pjt_no.values == D.pjt_no.values).all()
P = pd.read_parquet(os.path.join(CACHE, "n9_outcomes_proj.parquet"))
vec = pickle.load(open(os.path.join(N9, "cache", "txt_vec.pkl"), "rb"))
PT = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["pjt_no", "title_ko", "kw_ko"]).drop_duplicates("pjt_no").set_index("pjt_no")
T = vec.transform((PT.title_ko.fillna("") + " " + PT.kw_ko.fillna("").str.replace(",", " ")).reindex(D.pjt_no.values).fillna("").values)
del PT
log("tfidf", T.shape)
t0 = D.t0.values


def txtclf(yv, v):
    tr = np.where(t0 <= v - GAP)[0]; te = np.where(t0 == v)[0]
    assert t0[tr].max() + 1 <= DEV_MAX_APPL
    clf = lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=None, random_state=0)
    s_te = clf().fit(T[tr], yv[tr]).decision_function(T[te])
    yrs = np.unique(t0[tr]); fold_of = {yy: i % 5 for i, yy in enumerate(yrs)}; f = np.array([fold_of[yy] for yy in t0[tr]]); s_tr = np.zeros(len(tr))
    for k in range(5):
        a = tr[f != k]; b = f == k
        if b.sum():
            s_tr[b] = clf().fit(T[a], yv[a]).decision_function(T[tr[b]])
    return pd.DataFrame({"row_id": np.r_[tr, te], "txt_clf": np.r_[s_tr, s_te].astype("float32")})


for nm in names:
    if nm == "y_frozen":
        tc = pd.read_parquet(os.path.join(N9, "cache", "txtclf_dev_2014.parquet"))
        chk = txtclf(Y.y_frozen.values, 2014)
        ok = np.allclose(chk.txt_clf.values, tc.txt_clf.values, atol=1e-3) and (chk.row_id.values == tc.row_id.values).all()
        log("txt_clf 복제(y_frozen, 2014):", ok); continue
    ycol = nm
    out = {}
    for v in sorted(D.t0.unique()):
        pr = P[P.year.between(v - 5, v - 3) & P[ycol].notna()]
        m = (D.t0 == v).values
        for col, cn in (("b1_top", "ksic_prior_b1"), ("top_ksic", "ksic_prior_top")):
            g = pr.groupby(col)[ycol].agg(["sum", "count"]); rate = (g["sum"] + 1) / (g["count"] + 20)
            out.setdefault(cn, np.full(len(D), np.nan))[m] = D.loc[m, col].map(rate).values
    pd.DataFrame(out).to_parquet(os.path.join(CACHE, f"n9_lab_{nm}.parquet"), index=False)
    yv = Y[ycol].values
    for v in ORIG:
        if os.path.exists(os.path.join(CACHE, f"n9_txtclf_{nm}_{v}.parquet")):   # 이어하기(시간 제한으로 끊긴 경우)
            continue
        txtclf(yv, v).to_parquet(os.path.join(CACHE, f"n9_txtclf_{nm}_{v}.parquet"), index=False)
        log(nm, v)
# ksic_prior 복제 점검(동결 y로 같은 코드 → 패널 값)
if "check" in names or not os.path.exists(os.path.join(OUT, "n9_labcheck.json")):
    Pf = P.copy()
    L = pd.read_parquet(os.path.join(EXT, "ntis_patent_link.parquet"), columns=["appl_no", "ProjectID", "appl_year"]); L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
    L = L[L.appl_year <= DEV_MAX_APPL].drop_duplicates(["ProjectID", "appl_no"])
    d = L.assign(py=L.ProjectID.map(Pf.set_index("pjt_no").year)).dropna(subset=["py"]); pos = set(d[(d.appl_year >= d.py) & (d.appl_year <= d.py + 1)].ProjectID)
    Pf["yr"] = np.where(Pf.year + 1 <= DEV_MAX_APPL, Pf.pjt_no.isin(pos).astype(float), np.nan)
    mx = 0.0
    for v in sorted(D.t0.unique()):
        pr = Pf[Pf.year.between(v - 5, v - 3) & Pf.yr.notna()]; m = (D.t0 == v).values
        for col, cn in (("b1_top", "ksic_prior_b1"), ("top_ksic", "ksic_prior_top")):
            g = pr.groupby(col).yr.agg(["sum", "count"]); rate = (g["sum"] + 1) / (g["count"] + 20)
            a = D.loc[m, col].map(rate).values.astype(float); b = D.loc[m, cn].astype(float).values
            mx = max(mx, float(np.nanmax(np.abs(a - b))))
    json.dump({"ksic_prior_replication_max_abs_diff": mx}, open(os.path.join(OUT, "n9_labcheck.json"), "w"))
    log("ksic_prior replication max diff", mx)
log("end")
