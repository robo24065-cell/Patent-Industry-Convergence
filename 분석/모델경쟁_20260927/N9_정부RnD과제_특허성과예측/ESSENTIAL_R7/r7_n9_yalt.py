# -*- coding: utf-8 -*-
"""N9 정의 시험(결과변수 대체): 제공 데이터 없이 '50개 산업 특허 성과'를 정의하면?
y_alt = 과제 성과 중 NTIS KR 특허 '출원' 레코드(출원연도 ∈ [t, t+1] <= 2021)의 첫 IPC 연계표 배정(free)이 25개 소분류 안인 것 >= 1.
(1) 과제 단위 y vs y_alt 일치(기저율, 자카드, 정밀도·재현율)
(2) '연계표 세계' 모형: 결과로 학습하는 열(ksic_prior_b1·ksic_prior_top·txt_clf)을 y_alt로 다시 만들고 동결 챔피언 설정으로 y_alt에 학습 →
    실제 결과 y(제공 50산업 특허 성과)로 채점. 기준선 M0_L2는 y로 학습한 동결 정의 그대로(m0l2_dev.parquet).
    제공 피처 3묶음은 연계표 대체판(alt)을 사용(완전한 '기존 데이터만' 판), 제공판을 쓴 경우도 병기.
산출: yalt_dev.csv, yalt_overlap.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); N9 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, N9)
from r7_concord import assign, first_ipc, G50
from n9_lib import EXT, CACHE, DEV_ORIGINS, GAP, NEG_FRAC, fit_pred, auc, skill, topk, log
from r7_n9_run import load, VARS, FEATS, P1, P2, P3
from sklearn.linear_model import SGDClassifier

D = load()
R = pd.read_parquet(f"{EXT}/ntis_rpatent_raw.parquet", columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber", "ipc_codes"])
R["Year"] = pd.to_numeric(R.Year, errors="coerce")
R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.RegistType == "출원") & (R.Year <= 2021)].drop_duplicates(["ProjectID", "RegistNumber"])
g, _ = assign(R.ipc_codes.map(first_ipc).values, "free"); R = R[np.isin(g, G50)]
# 과제 표(2007–2020, 보안 제외) + 대응 KSIC (n9_build 그대로)
RAW = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "secret_flag"])
P = RAW[(RAW.year >= 2007) & (RAW.year <= 2020) & (RAW.secret_flag.astype(float).fillna(0) == 0)].copy(); P["year"] = P.year.astype(int)
km = pd.read_parquet(f"{EXT}/ntis_project_ksic_text_map.parquet"); P = P.merge(km.drop(columns=["year"]), on="pjt_no", how="left")
P["b2_dict_assigned"] = P.b2_dict_assigned.astype(float).fillna(0).astype(int)
P["b1_top"] = pd.to_numeric(P.b1_tfidf_top_ksic, errors="coerce"); P["b2_top"] = pd.to_numeric(P.b2_dict_top_ksic, errors="coerce")
P["top_ksic"] = P.b2_top.where(P.b2_dict_assigned == 1, P.b1_top)
L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "ProjectID", "appl_year"]); L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
L = L[L.appl_year <= 2021].drop_duplicates(["ProjectID", "appl_no"])
py = P.set_index("pjt_no").year
def ylab(df, yearcol):
    d = df.assign(py=df.ProjectID.map(py)).dropna(subset=["py"])
    return set(d[(d[yearcol] >= d.py) & (d[yearcol] <= d.py + 1)].ProjectID)
pos_rep = ylab(L, "appl_year"); pos_alt = ylab(R.rename(columns={"Year": "ay"}), "ay")
P["y_rep"] = np.where(P.year + 1 <= 2021, P.pjt_no.isin(pos_rep).astype(float), np.nan)
P["y_alt"] = np.where(P.year + 1 <= 2021, P.pjt_no.isin(pos_alt).astype(float), np.nan)
D["y_alt"] = D.pjt_no.map(P.drop_duplicates("pjt_no").set_index("pjt_no").y_alt).astype(int)
assert (D.y.values == D.pjt_no.map(P.drop_duplicates("pjt_no").set_index("pjt_no").y_rep).values).all(), "y 복제 실패"
ev = D[D.t0.isin(DEV_ORIGINS)]
tp = int(((ev.y == 1) & (ev.y_alt == 1)).sum())
ov = dict(n=int(len(ev)), base_y=float(ev.y.mean()), base_y_alt=float(ev.y_alt.mean()), jaccard=float(tp / ((ev.y == 1) | (ev.y_alt == 1)).sum()),
          precision_alt_vs_y=float(tp / ev.y_alt.sum()), recall_alt_vs_y=float(tp / ev.y.sum()), auc_of_yalt_as_score_for_y=float(auc(ev.y, ev.y_alt)))
log(json.dumps(ov))
json.dump(ov, open(os.path.join(HERE, "yalt_overlap.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# 결과 학습 열 재생성
def ksic_prior(ycol):
    out = {}
    for v in sorted(D.t0.unique()):
        pr = P[P.year.between(v - 5, v - 3) & P[ycol].notna()]
        X = D[D.t0 == v]
        for col, nm in (("b1_top", "ksic_prior_b1"), ("top_ksic", "ksic_prior_top")):
            gg = pr.groupby(col)[ycol].agg(["sum", "count"]); rate = (gg["sum"] + 1) / (gg["count"] + 20)
            out.setdefault(nm, []).append(pd.Series(X[col].map(rate).values, index=X.index))
    return {k: pd.concat(v).loc[D.index].values for k, v in out.items()}
kp_rep = ksic_prior("y_rep")
for k in kp_rep:
    assert np.allclose(kp_rep[k], D[k].astype(float).values, atol=1e-5, equal_nan=True), f"{k} 복제 실패"
kp_alt = ksic_prior("y_alt")
vec = pickle.load(open(f"{CACHE}/txt_vec.pkl", "rb"))
PT = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "title_ko", "kw_ko"]).drop_duplicates("pjt_no").set_index("pjt_no")
T = vec.transform((PT.title_ko.fillna("") + " " + PT.kw_ko.fillna("").str.replace(",", " ")).reindex(D.pjt_no.values).fillna("").values)
del PT
def txtclf(yv, v):
    t0 = D.t0.values; tr = np.where(t0 <= v - GAP)[0]; te = np.where(t0 == v)[0]
    clf = lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=None, random_state=0)
    s_te = clf().fit(T[tr], yv[tr]).decision_function(T[te])
    yrs = np.unique(t0[tr]); fold_of = {yy: i % 5 for i, yy in enumerate(yrs)}; f = np.array([fold_of[yy] for yy in t0[tr]]); s_tr = np.zeros(len(tr))
    for k in range(5):
        a = tr[f != k]; b = f == k
        if b.sum():
            s_tr[b] = clf().fit(T[a], yv[a]).decision_function(T[tr[b]])
    out = np.full(len(D), np.nan); out[tr] = s_tr; out[te] = s_te
    return out
M0 = pd.read_parquet(os.path.join(HERE, "m0l2_dev.parquet"))
ORIG = [int(x) for x in sys.argv[1:]] or DEV_ORIGINS
rows = []
for v in ORIG:
    tc = pd.read_parquet(f"{CACHE}/txtclf_dev_{v}.parquet")
    if v == DEV_ORIGINS[0]:
        chk = txtclf(D.y.values, v); assert np.allclose(chk[tc.row_id.values], tc.txt_clf.values, atol=1e-4), "txt_clf 복제 실패"
    ta = txtclf(D.y_alt.values, v)
    for nm, use_alt_feats in (("D_YALT_altfeats", True), ("D_YALT_provfeats", False)):
        X = D.copy(); X["txt_clf"] = ta
        for k in kp_alt:
            X[k] = kp_alt[k]
        if use_alt_feats:
            for c in P1 + P2 + P3:
                X[c] = X[f"{c}__alt"].values
        rng = np.random.default_rng(0); ylr = X.y_alt.values
        keep = (ylr == 1) | (rng.random(len(X)) < NEG_FRAC); w = np.where(ylr == 1, 1.0, 1.0 / NEG_FRAC)
        msk = (X.t0 >= 2010).values & (X.t0 <= v - GAP).values & keep
        tr = X[msk]; te = X[X.t0 == v]
        p = fit_pred(tr[FEATS].astype(float), tr.y_alt, te[FEATS].astype(float), w=w[msk])
        y = te.y.values; pb = M0[M0.t0 == v].p.values; a, a0 = auc(y, p), auc(y, pb)
        rows.append(dict(variant=nm, origin=v, auc=a, auc_M0L2=a0, skill=skill(a, a0), auc_on_yalt=auc(te.y_alt.values, p),
                         top1=topk(y, p, 0.01)[0], top10=topk(y, p, 0.10)[0], hit100=topk(y, p, k=100)[1]))
        log(nm, v, round(a, 4), round(skill(a, a0), 4))
pd.DataFrame(rows).to_csv(os.path.join(HERE, f"yalt_dev_{'_'.join(map(str, ORIG))}.csv"), index=False, encoding="utf-8-sig")
