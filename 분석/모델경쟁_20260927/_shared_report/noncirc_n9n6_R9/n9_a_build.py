# -*- coding: utf-8 -*-
"""N9 단계 A: (1) 라벨 무관 결과 Y_ALL, Y_SEC_{G,H,C,A,B} (NTIS KR 특허 출원 성과, 출원 ≤2021)
(2) TXT 판 피처: P2 중심점(k_txt 5자리, 제공 풀 2003–09 제목 전체), P3 org_krapp_topk(k_txt == top_ksic)
    + 같은 코드로 제공판(rep)을 재계산해 동결 패널과 대조(코드 경로 검증).
산출: cache/n9_outcomes_panel.parquet(패널 행 순서), cache/n9_outcomes_proj.parquet(과제표: ksic_prior용),
      cache/n9_txt_feats.parquet(패널 행 순서), n9_build_check.json"""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r9_common import *
sys.path.insert(0, N9E); sys.path.insert(0, N9)
from r7_concord import first_ipc
from n9_lib import norm_name
import duckdb

D = pd.read_parquet(os.path.join(N9, "cache", "panel_dev.parquet"), columns=["pjt_no", "t0", "y", "org", "org2", "top_ksic", "org_krapp_topk",
                                                                            "pt_max", "pt_top", "pt_margin", "pt_ent"])
assert D.t0.max() == 2020 and D.t0.min() == 2010
chk = {}
# ---------------- (1) 결과 ----------------
RAW = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["pjt_no", "year", "secret_flag"])
P = RAW[(RAW.year >= 2007) & (RAW.year <= 2020) & (RAW.secret_flag.astype(float).fillna(0) == 0)].copy(); P["year"] = P.year.astype(int)
km = pd.read_parquet(os.path.join(EXT, "ntis_project_ksic_text_map.parquet")); P = P.merge(km.drop(columns=["year"]), on="pjt_no", how="left")
P["b2_dict_assigned"] = P.b2_dict_assigned.astype(float).fillna(0).astype(int)
P["b1_top"] = pd.to_numeric(P.b1_tfidf_top_ksic, errors="coerce"); P["b2_top"] = pd.to_numeric(P.b2_dict_top_ksic, errors="coerce")
P["top_ksic"] = P.b2_top.where(P.b2_dict_assigned == 1, P.b1_top)
P = P[["pjt_no", "year", "b1_top", "top_ksic"]].drop_duplicates("pjt_no")
R = pd.read_parquet(os.path.join(EXT, "ntis_rpatent_raw.parquet"), columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber", "ipc_codes"])
R["Year"] = pd.to_numeric(R.Year, errors="coerce")
R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.RegistType == "출원") & (R.Year <= DEV_MAX_APPL)].drop_duplicates(["ProjectID", "RegistNumber"])
assert R.Year.max() <= DEV_MAX_APPL
R["sec"] = R.ipc_codes.map(first_ipc).fillna("").str.strip().str[:1]
py = P.set_index("pjt_no").year
R["py"] = R.ProjectID.map(py)
R = R.dropna(subset=["py"])
W = R[(R.Year >= R.py) & (R.Year <= R.py + 1)]
outs = {"y_all": set(W.ProjectID)}
for s in SECTIONS:
    outs[f"y_{s}"] = set(W[W.sec == s].ProjectID)
for k, s in outs.items():
    P[k] = np.where(P.year + 1 <= DEV_MAX_APPL, P.pjt_no.isin(s).astype(float), np.nan)
P.to_parquet(os.path.join(CACHE, "n9_outcomes_proj.parquet"), index=False)
Y = pd.DataFrame({"pjt_no": D.pjt_no.values, "t0": D.t0.values, "y_frozen": D.y.values})
pm = P.set_index("pjt_no")
for k in outs:
    Y[k] = D.pjt_no.map(pm[k]).astype(int).values
Y.to_parquet(os.path.join(CACHE, "n9_outcomes_panel.parquet"), index=False)
ev = Y[Y.t0.between(2014, 2020)]
for k in outs:
    tp = int(((ev.y_frozen == 1) & (ev[k] == 1)).sum())
    chk[f"overlap_{k}"] = dict(base=float(ev[k].mean()), pos=int(ev[k].sum()), jaccard_vs_frozen=float(tp / ((ev.y_frozen == 1) | (ev[k] == 1)).sum()),
                               share_of_frozen_pos_covered=float(tp / ev.y_frozen.sum()))
chk["base_frozen"] = float(ev.y_frozen.mean())
log(json.dumps(chk, ensure_ascii=False))

# ---------------- (2) TXT 피처 ----------------
A = pd.read_parquet(os.path.join(R8, "cache", "kr_assign.parquet"), columns=["appl_no", "k_txt"])
assert A.appl_no.is_unique
A = A.dropna(subset=["k_txt"]); kt = A.set_index("appl_no").k_txt
HIST_MAX = 2019
kr = duckdb.sql(f"""select appl_no, any_value(applicant) applicant, any_value(appl_year) appl_year, list(distinct ksic) ksics,
                    min(least(coalesce(pub_date,'99999999'), coalesce(rgst_date,'99999999'))) known
                    from read_parquet('{META}') where cc='KR' and appl_year <= {HIST_MAX} group by appl_no""").df()
kr["known_year"] = pd.to_numeric(kr.known.str[:4], errors="coerce")
kr["nm"] = kr.applicant.map(norm_name).str.replace("산학협력단", "", regex=False)
kr["k_txt"] = pd.to_numeric(kr.appl_no.map(kt), errors="coerce")
orgset = set(D.org2)
kro = kr[kr.nm.isin(orgset)]
krk = kro[["nm", "appl_year", "known_year", "ksics"]].explode("ksics").rename(columns={"ksics": "ksic"}); krk["ksic"] = pd.to_numeric(krk.ksic)
KK = {"rep": krk.groupby(["nm", "ksic", "appl_year", "known_year"]).size().rename("n").reset_index(),
      "txt": kro.dropna(subset=["k_txt"]).rename(columns={"k_txt": "ksic"}).groupby(["nm", "ksic", "appl_year", "known_year"]).size().rename("n").reset_index()}
chk["kr_txt_coverage"] = float(kr.k_txt.notna().mean())
X = D[["pjt_no", "t0", "org2", "top_ksic"]].copy()
for tag in ("rep", "txt"):
    col = np.full(len(X), np.nan)
    for v in sorted(X.t0.unique()):
        m = (X.t0 == v).values
        kk = KK[tag][(KK[tag].appl_year <= v - 2) & (KK[tag].known_year <= v - 1)].groupby(["nm", "ksic"]).n.sum()
        key2 = pd.MultiIndex.from_arrays([X.org2.values[m], X.top_ksic.fillna(-1).astype(int).values[m]])
        col[m] = np.log1p(kk.reindex(key2).fillna(0).values)
    X[f"org_krapp_topk__{tag}"] = col
log("topk done")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
t = duckdb.sql(f"select appl_no, ksic, title from read_parquet('{META}') where cc='KR' and appl_year between 2003 and 2009 and title is not null").df()
PR = pd.read_parquet(os.path.join(EXT, "ntis_project_raw.parquet"), columns=["pjt_no", "year", "title_ko", "kw_ko"]); PR = PR[PR.year >= 2007]
txt = (PR.title_ko.fillna("") + " " + PR.kw_ko.fillna("").str.replace(",", " ")).values
pts = {}
for tag in ("rep", "txt"):
    if tag == "rep":
        titles = t.title.values; lab = t.ksic.astype(str).values
    else:
        tt = t.drop_duplicates("appl_no"); tt = tt.assign(k=tt.appl_no.map(kt)).dropna(subset=["k"])
        titles = tt.title.values; lab = tt.k.astype(str).values
        chk["txt_centroid_titles"] = int(len(tt)); chk["rep_centroid_rows"] = int(len(t))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=200000, min_df=5, sublinear_tf=True, dtype=np.float32)
    Xp = vec.fit_transform(titles)
    ks = sorted(set(lab)); codes = pd.Categorical(lab, categories=ks).codes
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(ks), len(codes)))
    C = normalize(np.asarray((G @ Xp).todense()))
    rows = []
    for i in range(0, len(txt), 100000):
        S = np.asarray(vec.transform(txt[i:i + 100000]) @ C.T); o = np.argsort(-S, axis=1)
        mx = S[np.arange(len(S)), o[:, 0]]; sc = S[np.arange(len(S)), o[:, 1]]
        e = np.exp(S * 20); e = e / e.sum(1, keepdims=True); ent = -(e * np.log(e + 1e-12)).sum(1)
        rows.append(pd.DataFrame({"pt_max": mx, "pt_top": np.array(ks)[o[:, 0]].astype(int), "pt_margin": mx - sc, "pt_ent": ent}))
    Rr = pd.concat(rows, ignore_index=True); Rr.insert(0, "pjt_no", PR.pjt_no.values); pts[tag] = Rr.drop_duplicates("pjt_no").set_index("pjt_no")
    log("centroids", tag, len(ks))
    for c in ["pt_max", "pt_top", "pt_margin", "pt_ent"]:
        X[f"{c}__{tag}"] = X.pjt_no.map(pts[tag][c]).values
rep = {}
for c in ["org_krapp_topk", "pt_max", "pt_top", "pt_margin", "pt_ent"]:
    a = pd.to_numeric(D[c], errors="coerce").astype(float).values; b = X[f"{c}__rep"].astype(float).values
    rep[c] = dict(max_abs_diff=float(np.nanmax(np.abs(a - b))), share_equal=float(np.mean(np.isclose(a, b, atol=1e-4, equal_nan=True))),
                  corr_txt=float(pd.Series(a).corr(pd.Series(X[f"{c}__txt"].astype(float).values))))
chk["replication_rep_vs_frozen"] = rep
chk["pt_top_txt_eq_prov_share"] = float((X.pt_top__txt == D.pt_top).mean())
X[["pjt_no", "t0"] + [c for c in X.columns if c.endswith("__txt") or c.endswith("__rep")]].to_parquet(os.path.join(CACHE, "n9_txt_feats.parquet"), index=False)
json.dump(chk, open(os.path.join(OUT, "n9_build_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(json.dumps(rep, ensure_ascii=False))
log("end")
