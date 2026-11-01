# -*- coding: utf-8 -*-
"""N9 R7 대체 피처 생성(개발 모드만, 봉인 준수: 성과 출원연도 <= 2019 이력만, 결과 레이블은 동결 패널 값 그대로).
제공 데이터 유래 피처 3묶음을 (a) 동결 코드와 같은 방식으로 '제공판'을 다시 계산해 동결 패널과 대조(복제 검증)하고,
(b) 제공 KSIC 라벨 대신 공식 IPC-KSIC 연계표(2013) 배정으로 '대체판'을 만든다.
  P1 제공 50산업 성과 이력: org_p50_L2, org_p50_rate_L2, org_p50_share_L2, pre_p50_L2, chain_p50_L2
     제공판 = NTIS 성과 중 제공 KR(50 KSIC) 특허에 연결된 것 / 대체판 = NTIS KR 특허 출원 성과 중 첫 IPC의 연계표 배정(free)이 25개 소분류 안인 것
  P2 제공 KR 제목 KSIC 중심점: pt_max, pt_top, pt_margin, pt_ent
     제공판 = 제공 KSIC 라벨별 중심점(n9_text.py) / 대체판 = 같은 제목(제공 KR 2003–09)을 메인 IPC 연계표 배정(free, 25개 소분류 안만)별 중심점으로
  P3 제공 KR 출원 이력(기관명 매칭): org_krapp, org_krapp3, org_krapp_topk
     대체판 = 메인 IPC 연계표 배정(free)이 25개 소분류 안인 출원만, topk는 과제 대응 KSIC의 소분류 3자리와 연계표 소분류 일치
산출: alt_features_dev.parquet (panel_dev 행 순서, 대체판 열 + 제공판 재계산 열), alt_replication.json"""
import os, sys, json, re, time
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); N9 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, N9)
from r7_concord import assign, first_ipc, G50
from n9_lib import EXT, META, CACHE, norm_name, log

HIST_MAX = 2019          # dev 패널 피처용 이력 출원연도 상한(n9_build: max(origins)-1)
D = pd.read_parquet(f"{CACHE}/panel_dev.parquet", columns=["pjt_no", "t0", "year", "org", "org2", "pre_pjt", "has_pre", "top_ksic",
                                                           "org_p50_L2", "org_p50_rate_L2", "org_p50_share_L2", "pre_p50_L2", "chain_p50_L2",
                                                           "org_krapp", "org_krapp3", "org_krapp_topk", "pt_max", "pt_top", "pt_margin", "pt_ent"])
assert D.t0.max() == 2020
cols = ["pjt_no", "year", "research_agency", "pre_pjt", "secret_flag"]
RAW = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=cols)
P = RAW[(RAW.year >= 2007) & (RAW.year <= 2020) & (RAW.secret_flag.astype(float).fillna(0) == 0)].copy()
P["year"] = P.year.astype(int); P["org"] = P.research_agency.map(norm_name)
P["pre1"] = P.pre_pjt.astype(str).str.split(",").str[0].where(P.pre_pjt.notna())

# ---- 성과 ----
L = pd.read_parquet(f"{EXT}/ntis_patent_link.parquet", columns=["appl_no", "ksic", "ProjectID", "appl_year"])
L["appl_year"] = pd.to_numeric(L.appl_year, errors="coerce")
L = L[L.appl_year <= 2021].drop_duplicates(["ProjectID", "appl_no"])
R = pd.read_parquet(f"{EXT}/ntis_rpatent_raw.parquet", columns=["ProjectID", "Year", "is_kr", "IprType", "RegistType", "RegistNumber", "ipc_codes"])
R["Year"] = pd.to_numeric(R.Year, errors="coerce")
R = R[R.is_kr.fillna(False).astype(bool) & (R.IprType == "특허") & (R.RegistType == "출원") & (R.Year <= HIST_MAX)].drop_duplicates(["ProjectID", "RegistNumber"])
g, _ = assign(R.ipc_codes.map(first_ipc).values, "free"); R["c50"] = np.isin(g, G50)
log("R", len(R), "conc50 share", round(R.c50.mean(), 4))
PP = P[["pjt_no", "org", "year"]]
Lc = L.groupby(["ProjectID", "appl_year"]).size().rename("n").reset_index().merge(PP, left_on="ProjectID", right_on="pjt_no", how="inner")
Rc = R.groupby(["ProjectID", "Year"]).size().rename("n").reset_index().rename(columns={"Year": "appl_year"}).merge(PP, left_on="ProjectID", right_on="pjt_no", how="inner")
Ac = R[R.c50].groupby(["ProjectID", "Year"]).size().rename("n").reset_index().rename(columns={"Year": "appl_year"}).merge(PP, left_on="ProjectID", right_on="pjt_no", how="inner")
LcH = Lc[Lc.appl_year <= HIST_MAX]; AcH = Ac[Ac.appl_year <= HIST_MAX]

# ---- 선행과제 사슬(n9_build 그대로) ----
pre_map = RAW[RAW.year <= 2020].assign(pre1=lambda d: d.pre_pjt.astype(str).str.split(",").str[0].where(d.pre_pjt.notna())).set_index("pjt_no").pre1
pre_map = pre_map[~pre_map.index.duplicated()]
anc = []; cur = P[["pjt_no", "year"]].copy(); cur["anc"] = P.pre1.values; depth = 0
while depth < 15:
    cur = cur[cur.anc.notna() & (cur.anc != cur.pjt_no)]
    if len(cur) == 0:
        break
    depth += 1; anc.append(cur.assign(d=depth)); cur = cur.assign(anc=cur.anc.map(pre_map))
A = pd.concat(anc, ignore_index=True)

# ---- 제공 KR 출원 ----
kr = duckdb.sql(f"""select appl_no, any_value(applicant) applicant, any_value(appl_year) appl_year, any_value(main_ipc) ipc, list(distinct ksic) ksics,
                    min(least(coalesce(pub_date,'99999999'), coalesce(rgst_date,'99999999'))) known
                    from read_parquet('{META}') where cc='KR' and appl_year <= {HIST_MAX} group by appl_no""").df()
kr["known_year"] = pd.to_numeric(kr.known.str[:4], errors="coerce")
kr["nm"] = kr.applicant.map(norm_name).str.replace("산학협력단", "", regex=False)
gk, _ = assign(kr.ipc.values, "free"); kr["cg"] = gk; kr["c50"] = np.isin(gk, G50)
orgset = set(D.org2)
kro = kr[kr.nm.isin(orgset)]
kroc = kro.groupby(["nm", "appl_year", "known_year"]).size().rename("n").reset_index()
krk = kr[["nm", "appl_year", "known_year", "ksics"]].explode("ksics").rename(columns={"ksics": "ksic"}); krk["ksic"] = pd.to_numeric(krk.ksic)
krok = krk[krk.nm.isin(orgset)].groupby(["nm", "ksic", "appl_year", "known_year"]).size().rename("n").reset_index()
kroc_a = kro[kro.c50].groupby(["nm", "appl_year", "known_year"]).size().rename("n").reset_index()
krok_a = kro[kro.c50].groupby(["nm", "cg", "appl_year", "known_year"]).size().rename("n").reset_index()
log("kr", len(kr), "c50", round(kr.c50.mean(), 4))

out = []
for v in sorted(D.t0.unique()):
    X = D[D.t0 == v].copy()
    hist = P[P.year.between(v - 3, v - 1)]; nproj = hist.groupby("org").size()
    amax = v - 2
    RcW = Rc[Rc.year.between(v - 3, v - 1) & (Rc.appl_year <= amax)]; ra = RcW.groupby("org").n.sum()
    for tag, H in (("rep", LcH), ("alt", AcH)):
        W = H[H.year.between(v - 3, v - 1) & (H.appl_year <= amax)]
        la = W.groupby("org").n.sum(); lp = W.groupby("org").pjt_no.nunique()
        X[f"org_p50_L2__{tag}"] = np.log1p(X.org.map(la).fillna(0))
        X[f"org_p50_rate_L2__{tag}"] = X.org.map(lp).fillna(0) / (X.org.map(nproj).fillna(0) + 1)
        X[f"org_p50_share_L2__{tag}"] = X.org.map(la).fillna(0) / (X.org.map(ra).fillna(0) + 1)
        lpre = H[H.appl_year <= amax].groupby("ProjectID").n.sum()
        X[f"pre_p50_L2__{tag}"] = np.log1p(X.pre_pjt.map(lpre).astype(float).fillna(0)) * X.has_pre
        Av = A[A.year == v]
        ch = Av.merge(H[H.appl_year <= v - 2][["ProjectID", "n"]], left_on="anc", right_on="ProjectID", how="inner").groupby("pjt_no").n.sum()
        X[f"chain_p50_L2__{tag}"] = np.log1p(X.pjt_no.map(ch).astype(float).fillna(0))
    for tag, KC, KK, kcol, kfun in (("rep", kroc, krok, "ksic", lambda s: s.fillna(-1).astype(int).values),
                                    ("alt", kroc_a, krok_a, "cg", lambda s: (s.fillna(-100) // 100).astype(int).astype(str).values)):
        kc = KC[(KC.appl_year <= v - 2) & (KC.known_year <= v - 1)]
        X[f"org_krapp__{tag}"] = np.log1p(X.org2.map(kc.groupby("nm").n.sum()).fillna(0))
        X[f"org_krapp3__{tag}"] = np.log1p(X.org2.map(kc[kc.appl_year >= v - 4].groupby("nm").n.sum()).fillna(0))
        kk = KK[(KK.appl_year <= v - 2) & (KK.known_year <= v - 1)].groupby(["nm", kcol]).n.sum()
        key2 = pd.MultiIndex.from_arrays([X.org2.values, kfun(X.top_ksic)])
        X[f"org_krapp_topk__{tag}"] = np.log1p(kk.reindex(key2).fillna(0).values)
    out.append(X)
    log(v, len(X))
X = pd.concat(out).loc[D.index]

# ---- P2 중심점: 연계표 그룹별 ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
t = duckdb.sql(f"select ksic, main_ipc, title from read_parquet('{META}') where cc='KR' and appl_year between 2003 and 2009 and title is not null").df()
cg, _ = assign(t.main_ipc.values, "free"); t["cg"] = cg
PR = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "title_ko", "kw_ko"]); PR = PR[PR.year >= 2007]
txt = (PR.title_ko.fillna("") + " " + PR.kw_ko.fillna("").str.replace(",", " ")).values
pts = {}
for tag, lab, keep in (("rep", t.ksic.astype(str).values, np.ones(len(t), bool)), ("alt", t.cg.values, np.isin(t.cg.values, G50))):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=200000, min_df=5, sublinear_tf=True, dtype=np.float32)
    Xp = vec.fit_transform(t.title.values[keep]) if tag == "alt" else vec.fit_transform(t.title.values)
    lb = lab[keep]
    ks = sorted(set(lb)); codes = pd.Categorical(lb, categories=ks).codes
    G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(ks), len(codes)))
    C = normalize(np.asarray((G @ Xp).todense()))
    rows = []
    for i in range(0, len(txt), 100000):
        S = np.asarray(vec.transform(txt[i:i + 100000]) @ C.T); o = np.argsort(-S, axis=1)
        mx = S[np.arange(len(S)), o[:, 0]]; sc = S[np.arange(len(S)), o[:, 1]]
        e = np.exp(S * 20); e = e / e.sum(1, keepdims=True); ent = -(e * np.log(e + 1e-12)).sum(1)
        rows.append(pd.DataFrame({f"pt_max__{tag}": mx, f"pt_top__{tag}": np.array(ks)[o[:, 0]].astype(int), f"pt_margin__{tag}": mx - sc, f"pt_ent__{tag}": ent}))
    Rr = pd.concat(rows, ignore_index=True); Rr.insert(0, "pjt_no", PR.pjt_no.values); pts[tag] = Rr.drop_duplicates("pjt_no").set_index("pjt_no")
    log("centroids", tag, len(ks), int(keep.sum()))
for tag in ("rep", "alt"):
    for c in ["pt_max", "pt_top", "pt_margin", "pt_ent"]:
        X[f"{c}__{tag}"] = X.pjt_no.map(pts[tag][f"{c}__{tag}"]).values

# ---- 복제 검증 ----
rep = {}
for c in ["org_p50_L2", "org_p50_rate_L2", "org_p50_share_L2", "pre_p50_L2", "chain_p50_L2", "org_krapp", "org_krapp3", "org_krapp_topk", "pt_max", "pt_top", "pt_margin", "pt_ent"]:
    a = pd.to_numeric(X[c], errors="coerce").astype(float).values; b = X[f"{c}__rep"].astype(float).values
    rep[c] = dict(max_abs_diff=float(np.nanmax(np.abs(a - b))), share_equal=float(np.mean(np.isclose(a, b, atol=1e-4, equal_nan=True))),
                  corr_alt=float(pd.Series(a).corr(pd.Series(X[f"{c}__alt"].astype(float).values))))
print(json.dumps(rep, ensure_ascii=False, indent=0))
json.dump(dict(replication=rep, n=int(len(X)), conc50_share_ntis_kr_app=float(R.c50.mean()), conc50_share_provided_kr=float(kr.c50.mean())),
          open(os.path.join(HERE, "alt_replication.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
keep = ["pjt_no", "t0"] + [c for c in X.columns if c.endswith("__alt") or c.endswith("__rep")]
X[keep].reset_index(drop=True).to_parquet(os.path.join(HERE, "alt_features_dev.parquet"), index=False)
log("saved")
