# -*- coding: utf-8 -*-
"""과제 텍스트 ↔ 제공 KR 특허 제목 KSIC 중심점 유사도 (6R 도전자 C06용 열).
중심점: 제공 KR 특허(출원 2003–2009, 2011년까지 모두 공개) 제목의 글자 2–3그램 TF-IDF를 KSIC 50별로 평균.
  → 패널의 어떤 과제(t >= 2010)의 결과 특허(출원 >= 2010)도 중심점에 들어가지 않는다. 레이블을 쓰지 않는다(제목·KSIC 라벨만).
과제 텍스트: 과제명 + 국문 키워드(t년 값). 산출: cache/proj_ptsim.parquet (pjt_no, pt_max, pt_top, pt_margin, pt_ent)."""
import numpy as np, pandas as pd, duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from n9_lib import *

kr = duckdb.sql(f"select ksic, title from read_parquet('{META}') where cc='KR' and appl_year between 2003 and 2009 and title is not null").df()
log("patents", len(kr))
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=200000, min_df=5, sublinear_tf=True, dtype=np.float32)
Xp = vec.fit_transform(kr.title.values)
ks = sorted(kr.ksic.unique())
codes = pd.Categorical(kr.ksic, categories=ks).codes
import scipy.sparse as sp
G = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(ks), len(codes)))
C = normalize(np.asarray((G @ Xp).todense()))
log("centroids", C.shape)
P = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "title_ko", "kw_ko"])
P = P[P.year >= 2007]
txt = (P.title_ko.fillna("") + " " + P.kw_ko.fillna("").str.replace(",", " ")).values
out = []
for i in range(0, len(txt), 100000):
    S = vec.transform(txt[i:i + 100000]) @ C.T
    S = np.asarray(S)
    o = np.argsort(-S, axis=1)
    mx = S[np.arange(len(S)), o[:, 0]]; sc = S[np.arange(len(S)), o[:, 1]]
    e = np.exp(S * 20); e = e / e.sum(1, keepdims=True); ent = -(e * np.log(e + 1e-12)).sum(1)
    out.append(pd.DataFrame({"pt_max": mx, "pt_top": np.array(ks)[o[:, 0]].astype(int), "pt_margin": mx - sc, "pt_ent": ent}))
R = pd.concat(out, ignore_index=True); R.insert(0, "pjt_no", P.pjt_no.values)
R = R.drop_duplicates("pjt_no")
for c in ["pt_max", "pt_margin", "pt_ent"]:
    R[c] = R[c].astype("float32")
R.to_parquet(f"{CACHE}/proj_ptsim.parquet", index=False)
log("saved", len(R), float(R.pt_max.mean()))
