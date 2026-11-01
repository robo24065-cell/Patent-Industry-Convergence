# -*- coding: utf-8 -*-
"""C02 과제 내용 ↔ KSIC50 '특허 제목' 대응 (제공 데이터의 KSIC 라벨을 다리로 사용).
- KSIC별 중심 벡터: 제공 KR 출원 중 출원연도 <= 2010 이고 2011-12-31까지 공개·등록으로 알려진 출원의 발명명칭
  (가장 이른 학습 원점 2012 이전에 알려진 자료만 → 모든 원점에서 시점 정합, 정적 사전).
- 과제 텍스트 = 과제명 + 국문 키워드. 글자 2~3그램 TF-IDF(특허 제목으로 적합), 코사인 유사도.
- 출력 cache/proj_ptsim.parquet: pjt_no, year, pt_sim_max(50개 중 최대), pt_top_ksic, pt_margin(1·2위 차), pt_sim_all(전체 중심)
제공 데이터 내용은 로컬에서만 처리(외부 전송 없음)."""
import sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from n6_lib import *
YEAR_HI = int(sys.argv[1]) if len(sys.argv) > 1 else 2021   # 과제연도 상한(future 모드만 2023). 중심 벡터는 연도와 무관하게 고정

k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "공개일자", "등록일자", "발명명칭", "세세분류코드"])
k.columns = ["appl_no", "ay", "pub", "rg", "title", "ksic"]
for c in ["pub", "rg"]:
    k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
k["known"] = k[["pub", "rg"]].min(axis=1)
k["ay"] = pd.to_numeric(k.ay, errors="coerce")
k = k[(k.ay <= 2010) & (k.known <= 20111231)].dropna(subset=["title"])
k["ksic"] = k.ksic.astype(str)
log("patent titles <=2010 known<=2011", len(k), k.ksic.nunique())
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=150000, min_df=3, sublinear_tf=True, dtype=np.float32)
X = vec.fit_transform(k.title.values)
ks = sorted(k.ksic.unique())
ki = k.ksic.map({c: i for i, c in enumerate(ks)}).values
M = sp.csr_matrix((np.ones(len(ki), dtype=np.float32), (ki, np.arange(len(ki)))), shape=(len(ks), len(ki)))
C = normalize(np.asarray((M @ X).todense()))            # 50 × V 중심(합 → L2 정규화 = 평균 방향)
call = normalize(np.asarray(X.sum(axis=0)))
log("centroids", C.shape)
nt = duckdb.sql(f"""select pjt_no, year, coalesce(title_ko,'') || ' ' || coalesce(kw_ko,'') txt from read_parquet('{EXTF}/ntis_project_raw.parquet')
  where year between 2008 and {YEAR_HI} and perform_grp in ('중소기업','중견기업','대기업')""").df()
P = vec.transform(nt.txt.values)
S = np.asarray(P @ C.T)                                 # n × 50 (P는 TF-IDF L2 정규화)
srt = np.sort(S, axis=1)
nt["pt_sim_max"] = srt[:, -1]; nt["pt_margin"] = srt[:, -1] - srt[:, -2]
nt["pt_top_ksic"] = np.array(ks)[S.argmax(1)]
nt["pt_sim_all"] = np.asarray(P @ call.T).ravel()
nt.drop(columns="txt").to_parquet(f"{CACHE}/proj_ptsim.parquet", index=False)
log("saved", len(nt), nt[["pt_sim_max", "pt_margin", "pt_sim_all"]].describe().round(3).to_dict())
