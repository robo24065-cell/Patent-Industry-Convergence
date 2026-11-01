# -*- coding: utf-8 -*-
"""C10 도전자: 과제 텍스트 분류기 점수(스태킹 열 txt_clf). 사용: python n9_textclf.py dev [원점...]
- 벡터라이저: 과제명+국문 키워드 글자 2–3그램 TF-IDF, 2007–2013년 과제 텍스트만으로 적합 후 고정(레이블 미사용, 모든 평가 원점 이전).
- 원점 v마다: 학습행(t <= v-2)으로 SGD 로지스틱(고정 alpha=1e-5) 적합 → 시험행(t=v) 점수.
  학습행 점수는 과제연도 묶음 5겹 OOF(같은 해 과제가 같은 겹) — 학습행이 자기 레이블로 적합된 점수를 갖지 않게.
- 산출: cache/txtclf_{mode}_{v}.parquet (row_id, txt_clf) — row_id는 panel_{mode}.parquet 행 번호."""
import sys, pickle
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from n9_lib import *


def get_vec():
    fp = f"{CACHE}/txt_vec.pkl"
    if os.path.exists(fp):
        return pickle.load(open(fp, "rb"))
    P = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "year", "title_ko", "kw_ko"])
    P = P[P.year.between(2007, 2013)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=300000, min_df=5, sublinear_tf=True, dtype=np.float32)
    vec.fit((P.title_ko.fillna("") + " " + P.kw_ko.fillna("").str.replace(",", " ")).values)
    pickle.dump(vec, open(fp, "wb"))
    return vec


def texts_for(D):
    P = pd.read_parquet(f"{EXT}/ntis_project_raw.parquet", columns=["pjt_no", "title_ko", "kw_ko"]).drop_duplicates("pjt_no").set_index("pjt_no")
    t = (P.title_ko.fillna("") + " " + P.kw_ko.fillna("").str.replace(",", " "))
    return t.reindex(D.pjt_no.values).fillna("").values


def run(mode, origins, D=None):
    if D is None:
        D = pd.read_parquet(f"{CACHE}/panel_{mode}.parquet", columns=["pjt_no", "t0", "y"])
    vec = get_vec()
    X = vec.transform(texts_for(D)); log("tfidf", X.shape)
    t0 = D.t0.values; y = D.y.values
    for v in origins:
        tr = np.where(t0 <= v - GAP)[0]; te = np.where(t0 == v)[0]
        assert (y[tr] >= 0).all()
        if mode in ("dev", "devcheck", "placebo"):
            assert t0[tr].max() + 1 <= DEV_MAX_APPL_YEAR
        elif mode == "final":
            assert t0[tr].max() + 1 < F, "학습 레이블 결과창이 F 이후와 겹침"
        clf = lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, tol=None, random_state=0)
        m = clf().fit(X[tr], y[tr]); s_te = m.decision_function(X[te])
        yrs = np.unique(t0[tr]); fold_of = {yy: i % 5 for i, yy in enumerate(yrs)}
        f = np.array([fold_of[yy] for yy in t0[tr]]); s_tr = np.zeros(len(tr))
        for k in range(5):
            a = tr[f != k]; b = f == k
            if b.sum() == 0:
                continue
            s_tr[b] = clf().fit(X[a], y[a]).decision_function(X[tr[b]])
        out = pd.DataFrame({"row_id": np.r_[tr, te], "txt_clf": np.r_[s_tr, s_te].astype("float32")})
        out.to_parquet(f"{CACHE}/txtclf_{mode}_{v}.parquet", index=False)
        if (y[te] >= 0).all():
            log(v, "txt-only AUC test", round(auc(y[te], s_te), 4), "oof", round(auc(y[tr], s_tr), 4))
        else:
            log(v, "scored")


if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode == "dev", "직접 실행은 dev만"
    origins = [int(x) for x in sys.argv[2:]] or DEV_ORIGINS
    run(mode, origins)
