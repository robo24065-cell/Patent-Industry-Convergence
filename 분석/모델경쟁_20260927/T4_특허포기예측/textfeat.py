# -*- coding: utf-8 -*-
"""발명 명칭 텍스트 점수: TF-IDF(단어 1-2gram + 글자 2-3gram) -> SGD 로지스틱.
학습 행은 출원인 GroupKFold 5겹 OOF, 평가 행은 학습 전체로 적합한 모형 점수. 학습 레이블만 사용."""
import os, numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold
import t4lib as T


def _vec(titles_fit):
    w = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=5, max_features=80000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), min_df=10, max_features=80000, sublinear_tf=True)
    w.fit(titles_fit); c.fit(titles_fit)
    return lambda t: sparse.hstack([w.transform(t), c.transform(t)]).tocsr()


def text_scores(tr, ev, seed=0):
    pt = T.load()['pt']
    title = pd.read_parquet(os.path.join(T.CACHE, 'patent.parquet'), columns=['rgst_no', 'title']).set_index('rgst_no').title.reindex(pt.rgst_no.values).fillna('').values
    # 고유 특허 단위로 벡터화 후 행에 인덱싱
    up = np.unique(np.r_[tr.pid.values, ev.pid.values])
    pos = pd.Series(np.arange(len(up)), index=up)
    tr_u = np.unique(tr.pid.values)
    V = _vec(title[tr_u])
    X = V(title[up])
    Xtr = X[pos[tr.pid.values].values]; Xev = X[pos[ev.pid.values].values]
    y = tr.y.values.astype(int)
    mk = lambda: SGDClassifier(loss='log_loss', alpha=2e-5, max_iter=15, tol=None, random_state=seed)
    oof = np.zeros(len(tr))
    grp = tr.acd.values if 'acd' in tr else pt.acd.values[tr.pid.values]
    for a, b in GroupKFold(5).split(Xtr, y, grp):
        m = mk().fit(Xtr[a], y[a]); oof[b] = m.decision_function(Xtr[b])
    m = mk().fit(Xtr, y)
    return oof, m.decision_function(Xev)
