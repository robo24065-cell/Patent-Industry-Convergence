# 범위안 특허 제목의 '2차 산업' 확률 — 등록≤2009-12-31 범위안으로 2-fold 교차적합(자기 라벨 과신 방지)
import pandas as pd, numpy as np, time
from lib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
t0 = time.time()
kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'title', 'ksic', 'rgst_date'])
kr['title'] = kr.title.fillna('')
fold = (pd.util.hash_array(kr.rgst_no.values.astype(object)) % 2).astype(int)
P = np.zeros((len(kr), 50), np.float16); K = load()['K']
for fo in (0, 1):
    tr = kr[(fold != fo) & (kr.rgst_date <= '20091231')]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.title)
    clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0).fit(X, tr.ksic)
    assert list(clf.classes_) == K
    idx = np.nonzero(fold == fo)[0]
    P[idx] = clf.predict_proba(vec.transform(kr.title.iloc[idx])).astype(np.float16)
    print('fold', fo, round(time.time() - t0), flush=True)
np.save(CACHE/'P_ins_cf.npy', P); kr[['rgst_no']].to_parquet(CACHE/'P_ins_cf_index.parquet', index=False)
lab = kr.ksic.map({k: i for i, k in enumerate(K)}).values
print('cross-fit top1 acc (all in-scope)', (P.argmax(1) == lab).mean().round(4))
