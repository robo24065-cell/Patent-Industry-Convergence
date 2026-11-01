# 분류기 변형 비교(진단): 학습=등록≤2009-12-31 범위안, 평가=2010~2011 출원 범위안(검증 기간, 기업 진입 결과변수와 무관)
import sys, pandas as pd, numpy as np, time, json
from lib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, log_loss
t0 = time.time()
kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title', 'ksic', 'rgst_date', 'ay']); kr['title'] = kr.title.fillna('')
tr = kr[kr.rgst_date <= '20091231']; te = kr[kr.ay.between(2010, 2011)].sample(30000, random_state=0)
V = {'a_base': lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32),
     'b_char+word': lambda: FeatureUnion([('c', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)),
                                         ('w', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32))]),
     'c_char15': lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=5, max_features=600000, sublinear_tf=True, dtype=np.float32)}
A = {'a_base': (2e-6, 12), 'b_char+word': (2e-6, 12), 'c_char15': (2e-6, 12), 'd_alpha1e-6': (1e-6, 20), 'e_alpha5e-6': (5e-6, 12)}
res = json.load(open(T1/'clf_study.json')) if (T1/'clf_study.json').exists() else {}
for nm in sys.argv[1:]:
    vec = V.get(nm, V['a_base'])(); a, it = A[nm]
    X = vec.fit_transform(tr.title); clf = SGDClassifier(loss='log_loss', alpha=a, max_iter=it, tol=None, random_state=0).fit(X, tr.ksic)
    Pt = clf.predict_proba(vec.transform(te.title)); p = clf.classes_[Pt.argmax(1)]
    res[nm] = dict(acc=accuracy_score(te.ksic, p), macroF1=f1_score(te.ksic, p, average='macro'), logloss=log_loss(te.ksic, Pt, labels=clf.classes_))
    print(nm, {k: round(v, 4) for k, v in res[nm].items()}, round(time.time() - t0), flush=True)
    json.dump(res, open(T1/'clf_study.json', 'w'), indent=1)
