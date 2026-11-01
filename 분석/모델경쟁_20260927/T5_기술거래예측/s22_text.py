# 제목 텍스트 점수: 문자 2~3gram TF-IDF + 로지스틱. 학습 L 구간은 권리자 군집 5-fold OOF, 검증 L 은 학습 전체로 적합.
# 사용: python s22_text.py <train_first> <train_last> <apply_years...>
import sys, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common import *
t0 = time.time()
a, b = int(sys.argv[1]), int(sys.argv[2]); apply_years = [int(x) for x in sys.argv[3:]]
cols = ['rgst_no', 'L', 'title', 'cur', 'y_MAIN']
tr = load_panel(range(a, b + 1), cols); ap = load_panel(apply_years, cols)
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), min_df=20, max_features=150000, sublinear_tf=True, dtype=np.float32)
Xtr = vec.fit_transform(tr.title.fillna('')); Xap = vec.transform(ap.title.fillna(''))
print('vocab', Xtr.shape, round(time.time() - t0, 1), flush=True)
oof = np.zeros(len(tr))
for f, (i, j) in enumerate(GroupKFold(5).split(Xtr, tr.y_MAIN, tr.cur.fillna(''))):
    m = LogisticRegression(C=0.5, max_iter=300, solver='liblinear'); m.fit(Xtr[i], tr.y_MAIN.values[i])
    oof[j] = m.decision_function(Xtr[j])
print('OOF AUC', round(roc_auc_score(tr.y_MAIN, oof), 4), round(time.time() - t0, 1), flush=True)
m = LogisticRegression(C=0.5, max_iter=300, solver='liblinear'); m.fit(Xtr, tr.y_MAIN.values)
pa = m.decision_function(Xap)
if ap.L.max() <= 2015:
    print('apply AUC', round(roc_auc_score(ap.y_MAIN, pa), 4))
tag = f'{a}_{b}'
pd.concat([pd.DataFrame({'rgst_no': tr.rgst_no, 'L': tr.L, 'text_score': oof}),
           pd.DataFrame({'rgst_no': ap.rgst_no, 'L': ap.L, 'text_score': pa})]).to_parquet(f'cache/text_{tag}.parquet', index=False)
voc = np.array(vec.get_feature_names_out()); co = m.coef_[0]
print('top+', list(voc[np.argsort(-co)[:30]])); print('top-', list(voc[np.argsort(co)[:20]]))
