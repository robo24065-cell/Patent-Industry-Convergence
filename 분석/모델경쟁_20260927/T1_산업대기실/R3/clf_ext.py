# T1 R3 — 디자인권·범위밖 실용신안 명칭 → KSIC 50 확률 (원점별 cap). 1라운드 clf_base와 같은 분류기 설정.
#   python R3/clf_ext.py --c 2012      (학습: cap=c-12-31까지 등록된 범위안 특허·실용신안 명칭 / 채점: cap까지 등록된 디자인·실용신안)
import sys, pathlib, argparse, time
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; sys.path.insert(0, str(T1))
import numpy as np, pandas as pd
from common import CACHE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import ctypes, ctypes.wintypes as wt
sys.path.insert(0, str(R3)); sys.path.insert(0, str(T1/'R2'))
from r2 import peak_gb

ap = argparse.ArgumentParser(); ap.add_argument('--c', type=int, required=True); a = ap.parse_args()
cap = f'{a.c}1231'; FC = R3/'cache'/f'o{a.c}'; FC.mkdir(parents=True, exist_ok=True); t0 = time.time()
kinds = [k for k in ('design', 'um') if not (FC/f'P_{k}.npy').exists()]
if kinds:
    kr = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['title', 'ksic', 'rgst_date'])
    tr = kr[kr.rgst_date <= cap].dropna(subset=['title']); del kr
    K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique())
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.title)
    clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=12, tol=None, random_state=0).fit(X, tr.ksic); del X
    assert list(clf.classes_) == K
    print('clf fit', len(tr), round(time.time() - t0), f'peak {peak_gb():.2f}GB', flush=True)
    for k in kinds:
        s = pd.read_parquet(R3/'cache'/f'src_{k}.parquet'); s = s[s.rgst_date <= cap].reset_index(drop=True)
        P = np.zeros((len(s), 50), np.float16)
        for i in range(0, len(s), 200000): P[i:i + 200000] = clf.predict_proba(vec.transform(s.title.iloc[i:i + 200000])).astype(np.float16)
        np.save(FC/f'P_{k}.npy', P); s[['applicant_cd', 'rgst_no', 'ay', 'rgst_date']].to_parquet(FC/f'P_{k}_index.parquet', index=False)
        print(k, len(s), round(time.time() - t0), flush=True)
print('done', a.c, round(time.time() - t0), f'peak {peak_gb():.2f}GB')
