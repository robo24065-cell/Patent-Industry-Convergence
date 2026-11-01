# v0 고정 기준선: 로지스틱(경과연수 + 출원인 유형 + KSIC + 청구항 수 + 등록연도)
# 학습: L=2006..2011 (결과창이 2014-01-01 이전 종료) / 검증: L=2014, 2015
import sys, numpy as np, pandas as pd, json
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from common import *
from evalkit import *

cols = ['rgst_no', 'L', 'age', 'orig_type', 'ksic', 'claims', 'rgst_year', 'y_MAIN', 'y_CORP', 'y_INH', 'y_ANY', 'cur']
tr = load_panel(range(2006, TRAIN_MAX_L + 1), cols); va = load_panel(VAL_L, cols)
for d in (tr, va):
    d['lclaims'] = np.log1p(d.claims.fillna(d.claims.median()))
num = ['age', 'lclaims', 'rgst_year']; cat = ['orig_type', 'ksic']
ct = ColumnTransformer([('n', StandardScaler(), num), ('c', OneHotEncoder(handle_unknown='ignore'), cat)])
m = make_pipeline(ct, LogisticRegression(max_iter=2000, C=1.0))
res = {}
for tgt in ['y_MAIN', 'y_ANY', 'y_CORP']:
    m.fit(tr[num + cat], tr[tgt]); p = m.predict_proba(va[num + cat])[:, 1]
    mt = metrics(va[tgt], p); res[tgt] = mt
    print(tgt, 'train n', len(tr), 'pos', int(tr[tgt].sum()), '| val n', len(va), 'pos', int(va[tgt].sum()), {k: round(v, 4) for k, v in mt.items()})
    if tgt == 'y_MAIN':
        pm = p
        for L in VAL_L:
            k = va.L == L; print('  L', L, {kk: round(v, 4) for kk, v in metrics(va.y_MAIN[k], p[k]).items()})
        ci = cluster_boot(va.y_MAIN.values, p, va.cur.values)
        print('  CI AUC', ci[0].round(4), 'PR', ci[1].round(4))
        pd.DataFrame({'rgst_no': va.rgst_no, 'L': va.L, 'p': p}).to_parquet('cache/pred_v0.parquet', index=False)
json.dump({k: {kk: float(v) for kk, v in d.items()} for k, d in res.items()}, open('cache/v0_metrics.json', 'w'))
log('v0', '고정 기준선 로지스틱(경과연수+출원인유형+KSIC+청구항수+등록연도), 학습 L2006-2011', '기준선', res['y_MAIN']['auc'], res['y_MAIN']['auc'], 'Y',
    f"PR-AUC {res['y_MAIN']['prauc']:.4f} (base rate {res['y_MAIN']['base']:.4f}), p@1% {res['y_MAIN']['p1']:.4f}; y_ANY AUC {res['y_ANY']['auc']:.4f}")
