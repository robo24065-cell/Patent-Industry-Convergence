# verify_0 v5(채점 뒤): s6 지표 산술 감사 — 저장된 잠금 레이블(cache\lock_labels.parquet, s6가 1회 계산)과 동결 점수로 sklearn 독립 지표 재계산. 레이블 재계산 없음.
import os, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
WD = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C05\C05-T4'
fz = pd.read_parquet(os.path.join(WD, 'frozen_scores.parquet')).merge(pd.read_parquet(os.path.join(WD, 'cache', 'lock_labels.parquet')), on='key', validate='one_to_one')
res = json.load(open(os.path.join(WD, 'final_result_C05T4.json'), encoding='utf-8'))
out = {}
for m in ['M0', 'M1', 'M1_pub_TXT', 'M1_pub_IPC', 'champion']:
    s = fz[f'score_{m}']
    top = fz.y.values[np.argsort(-s.values, kind='mergesort')[:1000]].mean()
    a = roc_auc_score(fz.y, s); ap = average_precision_score(fz.y, s)
    out[m] = dict(auc=round(a, 5), ap=round(ap, 6), top1000=round(float(top), 4), s6_auc=res['models'][m]['auc'], s6_ap=res['models'][m]['pr_auc'], s6_top=res['models'][m]['top1000'],
                  auc_match=abs(a - res['models'][m]['auc']) < 1e-5, ap_diff=round(ap - res['models'][m]['pr_auc'], 6))
out['n'] = int(len(fz)); out['pos'] = int(fz.y.sum()); out['pos_match'] = out['pos'] == res['pos']
out['score_ties_champion'] = int(len(fz) - fz.score_champion.nunique())
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(WD, 'verify_0', 'v5_metric_audit.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
