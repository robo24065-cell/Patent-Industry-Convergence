# verify_1 r1: dev_eval.csv·final_result의 M1 vs M0 수치를 sklearn으로 독립 재계산(학습 없음, 저장 예측만 읽음).
import os, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); C = os.path.join(W, 'cache')
EVAL = ['201812','201912','202012','202112','202212','202312']
out = {'dev': {}, 'lock': {}}
def top(y, p, k=1000):
    o = np.argsort(-p, kind='mergesort'); return y[o][:k].mean()
sk = lambda a, a0: (a - a0) / (1 - a0)
ds = []; ap = {'M0': [], 'M1': []}; tp = {'M0': [], 'M1': []}
for v in EVAL:
    d0 = pd.read_parquet(os.path.join(C, f'pred_{v}_M0.parquet')); d1 = pd.read_parquet(os.path.join(C, f'pred_{v}_M1.parquet'))
    assert (d0.key.values == d1.key.values).all() and (d0.y.values == d1.y.values).all()
    assert d0.key.is_unique
    y = d0.y.values.astype(int)
    a0 = roc_auc_score(y, d0.p.values); a1 = roc_auc_score(y, d1.p.values)
    r = dict(n=len(y), pos=int(y.sum()), auc0=a0, auc1=a1, skill=sk(a1, a0),
             ap0=average_precision_score(y, d0.p.values), ap1=average_precision_score(y, d1.p.values),
             top0=top(y, d0.p.values), top1=top(y, d1.p.values), ties0=int(len(y)-d0.p.nunique()), ties1=int(len(y)-d1.p.nunique()))
    out['dev'][v] = r; ds.append(r['skill'])
    for m in ['M0','M1']:
        ap[m].append(r['ap'+m[-1]]); tp[m].append(r['top'+m[-1]])
out['dev_mean'] = dict(skill=float(np.mean(ds)), n_pos=int(sum(x > 0 for x in ds)), ap0=float(np.mean(ap['M0'])), ap1=float(np.mean(ap['M1'])),
                       top0=float(np.mean(tp['M0'])), top1=float(np.mean(tp['M1'])))
f = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet')); l = pd.read_parquet(os.path.join(C, 'lock_labels.parquet'))
assert (f.key.values == l.key.values).all()
y = l.y.values.astype(int)
a0 = roc_auc_score(y, f.score_M0); a1 = roc_auc_score(y, f.score_M1)
out['lock'] = dict(n=len(y), pos=int(y.sum()), auc0=a0, auc1=a1, skill=sk(a1, a0), champ_eq_M1=bool(np.allclose(f.score_champion, f.score_M1)),
                   ap0=average_precision_score(y, f.score_M0), ap1=average_precision_score(y, f.score_M1),
                   top0=top(y, f.score_M0.values), top1=top(y, f.score_M1.values))
# 잠금 양성 문서 키 = 레이블 양성 키
pdk = pd.read_parquet(os.path.join(C, 'lock_pos_docs.parquet'))
out['lock']['posdocs_keys_eq_labels'] = bool(set(pdk.key) == set(l.key[l.y == 1]))
json.dump(out, open(os.path.join(W, 'verify_1', 'r1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
