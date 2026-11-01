# 3단계 진단(모형 변경 없음): 홀드아웃 성능 하락의 원인 분해
import sys, os, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = pd.read_parquet(os.path.join(H, 'holdout_pred.parquet'))
r = {}
pos = m[m.y_MAIN == 1]
bs = pos.groupby(['L', 'cur', 'y_date']).rgst_no.transform('size')
m['bsize'] = 0; m.loc[pos.index, 'bsize'] = bs.values
neg = m.y_MAIN == 0
for lab, lo, hi in [('단건', 1, 1), ('2-5', 2, 5), ('6-50', 6, 50), ('51+', 51, 10**9)]:
    k = neg | ((m.bsize >= lo) & (m.bsize <= hi))
    r[f'bundle_{lab}'] = dict(pos=int(((m.bsize >= lo) & (m.bsize <= hi)).sum()), auc=A(m.y_MAIN[k], m.p[k]), auc_base=A(m.y_MAIN[k], m.p_base[k]))
top = pos.groupby('cur').size().sort_values(ascending=False)
for n in [1, 3, 10]:
    ex = set(top.index[:n]); k = ~m.cur.isin(ex)
    r[f'excl_top{n}_sellers'] = dict(excluded=list(ex), n=int(k.sum()), pos=int(m.y_MAIN[k].sum()), auc=A(m.y_MAIN[k], m.p[k]), auc_base=A(m.y_MAIN[k], m.p_base[k]))
# 상위 매각 권리자의 예측 순위: 권리자 owner_hat 백분위
oh = m.groupby(['L', 'cur']).agg(owner_hat=('owner_hat', 'first'), n=('p', 'size'), rate=('y_MAIN', 'mean')).reset_index()
oh['pct_rank'] = oh.groupby('L').owner_hat.rank(pct=True)
r['top_sellers_owner_rank'] = oh[oh.cur.isin(top.index[:10])].sort_values(['cur', 'L']).round(4).to_dict('records')
# 권리자 단위 가중 AUC 대체지표: 권리자 내부(within-owner) AUC — 보유 20건 이상·양성/음성 모두 있는 권리자
w = []
for (L, c), d in m.groupby(['L', 'cur']):
    if len(d) >= 20 and 0 < d.y_MAIN.sum() < len(d): w.append((len(d), A(d.y_MAIN, d.p)))
w = np.array(w); r['within_owner_auc_weighted'] = float(np.average(w[:, 1], weights=w[:, 0])); r['within_owner_n'] = int(len(w))
# 권리자 수준(보유건수 가중 X): 권리자별 관측률 vs owner_hat 스피어만
from scipy.stats import spearmanr
o2 = oh[oh.n >= 10]; r['owner_level_spearman_n>=10'] = float(spearmanr(o2.owner_hat, o2.rate)[0]); r['owner_level_n'] = int(len(o2))
json.dump(r, open(os.path.join(H, 'holdout_check', 'diag.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
