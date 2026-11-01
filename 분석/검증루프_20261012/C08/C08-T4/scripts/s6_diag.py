# C08-T4 s6(판정 밖 진단): 평가 원점 2018–2022 단변량 AUC(결측은 중앙값 대치), 피처 간 상관.
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import C
from sklearn.metrics import roc_auc_score
D = pd.concat([pd.read_parquet(os.path.join(C, f'feat_{Y}.parquet')) for Y in range(2018, 2023)], ignore_index=True)
F = ['n_due_win', 'days_to_due', 'overdue', 'pat_late_n', 'last_lag', 'own_late_r3', 'own_aband3', 'app_type', 'age_y', 'next_ann', 'k_late1_prov', 'k_trend_prov', 'k_aband1_prov']
out = {}
for f in F:
    x = D[f].fillna(D[f].median())
    out[f] = round(float(np.mean([roc_auc_score(g.y, x[g.index]) for _, g in D.groupby('origin')])), 4)
out['corr_k_late1_vs_own_late_r3'] = round(float(D[['k_late1_prov', 'own_late_r3']].corr(method='spearman').iloc[0, 1]), 4)
out['share_n_due_win_0'] = round(float((D.n_due_win == 0).mean()), 4)
out['rate_if_n_due_win_0'] = round(float(D[D.n_due_win == 0].y.mean()), 5)
out['rate_if_n_due_win_ge1'] = round(float(D[D.n_due_win >= 1].y.mean()), 5)
# 출원인 유형 안에서의 산업 피처 단변량 AUC(유형 교란 확인)
out['k_late1_auc_within_type'] = {str(t): round(float(roc_auc_score(g.y, g.k_late1_prov)), 4) for t, g in D.groupby('app_type') if g.y.nunique() == 2 and len(g) > 5000}
json.dump(out, open(os.path.join(C, 's6_diag.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
