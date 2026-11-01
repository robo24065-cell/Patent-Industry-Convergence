# C08-T4 verify_0 v3: P4 결론(산업 피처 추가 효과 없음) 반박 시도 — 다른 학습기(sklearn HGB), 그리고 사후 변형(출원인 이력 피처를 뺀 M0−).
import sys, os, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
T = time.time()
CW = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C08\C08-T4\cache'
OUT = os.path.dirname(os.path.abspath(__file__))
M0 = ['age_y', 'util', 'days_to_due', 'overdue', 'n_due_win', 'next_ann', 'step', 'pat_late_n', 'pat_pay_n', 'last_lag', 'max_span',
      'own_late_r3', 'own_pay3', 'own_aband3', 'own_alive', 'app_type', 'n_app', 'n_transfer', 'n_lic', 'ipc_sec', 'nopay']
OWN = ['own_late_r3', 'own_pay3', 'own_aband3', 'own_alive']
KF = ['k_late1_prov', 'k_trend_prov', 'k_aband1_prov']
D = pd.concat([pd.read_parquet(os.path.join(CW, f'feat_{Y}.parquet'), columns=['origin', 'y'] + M0 + KF) for Y in range(2015, 2023)], ignore_index=True)
V = {'M0': M0, 'M1': M0 + KF, 'M0minus': [c for c in M0 if c not in OWN], 'M1minus': [c for c in M0 if c not in OWN] + KF}
res = {}
for v in [2018, 2019, 2020, 2021, 2022]:
    tr = D[D.origin < v]; ev = D[D.origin == v]
    rng = np.random.default_rng(5)
    keep = (tr.y.values == 1) | (rng.random(len(tr)) < 0.3); trs = tr[keep]
    w = np.where(trs.y.values == 1, 1.0, 1 / 0.3)
    for nm, cols in V.items():
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200, random_state=5)
        m.fit(trs[cols], trs.y, sample_weight=w)
        res[f'{v}_{nm}'] = float(roc_auc_score(ev.y, m.predict_proba(ev[cols])[:, 1]))
    print(v, {k: round(x, 4) for k, x in res.items() if k.startswith(str(v))}, round(time.time() - T, 1), flush=True)
sk = lambda a, b: np.mean([(res[f'{v}_{b}'] - res[f'{v}_{a}']) / (1 - res[f'{v}_{a}']) for v in range(2018, 2023)])
res['dskill_M1_vs_M0'] = float(sk('M0', 'M1')); res['dskill_M1minus_vs_M0minus'] = float(sk('M0minus', 'M1minus'))
json.dump(res, open(os.path.join(OUT, 'v3_out.json'), 'w', encoding='utf-8'), indent=1)
print({k: v for k, v in res.items() if k.startswith('dskill')})
