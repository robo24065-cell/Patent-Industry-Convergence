# C18-T4 s6: P0(b) 개발 ΔAUC(제공 − 공개 팔 각각), 원점별·평균, 두 학습기. s2_point.json만 읽는다(PREREG §5).
# 게이트는 G1을 통과한 후보×설계에만 적용한다. 다른 후보는 '참고'로만 적는다. 출력 cache/s6_p0b.json, cache/s6_p0b_rows.csv
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
S2 = json.load(open(C + r'\s2_point.json'))
NEED = {'K1': 4 / 5, 'K2': 1.0, 'K3': 1.0, 'K4': 2 / 3}
rows = []; res = {}
for key, b in S2['best_public'].items():
    K = key.split('_')[0]; L = key.split('_')[1] if '_' in key else 'xgb'
    best = b['best']
    for stat in ['auc', 'auc_strat']:
        d = {m: [S2[o][stat]['prov'] - S2[o][stat][m] for o in b['origins']] for m in PUB_ARMS[K]}
        mean_ok = {m: float(np.mean(x)) > 0 for m, x in d.items()}
        frac_best = float(np.mean([x > 0 for x in d[best]]))
        ok = all(mean_ok.values()) and frac_best >= NEED[K] - 1e-9
        r = {'cand': K, 'learner': L, 'stat': 'pooled(D0·D1a·D2)' if stat == 'auc' else 'stratified(D1b)', 'best_public': best,
             'origins': ';'.join(o.split('_')[1] for o in b['origins']),
             **{f'd_{m}_by_origin': ';'.join('%.5f' % x for x in d[m]) for m in PUB_ARMS[K]},
             **{f'd_{m}_mean': float(np.mean(d[m])) for m in PUB_ARMS[K]},
             'frac_origins_pos_vs_best': frac_best, 'need': NEED[K], 'P0b_pass': ok}
        rows.append(r)
        out(K, L, stat, best, json.dumps({m: round(float(np.mean(x)), 5) for m, x in d.items()}), 'frac>0 vs best', frac_best, 'pass', ok)
res['rows'] = rows
json.dump(res, open(C + r'\s6_p0b.json', 'w'), indent=1, ensure_ascii=False, default=float)
pd.DataFrame(rows).to_csv(C + r'\s6_p0b_rows.csv', index=False, encoding='utf-8-sig', float_format='%.6g')
