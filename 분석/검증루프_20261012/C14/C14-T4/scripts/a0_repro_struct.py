# C14-T4 a0: (1) 날짜 가드 (2) C12 원판 '이원'(곱 Poisson) SE 재현 — C12 evalkit12.Boot 그대로(시드 20261013, B=200)
# (3) 군집 구조 진단 q_f, q_s. 출력 cache/a0_out.json
import sys, os, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
t = time.time()
res = {}
# (1) 날짜 가드: C12 문서 캐시 최대 공개일
res['guard'] = {f'docs_{cc}': guard_parquet_pub(C12C + rf'\docs_{cc}.parquet') for cc in ['CN', 'EP', 'KR']}
out('guard', res['guard'])
fr = load_c12_frames(models=('prov', 'free', 'hsc'))
res['rows'] = {str(f.v.iloc[0]): {'n': len(f), 'pos': int(f.y.sum()), 'firms': int(f.groupby(['cc', 'aid']).ngroups), 'sc': int(f.s.nunique())} for f in fr}
out('rows', res['rows'])

# (2) 원판 재현
sys.path.insert(0, C12 + r'\scripts')
import evalkit12 as ek
bt = ek.Boot(fr, B=200)   # seed = lib12.SEED = 20261013
preps = [{m: AUCPrep(f[m].values, f.y.values) for m in ('prov', 'free', 'hsc')} for f in fr]
pt = [{m: preps[k][m].auc(f.w.values.astype(np.float64)) for m in preps[k]} for k, f in enumerate(fr)]
D_pt = np.mean([min(dskill(a['prov'], a['free']), dskill(a['prov'], a['hsc'])) for a in pt])
reps = {'two': [], 'firm': []}
for b in range(200):
    for kd in reps:
        r = []
        for k, f in enumerate(fr):
            w = f.w.values * bt.weights(k, b, kd)
            a = {m: preps[k][m].auc(w) for m in preps[k]}
            r.append(min(dskill(a['prov'], a['free']), dskill(a['prov'], a['hsc'])))
        reps[kd].append(float(np.mean(r)))
    if b % 50 == 0:
        out('repro b', b, round(time.time() - t, 1))
res['repro'] = {'D_point': float(D_pt),
                'two': {'ci': [float(np.percentile(reps['two'], 2.5)), float(np.percentile(reps['two'], 97.5))], 'sd': float(np.std(reps['two'], ddof=1))},
                'firm': {'ci': [float(np.percentile(reps['firm'], 2.5)), float(np.percentile(reps['firm'], 97.5))], 'sd': float(np.std(reps['firm'], ddof=1))},
                'c12_two_ci': [0.010302345504581192, 0.020614156913602966], 'c12_firm_ci': [0.012829001720007439, 0.01769976076191839]}
r2 = res['repro']
r2['c12_two_sd_from_ci'] = (r2['c12_two_ci'][1] - r2['c12_two_ci'][0]) / 3.92
r2['match'] = bool(max(abs(r2['two']['ci'][0] - r2['c12_two_ci'][0]), abs(r2['two']['ci'][1] - r2['c12_two_ci'][1])) <= 0.0003
                   and abs(r2['two']['sd'] / r2['c12_two_sd_from_ci'] - 1) <= 0.10)
out('repro', json.dumps(r2))

# (3) 구조 진단
st = {}
for f in fr:
    v = int(f.v.iloc[0])
    fk = f.cc.astype(str) + ':' + f.aid.astype(str)
    nsc_per_firm = f.groupby(fk).s.nunique()
    nfirm_per_sc = pd.DataFrame({'fk': fk, 's': f.s}).groupby('s').fk.nunique()
    q_f = float((fk.map(nsc_per_firm) >= 2).mean())
    q_s = float((f.s.map(nfirm_per_sc) >= 2).mean())
    pos = f[f.y == 1]
    pfk = pos.cc.astype(str) + ':' + pos.aid.astype(str)
    st[v] = {'q_f': q_f, 'q_s': q_s, 'sc_per_firm_median': float(nsc_per_firm.median()), 'firms_per_sc_median': float(nfirm_per_sc.median()),
             'pos_sc_per_firm_mean': float(pos.groupby(pfk).s.nunique().mean()), 'pos_firms_per_sc_median': float(pos.groupby('s').size().median()),
             'n_pos_sc': int(pos.s.nunique()), 'n_pos_firm': int(pfk.nunique()),
             'structure': 'nested_firm_in_sc' if q_f < 0.05 else ('nested_sc_in_firm' if q_s < 0.05 else ('crossed' if (q_f >= 0.5 and q_s >= 0.5) else 'partial_crossed'))}
    out('struct', v, st[v])
res['struct'] = st
res['ptAUC'] = [{m: float(x) for m, x in a.items()} for a in pt]
res['secs'] = round(time.time() - t, 1)
json.dump(res, open(C + r'\a0_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out('done', res['secs'])
