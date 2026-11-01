# P2·P4·반박(i)(ii)(iii)·민감도 평가(P0 실패 뒤라 모두 '판정 밖' 보고). 사용: python -I s4_eval.py part
#  part 'main' = 점추정 + P2·P4·(i)·(ii) CI(기업·이원, B=200) / 'sub' = (iii)·민감도·나라별(기업 군집 CI)
# 출력 cache/eval_{part}.json
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalkit12 import *

part = sys.argv[1]
t = time.time()
MODELS = ['SB', 'PI', 'pub', 'prov', 'free', 'hsc', 'all', 'pubctx', 'scagg']
fr = []
for v in EVAL_Y:
    P = None
    for g in ['sb', 'base', 'pubarms', 'ref']:
        q = pd.read_parquet(C + rf'\pred_{v}_{g}.parquet')
        if P is None:
            P = q
        else:
            assert (P.aid.values == q.aid.values).all() and (P.s.values == q.s.values).all() and (P.cc.values == q.cc.values).all()
            for m in q.columns:
                if m in MODELS:
                    P[m] = q[m].values
    u = pd.concat([pd.read_parquet(C + rf'\u_{cc}_{v}.parquet', columns=['aid', 's', 'f_log3', 'f_missc']) for cc in CCS], ignore_index=True)
    assert (u.aid.values == P.aid.values).all() and (u.s.values == P.s.values).all()
    P['f_log3'] = u.f_log3.values; P['f_missc'] = u.f_missc.values
    P['v'] = v
    fr.append(P)
out('loaded', round(time.time() - t, 1))


def stats(frames, B, kinds, models=MODELS):
    preps = [{m: AUCPrep(f[m].values, f.y.values) for m in models} for f in frames]
    bt = Boot(frames, B=B)
    pt = [{m: preps[k][m].auc(f.w.values.astype(np.float64)) for m in models} for k, f in enumerate(frames)]
    reps = {kd: [] for kd in kinds}
    for b in range(B):
        for kd in kinds:
            r = []
            for k, f in enumerate(frames):
                w = f.w.values * bt.weights(k, b, kd)
                r.append({m: preps[k][m].auc(w) for m in models})
            reps[kd].append(r)
    return pt, reps


def contrasts(a):
    """a: list(원점) of {model: auc} → 대비 dict(원점별 리스트)"""
    C_ = {}
    C_['pub_vs_PI'] = [dskill(x['pub'], x['PI']) for x in a]
    C_['pub_vs_SB'] = [dskill(x['pub'], x['SB']) for x in a]
    C_['prov_vs_free'] = [dskill(x['prov'], x['free']) for x in a]
    C_['prov_vs_hsc'] = [dskill(x['prov'], x['hsc']) for x in a]
    C_['d_v'] = [min(p, q) for p, q in zip(C_['prov_vs_free'], C_['prov_vs_hsc'])]
    C_['prov_vs_pub'] = [dskill(x['prov'], x['pub']) for x in a]
    C_['free_vs_pub'] = [dskill(x['free'], x['pub']) for x in a]
    C_['hsc_vs_pub'] = [dskill(x['hsc'], x['pub']) for x in a]
    if 'all' in a[0]:
        C_['E_pub_to_prov'] = [dskill(x['all'], x['prov']) for x in a]
        C_['E_prov_to_pub'] = [dskill(x['all'], x['pubctx']) for x in a]
        C_['prov_vs_scagg'] = [dskill(x['prov'], x['scagg']) for x in a]
        C_['scagg_vs_pub'] = [dskill(x['scagg'], x['pub']) for x in a]
    return C_


def summarize(pt, reps):
    cp = contrasts(pt)
    S = {'auc': {m: [x[m] for x in pt] for m in pt[0]}, 'point': {k: {'per': v, 'mean': float(np.mean(v)), 'n_pos': int(sum(np.array(v) > 0))} for k, v in cp.items()}}
    for kd, rr in reps.items():
        cr = [contrasts(r) for r in rr]
        S['ci_' + kd] = {k: ci([np.mean(c[k]) for c in cr]) for k in cp}
    return S


res = {}
if part == 'main':
    pt, reps = stats(fr, 200, ['firm', 'two'])
    res['all'] = summarize(pt, reps)
    S = res['all']
    res['P2'] = {'mean': S['point']['pub_vs_PI']['mean'], 'ci_firm': S['ci_firm']['pub_vs_PI'], 'ci_two': S['ci_two']['pub_vs_PI'],
                 'pass_offjudgment': bool(S['ci_firm']['pub_vs_PI'][0] > 0 and S['ci_two']['pub_vs_PI'][0] > 0)}
    res['P4'] = {'D': S['point']['d_v']['mean'], 'd_v': S['point']['d_v']['per'], 'n_pos': S['point']['d_v']['n_pos'],
                 'ci_firm': S['ci_firm']['d_v'], 'ci_two': S['ci_two']['d_v'],
                 'pass_offjudgment': bool(S['ci_firm']['d_v'][0] > 0 and S['ci_two']['d_v'][0] > 0 and S['point']['d_v']['n_pos'] == 5)}
    e1 = S['point']['E_prov_to_pub']['mean']; e2 = S['point']['E_pub_to_prov']['mean']
    res['ref_i'] = {'E_prov_to_pub': e1, 'E_pub_to_prov': e2, 'ci_firm_E_prov_to_pub': S['ci_firm']['E_prov_to_pub'],
                    'ci_firm_E_pub_to_prov': S['ci_firm']['E_pub_to_prov'],
                    'refuted': bool(S['ci_firm']['E_prov_to_pub'][0] <= 0 or e2 >= e1)}
    res['ref_ii'] = {'prov_vs_scagg': S['point']['prov_vs_scagg']['mean'], 'ci_firm': S['ci_firm']['prov_vs_scagg'],
                     'refuted': bool(S['ci_firm']['prov_vs_scagg'][0] <= 0)}
else:
    # (iii) 대형 출원인 제외
    allf = pd.concat(fr)
    pc = allf[allf.y == 1].groupby(['cc', 'aid']).size().sort_values(ascending=False)
    top10 = set(pc.index[:10])
    res['top10_pos_share'] = float(pc.iloc[:10].sum() / pc.sum())
    res['top10'] = [f'{a}:{b}' for a, b in pc.index[:10]]
    out('top10 share', res['top10_pos_share'])
    subs = {}
    subs['ex_top10'] = [f[~pd.MultiIndex.from_arrays([f.cc, f.aid]).isin(top10)].reset_index(drop=True) for f in fr]
    ex1 = []
    for f in fr:
        fl = f.drop_duplicates(['cc', 'aid'])[['cc', 'aid', 'f_log3']]
        thr = fl.f_log3.quantile(0.99)
        big = set(map(tuple, fl[fl.f_log3 >= thr][['cc', 'aid']].values))
        ex1.append(f[~pd.MultiIndex.from_arrays([f.cc, f.aid]).isin(big)].reset_index(drop=True))
    subs['ex_top1pct'] = ex1
    subs['ex_missc'] = [f[f.f_missc == 0].reset_index(drop=True) for f in fr]
    subs['CN'] = [f[f.cc == 'CN'].reset_index(drop=True) for f in fr]
    subs['EP'] = [f[f.cc == 'EP'].reset_index(drop=True) for f in fr]
    for nm, frs in subs.items():
        pt, reps = stats(frs, 200, ['firm'], models=['SB', 'PI', 'pub', 'prov', 'free', 'hsc'])
        cp = contrasts(pt)
        cr = [contrasts(r) for r in reps['firm']]
        res[nm] = {'n_rows': [len(f) for f in frs], 'n_pos': [int(f.y.sum()) for f in frs],
                   'D': float(np.mean(cp['d_v'])), 'd_v': cp['d_v'], 'ci_firm_D': ci([np.mean(c['d_v']) for c in cr]),
                   'prov_vs_pub': float(np.mean(cp['prov_vs_pub'])), 'ci_firm_prov_vs_pub': ci([np.mean(c['prov_vs_pub']) for c in cr]),
                   'pub_vs_PI': float(np.mean(cp['pub_vs_PI'])),
                   'auc': {m: [x[m] for x in pt] for m in pt[0]}}
        out(nm, res[nm]['D'], res[nm]['ci_firm_D'], round(time.time() - t, 1))
    r3 = [res['ex_top10'], res['ex_top1pct']]
    res['ref_iii'] = {'refuted': bool(any(r['D'] <= 0 or r['ci_firm_D'][0] <= 0 for r in r3))}
res['secs'] = round(time.time() - t, 1)
json.dump(res, open(C + rf'\eval_{part}.json', 'w'), indent=1)
out(json.dumps({k: v for k, v in res.items() if k != 'all'})[:3000])
out('done', res['secs'])
