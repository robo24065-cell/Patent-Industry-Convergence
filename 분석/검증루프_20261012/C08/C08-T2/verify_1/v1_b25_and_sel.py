# verify_1: B25 원천 재계산(r3lib 정의 y=log((fut+1)/(kr_W+1))) + verify_0 무작위 선택 재현 시도. 읽기 전용.
import sys, os, csv, json, random
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = sys.argv[1]
W = os.path.join(B, '검증루프_20261012', 'C08', 'C08-T2')
p = pd.read_parquet(os.path.join(B, '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3', 'preds', 'FINAL_R3_t0_2021.parquet'))
p['fut'] = (p.kr_W + 1) * np.exp(p.y) - 1
p['fut_r'] = np.rint(p.fut)
res = {'max_abs_round_err': float((p.fut - p.fut_r).abs().max()), 'n': len(p), 'pos': int(p.yb.sum())}
res['y_threshold_min_pos'] = float(p[p.yb == 1].y.min()); res['y_max_neg'] = float(p[p.yb == 0].y.max())
tab = {}
for k in [0, 1, 2, 3]:
    g = p[p.kr_W == k]
    ct = g.groupby('fut_r').yb.agg(['size', 'sum']).reset_index()
    tab[k] = {int(r.fut_r): (int(r['size']), int(r['sum'])) for _, r in ct.iterrows() if r.fut_r <= k + 8}
    pos = g[g.yb == 1]
    res[f'krW{k}_min_fut_pos'] = float(pos.fut_r.min()) if len(pos) else None
    res[f'krW{k}_min_increase_pos'] = float(pos.fut_r.min() - k) if len(pos) else None
res['table_fut:(n,pos)'] = tab
out = {'B25': res}
# 선택 재현
keys = []
for fn in ['audit_v4_v7.csv', 'audit_v8_v9.csv']:
    for r in csv.DictReader(open(os.path.join(W, fn), encoding='utf-8-sig')):
        v = r['verdict']
        if r['block'] == 'B25': v = '정확'  # verify_0 이전 판정
        if v == '정확': keys.append(f"{fn}:{r['block']}")
sel = json.load(open(os.path.join(W, 'verify_0', 'v0_selection.json'), encoding='utf-8'))
target = sorted(sel['random_ok'])
tries = {}
for name, lst in [('order', keys), ('sorted', sorted(keys))]:
    tries[name + '_random.sample'] = sorted(random.Random(20261013).sample(lst, 10))
    tries[name + '_np.choice'] = sorted(np.random.default_rng(20261013).choice(lst, 10, replace=False).tolist())
    tries[name + '_np.RandomState'] = sorted(np.random.RandomState(20261013).choice(lst, 10, replace=False).tolist())
out['selection'] = {'n_ok_recomputed': len(keys), 'n_ok_claimed': sel['n_ok'], 'match': {k: v == target for k, v in tries.items()}}
json.dump(out, open(os.path.join(W, 'verify_1', 'v1_b25_sel.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
