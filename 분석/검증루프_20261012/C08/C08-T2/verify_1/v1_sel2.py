import sys, os, csv, json, random, itertools
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
W = sys.argv[1]
target = sorted(json.load(open(os.path.join(W, 'verify_0', 'v0_selection.json'), encoding='utf-8'))['random_ok'])
rows = {}
for fn in ['audit_v4_v7.csv', 'audit_v8_v9.csv']:
    rows[fn] = [(fn, r['block'], ('정확' if r['block'] == 'B25' else r['verdict'])) for r in csv.DictReader(open(os.path.join(W, fn), encoding='utf-8-sig'))]
hits = []
for order in [['audit_v4_v7.csv', 'audit_v8_v9.csv'], ['audit_v8_v9.csv', 'audit_v4_v7.csv']]:
    base = [x for fn in order for x in rows[fn] if x[2] == '정확']
    for fmt in ['full', 'stem', 'block']:
        f = {'full': lambda x: f'{x[0]}:{x[1]}', 'stem': lambda x: f"{x[0][:-4]}:{x[1]}", 'block': lambda x: x[1]}[fmt]
        for srt in [False, True]:
            lst = [f(x) for x in base]; lst = sorted(lst) if srt else lst
            idx_lists = {
             'rs': random.Random(20261013).sample(range(len(lst)), 10),
             'rs_obj': None,
             'dr': np.random.default_rng(20261013).choice(len(lst), 10, replace=False).tolist(),
             'drperm': np.random.default_rng(20261013).permutation(len(lst))[:10].tolist(),
             'RS': np.random.RandomState(20261013).choice(len(lst), 10, replace=False).tolist(),
             'RSperm': np.random.RandomState(20261013).permutation(len(lst))[:10].tolist(),
             'shuffle': None}
            l2 = lst[:]; random.Random(20261013).shuffle(l2); idx_lists['shuffle'] = [lst.index(v) for v in l2[:10]]
            random.seed(20261013); idx_lists['rs_obj'] = [lst.index(v) for v in random.sample(lst, 10)]
            for k, ii in idx_lists.items():
                picked = sorted(f'{base_x[0]}:{base_x[1]}' for base_x in [ [x for x in base if f(x) == lst[i]][0] for i in ii])
                if picked == target: hits.append((order[0], fmt, srt, k))
print('hits', hits)
json.dump({'hits': hits}, open(os.path.join(W, 'verify_1', 'v1_sel2.json'), 'w', encoding='utf-8'), ensure_ascii=False)
