import pandas as pd, numpy as np, json, os, glob
MY = os.path.dirname(os.path.abspath(__file__))
P0 = pd.read_parquet(os.path.join(MY, 'v4_preds_S0.parquet')).set_index('cell'); y = P0.yb == 1; n = len(P0); k = int(round(.1 * n))
r0 = P0.p.rank(ascending=False, method='first'); t0 = r0 <= k; s0 = json.load(open(os.path.join(MY, 'v4_res_S0.json')))['skill']; rows = []
for f in sorted(glob.glob(os.path.join(MY, 'v4_preds_*.parquet'))):
    s = os.path.basename(f)[9:-8]
    if s == 'S0': continue
    P = pd.read_parquet(f).set_index('cell').loc[P0.index]; r = P.p.rank(ascending=False, method='first'); t = r <= k
    rows.append(dict(scen=s, dskill=json.load(open(os.path.join(MY, f'v4_res_{s}.json')))['skill'] - s0, top_swapped=int((t0 & ~t).sum()),
                     hits_lost=int((t0 & ~t & y).sum()), hits_gained=int((~t0 & t & y).sum()), mean_abs_rank_move=float((r - r0).abs().mean()),
                     mean_abs_rank_move_hits=float((r - r0)[t0 & y].abs().mean())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(MY, 'v4_null_compare.csv'), index=False, encoding='utf-8-sig'); print(R.round(4).to_string())
