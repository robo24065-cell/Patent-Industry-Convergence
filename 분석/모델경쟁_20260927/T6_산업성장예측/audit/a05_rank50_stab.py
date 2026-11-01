# 감사 5: 예측행 50개 내 순위(rank50) 방식의 시드 안정성 + 쌍대 부트스트랩(vs 305개 기준 순위)
import sys, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H); os.chdir(H)
import numpy as np, pandas as pd, importlib
import final_run as fr
R = importlib.import_module('70_run2')
F = fr.build_panel(2019, 2019)
b = pd.read_parquet('preds/baseline.parquet'); bm = np.abs(b.y).mean()
def pb(a, c):
    m = a.merge(c, on=['ksic', 'year', 'y'], suffixes=('a', 'c'))
    m['d'] = (m.y - m.preda).abs() - (m.y - m.predc).abs()
    g = m.groupby('ksic').d.agg(['sum', 'count']); rng = np.random.default_rng(0)
    bs = [(lambda i: g['sum'].values[i].sum() / g['count'].values[i].sum())(rng.integers(0, len(g), len(g))) for _ in range(3000)]
    return round(m.d.mean(), 5), np.percentile(bs, [2.5, 97.5]).round(5)
for feats, mono_on, tag in [(R.COMPACT, False, 'COMPACT'), (R.F41, True, 'F41+mono')]:
    sk50, sk305, diffs = [], [], []
    for k in range(4):
        sp = R.S(features=feats, seeds=list(range(5 * k, 5 * k + 5)), xgb=(dict(monotone_constraints=R.mono(feats)) if mono_on else None))
        p50 = R.rank50_rolling(F, fr.VAL_YEARS, sp)[['ksic', 'year', 'y', 'pred']]
        p305 = fr.rolling(F, fr.VAL_YEARS, sp)[['ksic', 'year', 'y', 'pred']]
        sk50.append(1 - (p50.y - p50.pred).abs().mean() / bm); sk305.append(1 - (p305.y - p305.pred).abs().mean() / bm)
        if k == 0: diffs = pb(p50, p305)
    print(tag, 'rank50', np.round(sk50, 4), 'mean', round(np.mean(sk50), 4), '| rank305', np.round(sk305, 4), 'mean', round(np.mean(sk305), 4), '| diff(seed0-4) MAE50-MAE305', diffs)
