# 최선 후보(v31) 강건성: 시드 집합 변경, 연도별 skill, 산업 부트스트랩, 특허 절제, 연도 부트스트랩. 결과 diag/robust_v31.txt
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from t6lib import *
from models import *
from versions import COMPACT, SECT, PATC
F = load_panel()
b = pd.read_parquet('preds/baseline.parquet')
out = []
def sk(p):
    return 1 - (p.y - p.pred).abs().mean() / p.y.abs().mean()
for seeds in [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)]:
    p = rolling_eval(F, guard(xgb_bag(COMPACT, rank=True, seeds=seeds), min_rows=500), train_filter=SECT)
    out.append(f'seeds={seeds} skill={sk(p):.4f} MAE={metrics(p.y,p.pred)["MAE"]:.4f}')
p = pd.read_parquet('preds/v31.parquet')
yr = p.assign(a=(p.y - p.pred).abs(), a0=p.y.abs()).groupby('year')[['a', 'a0']].mean()
yr['skill'] = 1 - yr.a / yr.a0
out.append('연도별:\n' + yr.round(4).to_string())
lo, hi = boot_ci(p, b, B=5000)
out.append(f'산업 군집 부트스트랩 95% CI skill: [{lo:.4f}, {hi:.4f}]')
# 연도 군집 부트스트랩(활성 2013-2019)
rng = np.random.default_rng(1)
a = p[p.year >= 2013].assign(a=lambda d: (d.y - d.pred).abs(), a0=lambda d: d.y.abs()).groupby('year')[['a', 'a0']].sum()
vals = [1 - a.a.values[i].sum() / a.a0.values[i].sum() for i in [rng.integers(0, len(a), len(a)) for _ in range(5000)]]
out.append(f'연도 군집 부트스트랩(2013-2019) 95% CI: [{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]')
act = p[p.year >= 2013]
out.append(f'활성 연도(2013-2019, n={len(act)}) skill={sk(act):.4f}')
# 특허 절제 요약
for v in ['v31', 'v32', 'v33', 'v21', 'v23', 'v24', 'v26', 'v27', 'v8', 'v9', 'v10']:
    q = pd.read_parquet(f'preds/{v}.parquet')
    out.append(f'{v}: skill={sk(q):.4f} MAE={metrics(q.y,q.pred)["MAE"]:.4f} DIR={metrics(q.y,q.pred)["DIR"]:.3f}')
# 쌍대 비교: v32(특허有) - v31(특허無) 산업 부트스트랩
q = pd.read_parquet('preds/v32.parquet').merge(p[['ksic', 'year', 'pred']].rename(columns={'pred': 'p31'}), on=['ksic', 'year'])
q['d'] = (q.y - q.pred).abs() - (q.y - q.p31).abs()
g = q.groupby('ksic').d.sum().values
bs = [g[rng.integers(0, len(g), len(g))].sum() / len(q) for _ in range(5000)]
out.append(f'특허 추가(v32-v31) MAE 차이 평균={q.d.mean():.5f}, 95% CI [{np.percentile(bs,2.5):.5f}, {np.percentile(bs,97.5):.5f}] (양수=특허가 악화)')
txt = '\n'.join(out)
open('diag/robust_v31.txt', 'w', encoding='utf-8').write(txt)
print(txt)
