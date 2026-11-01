# 버전 실행기: python 20_run.py v1 [v2 ...]  (정의는 versions.py)
import sys, json, time
import pandas as pd
from t6lib import *
from versions import VERSIONS
base = json.load(open('baseline.json'))
BMAE = base['MAE']
bpred = pd.read_parquet('preds/baseline.parquet')
F = load_panel(with_patent=True)
for v in sys.argv[1:]:
    t = time.time()
    V = VERSIONS[v]
    pred = rolling_eval(F, V['fp'], train_filter=V.get('train_filter'))
    ch = getattr(V['fp'], 'chosen', None)
    if ch: V['note'] = V.get('note', '') + ' 선택=' + str(ch).replace(',', ' ')
    if 'post' in V:
        pred = V['post'](pred)
    lo, hi = boot_ci(pred, bpred)
    mt, sk = log_version(v, V['change'], V['hyp'], pred, BMAE, V.get('kept', '?'),
                         note=f'skill95CI(산업부트)=[{lo:.3f},{hi:.3f}]; ' + V.get('note', ''))
    print('  연도별 MAE', by_year(pred).round(4).to_dict(), f'{time.time()-t:.0f}s')
