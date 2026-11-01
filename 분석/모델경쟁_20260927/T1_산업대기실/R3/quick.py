# 부분 점검: 실험별 완료 원점의 skill(챔피언 대비)
import sys, json, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
import r3run
vs = sys.argv[1]; rv = json.load(open(R3/'runs'/vs/'summary.json', encoding='utf-8'))['per_origin']
for e in sys.argv[2:]:
    out = []
    for c in ORIGINS:
        if (R3/'runs'/e/f'o{c}.parquet').exists():
            m, _ = r3run.score_origin(e, c); b = baselines(c)['strongest']; s = skill(m['auc_all'], b['auc_all'])
            out.append(f'{c}:{s:.4f}({s - rv[str(c)]["skill"]:+.4f})')
    print(e, ' '.join(out))
