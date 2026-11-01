# -*- coding: utf-8 -*-
"""verify_1: S0(공식 0.42289·AUC 0.79354) 재현 확인 — sklearn roc_auc_score 로 공식 예측 파일과 C11 S0 예측을 각각 독립 채점, p 비교.
개발 9원점 평균 skill 0.4275(final_spec dev.mean_skill)도 C11 S0 dev 예측으로 재계산. 민감도 V1·V2·V3 최종 skill 도 같은 방식으로 다시 계산."""
import json, sys
sys.dont_write_bytecode = True
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
R3 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측/R3"
C = BASE + "/분석/검증루프_20261012/C11/C11-T3/cache"
OUT = BASE + "/분석/검증루프_20261012/C11/C11-T3/verify_1"
def sk(y, p, b):
    a, ab = roc_auc_score(y, p), roc_auc_score(y, b); return (a - ab) / (1 - ab), a, ab
off = pd.read_parquet(R3 + "/preds/FINAL_R3_t0_2021.parquet")
mine = pd.read_parquet(C + "/preds_S0_final.parquet")
fr = json.load(open(R3 + "/final_result_r3.json", encoding="utf-8"))
res = {}
res['official_file'] = dict(zip(['skill', 'auc', 'base_auc'], sk(off.yb.values, off.p.values, off.p_base.values)))
res['c11_S0'] = dict(zip(['skill', 'auc', 'base_auc'], sk(mine.yb.values, mine.p.values, mine.p_base.values)))
m = off.merge(mine, on='cell', suffixes=('_o', '_m'))
res['n_merge'] = len(m); res['max_abs_dp'] = float((m.p_o - m.p_m).abs().max()); res['max_abs_dbase'] = float((m.p_base_o - m.p_base_m).abs().max())
res['label_mismatch'] = int((m.yb_o != m.yb_m).sum())
res['final_result_json_keys'] = list(fr.keys())[:20]
dev = pd.concat([pd.read_parquet(C + f"/preds_S0_dev_{s}.parquet") for s in ('2010-2011-2012-2013-2014', '2015-2016-2017-2018')])
res['dev_mean_skill'] = float(np.mean([sk(g.yb.values, g.p.values, g.p_base.values)[0] for _, g in dev.groupby('t0')]))
spec = json.load(open(R3 + "/final_spec_r3.json", encoding="utf-8"))
res['dev_spec_mean_skill'] = spec['dev']['mean_skill']
for V in ('V1', 'V2', 'V3'):
    x = pd.read_parquet(C + f"/preds_{V}_final.parquet"); res[f'{V}_final_skill'] = sk(x.yb.values, x.p.values, x.p_base.values)[0]
res['verdict'] = bool(abs(res['official_file']['skill'] - 0.42289) < 5e-6 and abs(res['c11_S0']['auc'] - 0.79354) < 5e-6 and res['max_abs_dp'] == 0.0
                      and abs(res['dev_mean_skill'] - res['dev_spec_mean_skill']) < 1e-9)
json.dump(res, open(OUT + "/v1_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
