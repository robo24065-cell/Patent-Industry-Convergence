# -*- coding: utf-8 -*-
"""P0: 공식 저장 예측 + 공식 레이블을 트랙 채점 경로(r2.mets·skill·final_run_r3.blend·topk)로 채점해 비트 비교 → p0_result.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c17t1 import *
import final_run_r3 as F
from r2 import mets, skill
fr_ = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
O = official_frame(); keep = ~O.prior_any.values
L15 = pd.read_parquet(C15/'cache'/'t1'/'final_labels.parquet'); assert (L15.firm.values == O.firm.values).all()
oo = O[keep].reset_index(drop=True); y = oo.y_off.values; nc = oo.newc.values.astype(bool)
s = F.blend(oo[['px', 'pl']].copy())
mm = mets(y, s, nc); bm = {k: mets(y, oo[k].values, nc) for k in BK}; st = max(v['auc_all'] for v in bm.values()); tk = F.topk(y, s)
res = dict(labels_eq_C15=bool((L15.y_u.values == O.y_off.values).all()), n=int(len(oo)), pos=int(y.sum()), newc_pairs=int(nc.sum()), newc_pos=int(y[nc].sum()),
           skill=skill(mm['auc_all'], st), skill_official=fr_['skill']['auc_all'], auc=mm['auc_all'], auc_official=fr_['model']['auc_all'],
           top1000=tk['1000']['hits'], top100=tk['100']['hits'], top1000_official=fr_['topk']['model']['1000']['hits'], M0=st, M0_official=fr_['baseline_strongest']['auc_all'],
           skill_new=skill(mm['auc_new'], max(v['auc_new'] for v in bm.values())) if 'auc_new' in mm else None, skill_new_official=fr_['skill'].get('auc_new'))
res['skill_diff'] = res['skill'] - res['skill_official']; res['auc_diff'] = res['auc'] - res['auc_official']
res['P0_pass'] = bool(res['skill_diff'] == 0 and res['auc_diff'] == 0 and res['top1000'] == 313 and res['labels_eq_C15'])
jdump(res, HERE/'p0_result.json'); print(json.dumps(res, ensure_ascii=False, indent=1))
