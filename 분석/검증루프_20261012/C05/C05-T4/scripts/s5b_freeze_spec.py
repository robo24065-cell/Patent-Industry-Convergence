# s5b: 동결 — 잠금 점수 파일(레이블 없음) 저장 + sha256, final_spec_C05T4.json 작성.
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
from libM import FEATS, XGB_PARAMS, LR_PARAMS, M_TE
rule = json.load(open(os.path.join(C, 'champion_rule.json'), encoding='utf-8'))
champ = rule['champion']
fz = None
for m in ['M0', 'M1', 'M1_pub_TXT', 'M1_pub_IPC']:
    d = pd.read_parquet(os.path.join(C, f'lockscore_{m}.parquet'))[['key', 'p']].rename(columns={'p': f'score_{m}'})
    fz = d if fz is None else fz.merge(d, on='key', validate='one_to_one')
fz['score_champion'] = fz[f'score_{champ}']
assert 'y' not in fz.columns and fz.key.is_unique
fz = fz[['key', 'score_champion', 'score_M0', 'score_M1', 'score_M1_pub_TXT', 'score_M1_pub_IPC']]
p = os.path.join(W, 'frozen_scores.parquet'); fz.to_parquet(p, index=False)
hz = sha256(p)
open(os.path.join(W, 'frozen_scores.sha256'), 'w', encoding='utf-8').write(f'{hz}  frozen_scores.parquet\n')
code = ['scripts/libC.py', 'scripts/libM.py', 'scripts/libB.py', 'scripts/s1_units.py', 'scripts/s2_bridge.py', 'scripts/s3_dev_fit.py',
        'scripts/s4_dev_eval.py', 'scripts/s5_freeze_fit.py', 'scripts/s5b_freeze_spec.py', 'scripts/s6_lock_score.py', 'PREREG.txt']
spec = dict(task='C05-T4 무특허 NPS 법인의 첫 제공범위 KR 특허 공개(추가 실험, 공식 외)', champion=champ, champion_rule=rule,
            features={m: FEATS[m] for m in FEATS}, champion_features=FEATS[champ],
            hyperparameters=dict(xgb=XGB_PARAMS, logistic=LR_PARAMS, ensemble='mean(p_lr, p_xgb)', target_encoding_m=M_TE,
                                 pen_flow_smoothing_m=50, bridge_smoothing_m=5),
            train_origins=DEV, lock=dict(S0=LOCK, ref=REF[LOCK], window='2025-01-01..2025-06-30', n_units=int(len(fz))),
            frozen_scores_sha256=hz, prereg_final_sha256=sha256(os.path.join(W, 'PREREG_final.txt')),
            code_sha256={c: sha256(os.path.join(W, c)) for c in code},
            input_sha256={'units.parquet': sha256(os.path.join(C, 'units.parquet')), 'bridge.parquet': sha256(os.path.join(C, 'bridge.parquet'))},
            note='score_champion은 챔피언 모형 점수. M0·M1·M1_pub 점수는 skill 기준·판정 밖 보고용. 레이블 없음.')
json.dump(spec, open(os.path.join(W, 'final_spec_C05T4.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('champion', champ, 'n', len(fz), 'sha', hz)
