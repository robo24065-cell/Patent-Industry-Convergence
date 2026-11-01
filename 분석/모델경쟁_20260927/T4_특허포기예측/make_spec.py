# -*- coding: utf-8 -*-
"""final_spec.json 생성기. 사용: python make_spec.py [--extras mac,ag,ow] [--out final_spec.json]
v37 구성(4개 모형 동일가중 로짓평균)을 기준으로, 감사 수정(시점보정 패밀리·dspst)을 항상 적용하고 선택한 추가 피처를 각 구성요소에 넣는다."""
import json, argparse, os
import feat2 as F2
from final_model import CLS_N, HAZ_N, LOG_N, LOG1P, XGBP
from versions import F6C

ap = argparse.ArgumentParser()
ap.add_argument('--extras', default='')
ap.add_argument('--out', default='final_spec.json')
ap.add_argument('--weights', default='0.25,0.25,0.25,0.25')
ap.add_argument('--cap_cls', type=int, default=1_200_000)
ap.add_argument('--cap_haz', type=int, default=1_200_000)
ap.add_argument('--hgb_max_rows', type=int, default=1_200_000)
ap.add_argument('--n_bag', type=int, default=1)
a = ap.parse_args()
ex = [e for e in a.extras.split(',') if e]
EXTRA = []
if 'mac' in ex: EXTRA += F2.MACRO
if 'ag' in ex: EXTRA += F2.AGENT
if 'ow' in ex: EXTRA += F2.OWNER
w = [float(x) for x in a.weights.split(',')]
xp = dict(XGBP)
spec = {
    'track': 'T4 특허 포기(등록료불납) 예측',
    'version': 'v37-audit' + ('+' + '+'.join(ex) if ex else '') + (f'+bag{a.n_bag}' if a.n_bag > 1 else ''),
    'C': {'dryrun': '2014-01-01', 'holdout': '2019-01-01', 'future': '2026-07-01'},
    'eval_L': {'dryrun': ['2014-01-01', '2015-07-01'], 'holdout': ['2019-01-01', '2021-07-01'], 'future': ['2012-01-01', '2026-01-01']},
    'train': {'first_landmark': '2005-01-01', 'cls_label_rule': '반기 기준일 L, L+3년+400일 <= C (label_asof: C 시점 공개정보만)',
              'haz_label_rule': '연 1회 기준일 2005..(C연도-2), 결정 d+450일 <= C, 납부 pd<=C / 소멸등록 cd<=C 기록에서만',
              'weight_base_year': 2005, 'weight_div': 2.0, 'cap_rows_cls': a.cap_cls, 'cap_rows_haz': a.cap_haz,
              'cap_rule': '학습행이 상한을 넘으면 기준일마다 같은 비율로 무작위 추출(seed 0). 기준선은 추출 없이 전량.'},
    'categorical': F6C,
    'audit_fixes': ['fam_dated: 해외 패밀리 토큰을 추정 공개연도 <= L연도-1 인 것만 집계(스냅샷 누수 제거)',
                    'fix_dspst: 등록 후 사건(이의신청후 등록유지·이의신청에 의한 등록취소) 최종처분 값을 등록결정(일반)으로'],
    'extras': ex,
    'components': {
        'cls': {'type': 'xgboost 3년 분류형', 'num': CLS_N + EXTRA, 'params': xp, 'rounds': 600, 'n_bag': a.n_bag, 'sec_hint': 260},
        'haz': {'type': 'xgboost 랜드마크 이산시간 위험모형, P=1-prod(1-h_j)', 'num': HAZ_N + EXTRA, 'params': xp, 'rounds': 600, 'n_bag': a.n_bag, 'sec_hint': 340},
        'logit': {'type': 'sklearn LogisticRegression', 'num': LOG_N + EXTRA, 'log1p': LOG1P, 'C': 0.5, 'max_iter': 3000, 'max_rows': 800_000},
        'hgb': {'type': 'sklearn HistGradientBoosting', 'num': CLS_N + EXTRA, 'max_rows': a.hgb_max_rows,
                'params': dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=48, min_samples_leaf=200, l2_regularization=1.0)},
    },
    'ensemble': {'method': '확률 로짓의 가중평균(가중치 사전 고정, 검증으로 튜닝하지 않음)', 'weights': dict(zip(['cls', 'haz', 'logit', 'hgb'], w))},
    'baseline': {'desc': '고정 기준선 로지스틱: 경과연수 + 출원인유형 + KSIC + log(청구항 수) + 등록연도', 'num': ['age', 'claims', 'rgst_year'],
                 'cat': ['atype', 'ksic'], 'C': 1.0, 'min_frequency': 20, 'train_rows': '분류형과 같은 기준일·레이블, 추출 없이 전량'},
    'bootstrap': {'B': 200, 'cluster': '1순위 출원인코드(없으면 특허 단위)'},
    'future': {'universe_complete_until': '2025-07-01', 'trend_base_L': '2014-01-01',
               'note': 'KR 등록 데이터 최종일 2025-08-22 → 2026-01-01 기준일은 그 이후 등록분 누락(universe_complete=False)'},
    'paths': {'ksic_names': r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv'},
}
if os.path.exists(a.out):
    old = json.load(open(a.out, encoding='utf-8'))
    if 'validation' in old and old.get('version') == spec['version']:
        spec['validation'] = old['validation']
json.dump(spec, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', a.out, spec['version'], {k: len(v['num']) for k, v in spec['components'].items()})
