# 챔피언 실험 설정을 final_spec_r2.json으로 동결: python R2/make_spec.py E14
import sys, json, pathlib, datetime
R2 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R2))
from r2exps import EXPS, resolve
eid = sys.argv[1]
chain = []; e = eid
while e is not None:
    chain.append(dict(exp=e, change=EXPS[e].get('change', ''))); e = EXPS[e].get('base')
summ = json.load(open(R2/'runs'/eid/'summary.json', encoding='utf-8'))
start = json.load(open(R2/'runs'/'E00'/'summary.json', encoding='utf-8'))
spec = dict(
    track='T1 산업 대기실 — 기업×산업 첫 진입 예측 (2라운드 동결)', frozen=str(datetime.date.today()), exp_id=eid,
    config=resolve(eid), lineage=list(reversed(chain)),
    task=dict(unit='(법인 applicant_cd × KSIC 세세분류 50) 위험집합 쌍', cluster='법인',
              entry_label='공개 기준: (c, c+5] 출원 범위안 특허가 등록(registry 원출원인 코드) 또는 공개(kr_patents 출원인명→법인 별칭 매칭)된 경우 1 — 데이터 끝 2025-08-22',
              risk_set='기준시점 c 이하 출원 중 그 KSIC 등록 0건(1라운드 정의) AND 그 KSIC 공개·등록 선행출원 0건(2라운드 추가, 공개 기준 판정과 일관)',
              why_public='등록 기준은 최종시험 창(2018–22 출원)에서 심사 지연으로 절단(2012–14 코호트 기준 2021·2022 출원의 관측 등록 비율 59%·52% 추정, 외국법인 44%·27%) → 개발 창(≤2017, 거의 완결)과 레이블 완결도가 달라짐. 공개(18개월)는 두 구간 모두 완결.'),
    origins=dict(dev=[2008, 2009, 2010, 2011, 2012], dev_rule='학습 원점 c−8..c−5, 레이블·분류기·IPW·공개 보강 cap=c-12-31, 평가 진입창 (c, c+5] ≤ 2017',
                 final=dict(c=2017, window='2018–2022', F=2018, train_cuts=[2009, 2010, 2011, 2012], cap='20171231'),
                 future=dict(c=2024, window='2025–2029', train_cuts=[2014, 2015, 2016, 2017], cap='20250822')),
    metric='전체 AUC skill = (AUC_모형 − AUC_최강기준선)/(1 − AUC_최강기준선). 최강기준선 = 고정 기준선 6변형(로지스틱 스택/마지막원점 × XGB, 학습 레이블 등록/공개기준) 중 원점별 최댓값. 채택: 평균 skill +0.003 이상 AND 원점 60% 이상 우세.',
    dev_result=dict(mean_skill=summ['mean_skill'], mean_skill_new=summ['mean_skill_new'], per_origin={k: dict(auc_all=v['auc_all'], skill=v['skill'], auc_new=v['auc_new'], skill_new=v['skill_new']) for k, v in summ['per_origin'].items()}),
    start_champion=dict(exp='E00', mean_skill=start['mean_skill'], mean_skill_new=start['mean_skill_new']),
    compute='음성 25% 무작위 추출(RandomState 1234+c, 음성 가중 4)·XGB 2시드·nthread 2·float32 — 4GB·9분 규칙',
    run='python R2/final_run_r2.py --mode dev|final|future (종료코드 3이면 같은 명령 재실행)')
json.dump(spec, open(R2/'final_spec_r2.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('saved final_spec_r2.json', eid, summ['mean_skill'])
