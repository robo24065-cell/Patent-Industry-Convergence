# B단계 챔피언 설정을 R3/final_spec_r3.json으로 동결: python R3/make_spec_r3.py R3E??
import sys, json, pathlib, datetime
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3exps import EXPS, resolve
eid = sys.argv[1]
chain = []; e = eid
while e is not None:
    chain.append(dict(exp=e, change=EXPS[e].get('change', ''))); e = EXPS[e].get('base')
summ = json.load(open(R3/'runs'/eid/'summary.json', encoding='utf-8'))
aux = R3/'runs'/eid/'summary_aux.json'
start = json.load(open(R3/'runs'/'R3E11'/'summary.json', encoding='utf-8'))
cfg = resolve(eid); cfg.pop('nts_filter', None)
ext_names = {'io': '한국은행 산업연관표 2010(R2)', 'design': 'KIPRIS 디자인권 명칭→KSIC(R3E01)', 'um': 'KIPRIS 범위밖 실용신안 명칭→KSIC(R3E11)',
             'price': 'ECOS 생산자·수출입 물가(R3E11)', 'oa': 'OpenAlex 기술셀(IPC 메인그룹) 논문(r3oa)', 'ros': 'Reliance on Science 특허→논문 인용(r3ros)',
             'ipcx': '제공 KR 공개·등록 출원 전체 IPC 서브클래스 적합도(내부, R3E20)', 'ipcm': '제공 KR 공개·등록 출원 전체 IPC 메인그룹 적합도(내부, R3E29)', 'entfit': '최근 진입자 닮음(내부, R3E29)', 'peer': '기술 이웃 법인 보유(내부, R3E29)', 'flow': '제공 KSIC 진입 흐름(내부)', 'ipcsim': '제공 IPC×KSIC 근접도(내부)', 'ipcfit': '제공 IPC×KSIC 적합도(내부)'}
spec = dict(
    track='T1 산업 대기실 — 기업×산업 첫 진입 예측 (3라운드 B단계 동결)', frozen=str(datetime.date.today()), exp_id=eid,
    config=cfg, lineage=list(reversed(chain)),
    task=dict(unit='(법인 applicant_cd × KSIC 세세분류 50) 위험집합 쌍', cluster='법인', h=4,
              entry_label='공개 기준: (c, c+4] 출원 범위안 특허가 등록(registry 원출원인 코드) 또는 공개(kr_patents 출원인명→법인 별칭 매칭)된 경우 1 — 데이터 끝 2025-08-22',
              risk_set='기준시점 c 이하 출원 기준 ≥3건 법인 × 그 KSIC 등록 0건 AND 그 KSIC 공개·등록 선행출원 0건(R2 정의 그대로)'),
    origins=dict(dev=[2008, 2009, 2010, 2011, 2012], dev_rule='학습 원점 c−7..c−4, 레이블·분류기·IPW·공개 보강 cap=c-12-31, 평가 진입창 (c, c+4] ≤ 2016',
                 aux=[2015], aux_rule='채택 판정 미사용. 국민연금 보정층 적합·국세청 폐업 정제 효과 확인용(진입 2016–2019)',
                 final=dict(c=2019, window='2020–2023', F=2020, train_cuts=[2012, 2013, 2014, 2015], cap='20191231'),
                 future=dict(c=2024, window='2025–2028', train_cuts=[2016, 2017, 2018, 2019], cap='20250822')),
    metric='전체 AUC skill = (AUC_모형 − AUC_최강기준선)/(1 − AUC_최강기준선). 최강기준선 = 고정 기준선 6변형(로지스틱 스택/마지막원점 × XGB, 학습 레이블 등록/공개기준) 중 최댓값(부트스트랩 재표본마다 다시 고름). 법인 군집 부트스트랩 R=200.',
    external_in_model=[ext_names.get(x, x) for x in cfg['ext']],
    secondary_preregistered=['model_nts: 국세청 폐업일 ≤ 2019-12-31 법인의 쌍 점수 최저(위험집합 정제)',
                             'model_nps: 국민연금 2단 보정층(주 모형 점수 로짓 + NPS 6피처 로지스틱, c=2015에서 적합한 계수 R3/nps_layer_r3.json 동결, 2019-12 스냅샷 적용)',
                             'model_nps_nts: 둘 다'],
    dev_result=dict(mean_skill=summ['mean_skill'], mean_skill_new=summ['mean_skill_new'], mean_auc_all=summ['mean_auc_all'],
                    per_origin={k: dict(auc_all=v['auc_all'], skill=v['skill'], auc_new=v['auc_new'], skill_new=v['skill_new'], base_auc_all=v['base_auc_all']) for k, v in summ['per_origin'].items()}),
    aux_result_c2015=(json.load(open(aux, encoding='utf-8'))['per_origin'] if aux.exists() else
                      (json.load(open(R3/'dev_result_r3.json', encoding='utf-8'))['aux_c2015'] if (R3/'dev_result_r3.json').exists() else None)),
    frozen_dev_repro=(json.load(open(R3/'dev_result_r3.json', encoding='utf-8'))['max_abs_repro_diff'] if (R3/'dev_result_r3.json').exists() else None),
    start_champion=dict(exp='R3E11(A단계 챔피언)', mean_skill=start['mean_skill'], mean_skill_new=start['mean_skill_new']),
    compute='음성 25% 무작위 추출(RandomState 1234+c, 음성 가중 4)·XGB 2시드·nthread 2·float32, 로지스틱 10% 순위 혼합 — 단계별 별도 프로세스(4GB·9분 규칙)',
    run='bash R3/run_final_r3.sh dev|final|future (= python R3/final_run_r3.py --mode <m>, 종료코드 3이면 같은 명령 재실행)')
json.dump(spec, open(R3/'final_spec_r3.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('saved final_spec_r3.json', eid, summ['mean_skill'])
