# T1 R2 실험 등록부: 각 실험 = base(부모 설정) + delta(변경 1개). resolve()가 부모 사슬을 따라 전체 설정을 만든다.
import copy, json, pathlib
SPEC1 = json.load(open(pathlib.Path(__file__).resolve().parent.parent/'final_spec.json', encoding='utf-8'))

V38 = dict(cols=list(SPEC1['cols']), params={}, seeds=[0, 1, 2], neg_q=None, blend_w=0.9, ipw='reg', publab_w=1.0,
           train_filter_prior=False, ext=[], train_cuts_n=4, f_any_label='reg')

EXPS = {
    'E00': dict(base=None, cfg=V38, change='출발 챔피언: 1라운드 동결 v38 그대로(전체 학습표본, 3시드)', decision='START'),
    'E00f': dict(base='E00', delta=dict(neg_q=0.25, seeds=[0, 1]),
                 change='[계산 체계] 음성 25% 추출(가중 4)+2시드 — 4GB·9분 규칙 준수용. 이후 모든 실험의 비교 기준(같은 체계끼리 쌍 비교)',
                 decision='REBASE'),
}

IO_COLS = ['io_down', 'io_up', 'io_sell', 'io_max', 'io_same']
TRADE_COLS = ['tr_imp_l', 'tr_imp_g1', 'tr_imp_g3', 'tr_cn_sh', 'tr_cn_g3']
EXPORT_COLS = ['tr_exp_l', 'tr_exp_g3', 'tr_nx']
KOSIS_COLS = ['ks_est', 'ks_emp', 'ks_ship', 'ks_va']
FLOW_COLS = ['fl_mix', 'fl_max', 'dv_new3', 'dv_new_p3', 'dv_held']
A = 'E00f'   # 1차 선별 라운드의 챔피언
EXPS.update({
    'E01': dict(base=A, delta=dict(train_filter_prior=True), change='학습 위험집합도 R2 판정과 일치: 기준시점 이하 공개·등록 선행출원이 있는 쌍 제외'),
    'E02': dict(base=A, delta=dict(ipw='none'), change='등록지연 IPW 제거(공개 기준 레이블에서는 불필요 가설)'),
    'E03': dict(base='E04', delta=dict(f_any_label='union'), change='법인 단계 f_any 레이블을 등록∪공개로'),
    'E04': dict(base=A, delta=dict(add_ext=['io'], add_cols=IO_COLS), change='[외부1 산업연관표] 기업 보유산업 믹스×2010 IO 투입·산출 연결(전방·후방·판매비중·최대·동일부문) 5피처'),
    'E05': dict(base='E04', delta=dict(add_ext=['trade'], add_cols=TRADE_COLS), change='[외부2 무역] 목표 산업 HS6 수입 규모·1/3년 증가·중국 비중·중국 증가(t−1) 5피처'),
    'E06': dict(base='E04', delta=dict(add_ext=['trade_x'], add_cols=TRADE_COLS + EXPORT_COLS), change='[외부2′ 무역] 수입+수출(규모·증가·순수출비) 8피처'),
    'E07': dict(base='E04', delta=dict(add_ext=['kosis'], add_cols=KOSIS_COLS), change='[외부3 KOSIS] 광업제조업조사 5자리 산업 규모(2008 값 고정, 구조 속성) 4피처'),
    'E08': dict(base=A, delta=dict(add_ext=['flow'], add_cols=FLOW_COLS), change='[내부 KSIC] 최근 3년 산업 간 진입 흐름 행렬(보유산업→목표산업)·법인 다각화 모멘텀 5피처'),
    'E09': dict(base='E04', delta=dict(params=dict(max_depth=8)), change='XGB max_depth 6→8'),
    'E10': dict(base='E16', delta=dict(params=dict(colsample_bytree=0.6)), change='XGB colsample_bytree 0.4→0.6'),
    'E11': dict(base=A, delta=dict(blend_w=1.0), change='로지스틱 혼합 제거(XGB 단독)'),
    'E12': dict(base='E16', delta=dict(train_cuts_n=3), change='학습 원점 4→3개(가장 오래된 원점 제외)'),
    'E13': dict(base='E16', delta=dict(neg_q=0.5), change='음성 추출 25%→50%'),
    'E14': dict(base='E24', delta=dict(add_cols=['k']), change='산업 코드(KSIC 정렬 순번) 직접 피처'),
    'E16': dict(base='E04', delta=dict(add_ext=['flow'], add_cols=FLOW_COLS), change='[조합] E04(IO) 챔피언 위에 E08(진입 흐름·다각화) 추가 — E08은 E00f 대비 채택'),
    'E15': dict(base=A, delta=dict(blend_w=0.8), change='혼합 가중 XGB 0.9→0.8(E11 롤백 후 반대 방향 변형)'),
})

LEO_COLS = ['leo_in', 'leo_out']
GLOB_COLS = ['gl_us_g', 'gl_cn_g', 'gl_jp_g', 'gl_ep_g', 'gl_for_g', 'gl_kr_sh']
C2 = 'E16'   # 2차 라운드 챔피언(IO + 진입 흐름)
EXPS.update({
    'E17': dict(base=C2, delta=dict(params=dict(objective='rank:pairwise')), change='XGB 목적함수 binary:logistic→rank:pairwise(AUC 직접 최적화)'),
    'E18': dict(base=C2, delta=dict(add_ext=['leo'], add_cols=LEO_COLS), change='[외부1′ 산업연관표] 생산유발계수(레온티에프 역행렬) 간접 연결 2피처'),
    'E19': dict(base=C2, delta=dict(add_ext=['glob'], add_cols=GLOB_COLS), change='[제공 5개국] 목표 산업의 해외(US·CN·JP·EP) 출원 모멘텀·KR 비중(공개분, 최근 2년 제외) 6피처'),
    'E20': dict(base=C2, delta=dict(params=dict(min_child_weight=20)), change='XGB min_child_weight 50→20'),
    'E21': dict(base=C2, delta=dict(publab_w=0.5), change='공개 보강 양성 가중 1.0→0.5'),
    'E22': dict(base=C2, delta=dict(add_ext=['trade_pair'], add_cols=['tp_home_g3', 'tp_home_cn', 'tp_gap_g3']), change='[외부2″ 무역] 쌍 수준: 기업 보유산업 믹스 가중 수입증가(전체·중국)와 목표산업과의 차이(t−1) 3피처'),
    'E23': dict(base=C2, delta=dict(params=dict(n_estimators=1200)), change='XGB 트리 800→1200'),
    'E25': dict(base='E24', delta=dict(ipw='none'), change='등록지연 IPW 제거(E02는 E00f 대비 +0.0018로 기준 미달 → 새 챔피언에서 재시험)'),
    'E26': dict(base=C2, delta=dict(cut_weight_step=0.25), change='학습 원점 최근성 가중(1, 1.25, 1.5, 1.75)'),
    'E27': dict(base='E24', delta=dict(add_ext=['flow5'], add_cols=['fl5_mix', 'fl5_max']), change='진입 흐름 5년 창 변형 2피처 추가'),
    'E28': dict(base='E24', delta=dict(publab_w=3.0), change='공개 보강 양성 가중 2.0→3.0(같은 방향 추가)'),
    'E29': dict(base='E24', delta=dict(pos_mult=2.0), change='전체 양성 가중 ×2(클래스 균형 쪽으로)'),
    'E30': dict(base='E24', delta=dict(params=dict(min_child_weight=100)), change='XGB min_child_weight 50→100(E20의 반대 방향)'),
    'E31': dict(base='E24', delta=dict(drop_cols=IO_COLS, ext=['flow']), change='[절제] 최종 후보(E24)에서 산업연관표 5피처 제거 — 외부데이터 한계기여 확인'),
    'E32': dict(base='E14', delta=dict(blend_w=1.0), change='[채점만] 새 챔피언 E14에서 로지스틱 혼합 제거(XGB 단독)'),
    'E33': dict(base='E14', delta=dict(blend_w=0.8), change='[채점만] 새 챔피언 E14에서 혼합 가중 0.9→0.8'),
    'E34': dict(base='E14', delta=dict(seeds=[0, 1, 2]), change='XGB 시드 2→3(평균 분산 감소; 시드 0·1은 E14와 동일 표본·동일 모형 재사용)'),
    'E35': dict(base='E14', delta=dict(add_ext=['ipcsim'], add_cols=['ipc_mix', 'ipc_max']), change='[제공 IPC×KSIC] 산업 간 기술 근접도(KSIC별 IPC 서브클래스 분포 코사인)×기업 산업 믹스 2피처'),
    'E36': dict(base='E14', delta=dict(params=dict(learning_rate=0.03)), change='XGB 학습률 0.02→0.03'),
    'E37': dict(base='E35', delta=dict(add_ext=['ipcfit'], add_cols=['ipc_fit', 'ipc_fit_rank']), change='[제공 IPC×KSIC] 기업 자체 IPC 프로파일 × 목표 KSIC IPC 프로파일 코사인·순위 2피처'),
    'E38': dict(base='E37', delta=dict(add_ext=['ipcfit_g'], add_cols=['ipcg_fit', 'ipcg_fit_rank']), change='IPC 적합도를 메인그룹 수준으로도 추가(서브클래스→메인그룹 세분화) 2피처'),
    'E39': dict(base='E37', delta=dict(drop_cols=IO_COLS, ext=['flow', 'ipcsim', 'ipcfit']), change='[절제] E37에서 산업연관표 5피처 제거 — IPC 근접도 추가 후에도 외부데이터(IO)가 기여하는지'),
    'E24': dict(base=C2, delta=dict(publab_w=2.0), change='공개 보강 양성 가중 1.0→2.0(E21 0.5가 크게 악화 → 반대 방향 변형)'),
})


def resolve(eid):
    e = EXPS[eid]
    if e.get('base') is None: return copy.deepcopy(e['cfg'])
    c = resolve(e['base']); d = copy.deepcopy(e.get('delta', {}))
    if 'add_cols' in d: c['cols'] = c['cols'] + [x for x in d.pop('add_cols') if x not in c['cols']]
    if 'drop_cols' in d: dc = d.pop('drop_cols'); c['cols'] = [x for x in c['cols'] if x not in dc]
    if 'add_ext' in d: c['ext'] = c['ext'] + [x for x in d.pop('add_ext') if x not in c['ext']]
    if 'params' in d: p = dict(c['params']); p.update(d.pop('params')); c['params'] = p
    c.update(d); return c
