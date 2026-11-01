# T1 R3 실험 등록부: 각 실험 = base(부모) + delta(변경 1개). R3E00 = R2 동결 챔피언 E37 설정을 R3 개발 정의(4년창·학습 원점 c−7..c−4)로 재채점.
import copy, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent/'R2'))
import r2exps

EXPS = {
    'R3E00': dict(base=None, cfg=r2exps.resolve('E37'), change='출발 챔피언: R2 동결 E37 설정 그대로, R3 개발 정의(진입창 4년, 개발 원점 2008–12, 학습 원점 c−7..c−4)로 재채점', decision='START'),
}


def resolve(eid):
    e = EXPS[eid]
    if e.get('base') is None: return copy.deepcopy(e['cfg'])
    c = resolve(e['base']); d = copy.deepcopy(e.get('delta', {}))
    if 'add_cols' in d: c['cols'] = c['cols'] + [x for x in d.pop('add_cols') if x not in c['cols']]
    if 'drop_cols' in d: dc = d.pop('drop_cols'); c['cols'] = [x for x in c['cols'] if x not in dc]
    if 'add_ext' in d: c['ext'] = c['ext'] + [x for x in d.pop('add_ext') if x not in c['ext']]
    if 'drop_ext' in d: de = d.pop('drop_ext'); c['ext'] = [x for x in c['ext'] if x not in de]
    if 'params' in d: p = dict(c['params']); p.update(d.pop('params')); c['params'] = p
    c.update(d); return c

DES_COLS = ['ds_lsum', 'ds_max', 'ds_n20', 'ds_rec3', 'ds_n']
UM_COLS = ['um_lsum', 'um_max', 'um_n20', 'um_rec3', 'um_n']
ACQ_COLS = ['acq_k', 'acq_k_rec', 'acq_n']
LIC_COLS = ['lic_k', 'lic_k_rec', 'lic_n']
SCI_COLS = ['sci_c_g3', 'sci_kr_g3', 'sci_kr_l', 'sci_all_g3', 'sci_gap']
PRICE_COLS = ['pr_ppi_g3', 'pr_ppi_g1', 'pr_exp_g3', 'pr_imp_g3', 'pr_gap']
C0 = 'R3E00'
EXPS.update({
    'R3E01': dict(base=C0, delta=dict(add_ext=['design'], add_cols=DES_COLS), change='[외부 KIPRIS 디자인권] 기업 디자인 등록 명칭→KSIC 분류 노출(합·최대·0.2초과 수·최근3년·총수) 5피처 — 제품시장 진입이 특허 진입을 선행한다는 가설'),
    'R3E02': dict(base=C0, delta=dict(add_ext=['acq'], add_cols=ACQ_COLS), change='[외부 등록원부 권리이전] 원출원인이 아닌 법인이 목표 KSIC 범위안 특허를 넘겨받은 수·최근3년·기업 총 매입 3피처 — 기술 매입이 진입을 선행'),
    'R3E03': dict(base=C0, delta=dict(add_ext=['sci'], add_cols=SCI_COLS), change='[외부 OpenAlex] 목표 산업 기업논문(전세계·한국)·전체논문 점유 성장, 한국 기업논문 규모, 기업 보유산업 대비 과학성장 격차 5피처(연도 ≤ c−1)'),
    'R3E04': dict(base='R3E01', delta=dict(add_ext=['price'], add_cols=PRICE_COLS), change='[외부 ECOS 물가] 목표 산업 PPI 3년·1년 변화, 수출·수입물가 3년 변화, 기업 보유산업 대비 PPI 격차 5피처(연평균 ≤ c−1)'),
    'R3E05': dict(base='R3E01', delta=dict(add_ext=['um'], add_cols=UM_COLS), change='[외부 KIPRIS 실용신안(범위밖)] 기업 실용신안 명칭→KSIC 노출 5피처'),
    'R3E06': dict(base='R3E01', delta=dict(add_ext=['lic'], add_cols=LIC_COLS), change='[외부 등록원부 실시권] 목표 KSIC 특허의 전용·통상실시권자 여부·최근3년·기업 총 실시권 3피처'),
})

C1 = 'R3E01'   # 2차 챔피언(디자인권 채택)
EXPS.update({
    'R3E07': dict(base=C1, delta=dict(add_ext=['acq', 'lic'], add_cols=ACQ_COLS + LIC_COLS), change='[외부 등록원부 기술도입] 권리이전(매입)+실시권 6피처 묶음(E02 단독은 +0.0024로 기준 미달 → 실시권과 합쳐 새 챔피언 위에서)'),
    'R3E08': dict(base=C1, delta=dict(train_cuts_n=5), change='학습 원점 4→5개(c−8..c−4, 같은 cap·4년 레이블) — 최종시험(c=2019)에서 학습 원점 2011–2015 사용 여부 결정용'),
})
TM_COLS = ['tm_n', 'tm_rec3', 'tm_g']
EXPS.update({
    'R3E09': dict(base=C1, delta=dict(add_ext=['design_top'], add_cols=['ds_top', 'ds_share']), change='[외부 KIPRIS 디자인권 변형] 주분류(최대확률>0.5) 디자인 수·기업 디자인 중 목표 산업 비중 2피처 추가'),
    'R3E10': dict(base=C1, delta=dict(add_ext=['tm'], add_cols=TM_COLS), change='[외부 KIPRIS 상표권] 기업 상표 등록 누적·최근3년·증가 3피처(기업 수준) — 브랜드 확장이 다각화를 선행'),
})
EXPS.update({
    'R3E11': dict(base=C1, delta=dict(add_ext=['um', 'price'], add_cols=UM_COLS + PRICE_COLS), change='[외부 결합] 개별로는 기준 미달(+0.0024·+0.0023)이지만 원점 4/5·5/5 우세였던 실용신안(KIPRIS)+ECOS 물가를 함께 추가 10피처'),
})
EXPS.update({
    'R3E12': dict(base=C1, delta=dict(add_ext=['um', 'tm'], add_cols=UM_COLS + TM_COLS), change='[외부 KIPRIS 권리 포트폴리오 결합] 개별 기준 미달(+0.0024 4/5, +0.0021 5/5)인 범위밖 실용신안+상표권을 함께 추가 8피처'),
})
C2 = 'R3E11'   # 3차 챔피언(디자인+실용신안+물가)
EXPS.update({
    'R3E13': dict(base=C2, delta=dict(add_ext=['tm'], add_cols=TM_COLS), change='[외부 KIPRIS 상표권] E12(실용신안+상표) 병렬 채택분 중 새 챔피언(E11)에 없는 상표 3피처만 추가'),
    'R3E14': dict(base=C2, delta=dict(train_cuts_n=5), change='학습 원점 4→5개(c−8..c−4) — E08(+0.0025, 5/5 기준 미달)을 새 챔피언 위에서 재시험'),
})
IO_COLS = ['io_down', 'io_up', 'io_sell', 'io_max', 'io_same']
EXPS.update({
    'R3E15': dict(base=C2, delta=dict(drop_cols=DES_COLS, drop_ext=['design']), change='[절제] 챔피언(E11)에서 디자인권 5피처 제거 — 실용신안·물가가 들어온 뒤에도 디자인이 기여하는지(제거가 +0.003·60% 이상이면 채택)'),
    'R3E16': dict(base=C2, delta=dict(drop_cols=IO_COLS, drop_ext=['io']), change='[절제] 챔피언(E11)에서 산업연관표 5피처 제거 — R2 채택 외부데이터(IO)의 4년창·새 외부데이터 하 한계기여'),
})

# ======================= B단계(2026-09-29): 3라운드 새 수집 외부데이터로 추가 개선 — 출발 챔피언 R3E11 =======================
OA_COLS = ['oa_k_g', 'oa_k_cov', 'oa_k_lvl', 'oa_fit_g', 'oa_fit_w', 'oa_gap']
ROS_COLS = ['ros_k_sh', 'ros_k_pc', 'ros_fit', 'ros_f_us', 'ros_f_sci', 'ros_f_sh', 'ros_f_pc', 'ros_gap']
KI_COLS = ['ki_n', 'ki_dn', 'ki_hhi', 'ki_g3', 'ki_invs']
EXPS.update({
    'R3E17': dict(base=C2, delta=dict(add_ext=['oa'], add_cols=OA_COLS), change='[외부 OpenAlex 기술셀 r3oa] 목표 KSIC 메인그룹 구성 가중 과학 점유성장·커버·수준, 기업 IPC 메인그룹×목표 KSIC 메인그룹 겹침 위 과학성장·겹침량, 기업 기술 대비 격차 6피처(논문 연도 ≤ c−1)'),
    'R3E18': dict(base=C2, delta=dict(add_ext=['ros'], add_cols=ROS_COLS), change='[외부 Reliance on Science r3ros] 목표 KSIC 과학 의존도(최근 10년 발행 US 패밀리 논문인용 비율·특허당 논문), 기업 기술×목표 셀 과학 의존도, 기업 자사 US 패밀리 과학 기반 4종, 보유산업 대비 격차 8피처(US 발행 ≤ c 연말)'),
})
EXPS.update({
    'R3E19': dict(base=C2, delta=dict(nts_filter=True), change='[외부 국세청 휴·폐업 r3nts] 폐업일 ≤ 원점 연말 법인의 쌍을 진입 불가로 보고 점수 최저(채점 단계 위험집합 정제, 재학습 없음)',
                  note='2013년 이전 폐업은 좌측 절단 → 개발 원점에서 폐업 법인 0~2개. 보조 원점 c=2015에서만 실효(runs/R3E19/nts_stats.json)'),
})
IPCX_COLS = ['ipcx_fit', 'ipcx_fit_rank']
EXPS.update({
    'R3E20': dict(base=C2, delta=dict(add_ext=['ipcx'], add_cols=IPCX_COLS), change='[제공 KR 출원×전체 IPC] 기업–목표 KSIC 기술 적합도 확장: 원점 연말까지 공개·등록된 자사 KR 출원의 전체 IPC 서브클래스(분할 가중) 프로파일 코사인·순위 2피처(E37은 등록 특허 메인 IPC만)'),
})
EXPS.update({
    'R3E21': dict(base=C2, delta=dict(add_ext=['kitem'], add_cols=KI_COLS), change='[외부 KOSIS 품목편 r3kosis] 목표 KSIC 생산 품목 수·3년 변화·품목 HHI·연쇄 생산성장 3년·재고/출하 5피처(자료연도 ≤ c−2) — 제품 다양화·집중도가 신규 진입 여지를 반영한다는 가설'),
})
IPCXD_COLS = ['ipcx_rfit', 'ipcx_kdfit']
EXPS.update({
    'R3E22': dict(base='R3E20', delta=dict(add_ext=['ipcxd'], add_cols=IPCXD_COLS), change='[제공 KR 출원×전체 IPC] 기술 적합도의 방향 2피처: 기업 최근 3년 출원 프로파일×목표 KSIC 적합도, 목표 KSIC 최근 3년 프로파일 쪽으로의 적합도 변화(대 3년 이전)'),
})
OA2_COLS = [x + '2' for x in OA_COLS]
ROS2_COLS = [x + '2' for x in ROS_COLS]
EXPS.update({
    'R3E23': dict(base='R3E20', delta=dict(add_ext=['oa2'], add_cols=OA2_COLS), change='[외부 OpenAlex 기술셀 r3oa 재시험] E17의 6피처를 기업·KSIC 메인그룹 프로파일을 공개·등록 출원 전체 IPC(E20 방식)로 바꿔 다시 계산(기업 기술 겹침 결측 85%→62%)'),
    'R3E24': dict(base='R3E20', delta=dict(add_ext=['ros2'], add_cols=ROS2_COLS), change='[외부 Reliance on Science r3ros 재시험] E18의 8피처를 공개·등록 출원 전체 IPC 메인그룹 프로파일로 다시 계산(기업 기술×목표 셀 과학 의존도 결측 78%→55%)'),
})

# ---- B단계 재시도(2026-09-29 오후): 챔피언 R3E20 위에서 ----
C3 = 'R3E20'   # B단계 챔피언(공개·등록 출원 전체 IPC 기술 적합도)
OC_COLS = ['oc_k_g', 'oc_k_sh', 'oc_fit_g', 'oc_gap']
IPCM_COLS = ['ipcm_fit', 'ipcm_fit_rank']
ENT_COLS = ['ent_fit', 'ent_fit_rank']
EXPS.update({
    'R3E25': dict(base=C3, delta=dict(add_ext=['oaco'], add_cols=OC_COLS), change='[외부 OpenAlex 기업소속 논문 r3oa 재시도분] 목표 KSIC 메인그룹 구성(E20 방식 공개·등록 출원) 가중 기업논문 점유 성장·기업 비중, 기업×목표 메인그룹 겹침 위 기업논문 성장, 기업 기술 대비 격차 4피처(발행연도 ≤ c−1)'),
    'R3E26': dict(base=C3, delta=dict(add_ext=['ipcm'], add_cols=IPCM_COLS), change='[제공 KR 출원×전체 IPC] 기술 적합도(E20)를 메인그룹 해상도로: 공개·등록 출원 전체 IPC 메인그룹(분할 가중) 코사인·순위 2피처'),
    'R3E27': dict(base=C3, delta=dict(add_ext=['entfit'], add_cols=ENT_COLS), change='[제공 KR 출원×전체 IPC] 최근 진입자 닮음: 목표 KSIC에 (c−4, c] 첫 공개·등록 출원을 낸 법인들의 KSIC 밖 서브클래스 프로파일 평균과 기업 프로파일의 코사인·순위 2피처'),
})
PEER_COLS = ['peer_k', 'peer_k_rank']
EXPS.update({
    'R3E28': dict(base=C3, delta=dict(add_ext=['peer'], add_cols=PEER_COLS), change='[제공 KR 출원×전체 IPC] 기술 이웃 보유(협업 필터링): 서브클래스 프로파일 코사인 상위 50개 이웃 법인 중 목표 KSIC에 이미 공개·등록 출원이 있는 비율(유사도 가중)·순위 2피처'),
})
EXPS.update({
    'R3E29': dict(base=C3, delta=dict(add_ext=['ipcm', 'entfit', 'peer'], add_cols=IPCM_COLS + ENT_COLS + PEER_COLS), change='[제공 KR 출원×전체 IPC 기술 근접 묶음] 개별 경계(E26·E27·E28)인 메인그룹 적합도·최근 진입자 닮음·기술 이웃 보유 6피처를 함께 추가'),
})
C4 = 'R3E29'   # B단계 재시도 챔피언(기술 근접 묶음)
EXPS.update({
    'R3E30': dict(base=C4, delta=dict(drop_ext=['ipcm'], drop_cols=IPCM_COLS), change='[절제] 챔피언(E29)에서 메인그룹 적합도 2피처 제거 — 서브클래스 적합도(E20)와 겹치는지(제거가 +0.003·60% 이상이면 채택)'),
    'R3E31': dict(base=C4, delta=dict(drop_ext=['peer'], drop_cols=PEER_COLS), change='[절제] 챔피언(E29)에서 기술 이웃 보유 2피처 제거 — 묶음 안 한계기여(제거가 +0.003·60% 이상이면 채택)'),
})
PEERE_COLS = ['peer_e', 'peer_e_rank']
EXPS.update({
    'R3E32': dict(base=C4, delta=dict(add_ext=['peerent'], add_cols=PEERE_COLS), change='[제공 KR 출원×전체 IPC] 기술 이웃의 최근 진입: 서브클래스 코사인 상위 50개 이웃 법인 중 목표 KSIC에 (c−4, c] 첫 공개·등록 출원을 낸 비율(유사도 가중)·순위 2피처(E28 보유 비율의 동적 판)'),
})
