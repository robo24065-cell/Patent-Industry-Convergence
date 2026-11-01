# -*- coding: utf-8 -*-
"""T4 3라운드 실험 정의: 함수 1개 = 변경 1개. 입력 = 현재 챔피언 spec(깊은 복사), 출력 = (새 spec, 변경 설명[, 추가 메타])."""
XGB = ('cls', 'haz')
ALL = ('cls', 'haz', 'logit', 'hgb')


def _add(spec, cols, comps=ALL, log1p=()):
    for c in comps:
        n = spec['components'][c]['num']
        for x in cols:
            if x not in n: n.append(x)
        if c == 'logit':
            spec['components'][c]['log1p'] = list(dict.fromkeys(spec['components'][c]['log1p'] + list(log1p)))
    return spec


def _drop(spec, cols, comps=ALL):
    for c in comps:
        spec['components'][c]['num'] = [x for x in spec['components'][c]['num'] if x not in cols]
    return spec


SEP = ['sep_self', 'sep_fam', 'sep_a_share']
NP = ['np_pres', 'np_act', 'np_lemp', 'np_gemp', 'np_exit', 'np_lossr']
KP = ['kp_n', 'kp_lxi']
PR = ['pr_ppi_g1', 'pr_ppi_g3', 'pr_exp_g3']
AP = ['ap_late1y', 'ap_late_d', 'ap_multi1y', 'ap_n1y', 'ap_lapacc']


# ---------------------------------------------------------------- 외부데이터 (조합 B: 특허 생존 = 기업 생존 × 표준 가치)
def x_sep(s):
    return _add(s, SEP), '외부A ETSI 표준필수특허: 자체 선언·해외 패밀리 선언(공지일<=L)·출원인 SEP 비중 (4개 구성요소)'


def x_nps_haz(s):
    return _add(s, NP, comps=('haz',)), ('외부B 국민연금 사업장(출원인코드→사업자번호): 가입 여부·활성·log 가입자·1년 증감·탈퇴·상실률, 스냅샷 게시월<L. '
                                          '위험모형에만(분류형 학습 기준일은 모두 2015-11 이전이라 값이 없음)')


def x_kpss(s):
    return _add(s, KP), '외부C KPSS 특허가치: US 패밀리(등록일<=L) 수·최대 xi (4개 구성요소)'


def x_price_xgb(s):
    return _add(s, PR, comps=XGB), '외부D ECOS 생산자·수출물가 1·3년 상대변화(이듬해 1월 공표 반영) — XGB 2개에만'


def i_ap(s):
    return _add(s, AP), '내부 권리자 정리 신호: 출원인 최근1년 기한후납부 비중·변화, 다년 선납 비중, 납부건수, 포기 가속도 (4개 구성요소)'


OA = ['oa_all_rg', 'oa_kr_rg', 'oa_all_lsh']


def x_oa_xgb(s):
    return _add(s, OA, comps=XGB), '외부E OpenAlex 기업 소속 저자 논문(KSIC 키워드, 출판<=L연도-1): 전세계·한국 2년 모멘텀(전체 대비), 논문 점유 — 기술 퇴조 과학 신호, XGB 2개에만'


def x_sep_pat(s):
    return _add(s, ['sep_self', 'sep_fam']), ('외부A 변형: ETSI SEP 특허 단위만(자체·해외 패밀리 선언). X01에서 출원인 SEP 비중(sep_a_share)이 '
                                             '초기 원점 위험모형을 흔든 부분만 롤백')


def m_haz_bag6(s):
    s['components']['haz']['n_bag'] = 6
    return s, '모형: 위험모형 배깅 3 → 6 구성원(시드·열순서 3개 추가). 구성원 간 AUC 편차(약 0.004~0.005)를 줄이는 분산 감소', {'copy_bags': {'haz': 'PARENT'}}


def m_cls_bag6(s):
    s['components']['cls']['n_bag'] = 6
    return s, '모형: 분류형 배깅 3 → 6 구성원(분산 감소)', {'copy_bags': {'cls': 'PARENT'}}


# ---------------------------------------------------------------- (재시도) 앙상블 단계 변경 — 재학습 없음(이미 학습된 예측 재조합)
def _pool(s, comp, members):
    s['ensemble'].setdefault('pool', {})[comp] = list(members)
    return s


def e_rank(s):
    s['ensemble']['how'] = 'rank'
    return s, '앙상블: 확률 로짓 가중평균 → 순위 가중평균(같은 가중치). 구성요소 간 보정 차이(위험모형 결합 P 의 척도)를 없앰'


def e_logit_half(s):
    w = s['ensemble']['weights']; w['logit'] = 0.1; w['haz'] = round(w['haz'] + 0.1, 3)
    return s, '앙상블: 가장 약한 로지스틱(모든 원점에서 구성요소 AUC 최저) 가중 0.2→0.1, 줄인 만큼 위험모형 0.4→0.5'


def e_bag6_both(s):
    _pool(s, 'haz', ['X09']); _pool(s, 'cls', ['X10'])
    return s, '모형: 위험모형·분류형 배깅을 동시에 3→6(X09·X10 각각은 기준 미달, 함께 적용)'


HAZ_VAR = ['E00', 'X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07']
CLS_VAR = ['E00', 'X01', 'X03', 'X04', 'X05', 'X06', 'X07']
SK_VAR = ['E00', 'X01', 'X03', 'X05', 'X07']


def e_haz_committee(s):
    _pool(s, 'haz', HAZ_VAR)
    return s, ('외부데이터 위원회(위험모형): 위험모형 자리 = 챔피언 + 외부데이터별 위험모형 7개(ETSI·NPS·KPSS·물가·납부행동·OpenAlex·SEP특허) 로짓 평균. '
               '각 외부데이터는 단독으로 롤백됐지만 서로 다른 신호로 학습된 모형이라 오차가 덜 겹친다는 가설')


def e_cls_committee(s):
    _pool(s, 'cls', CLS_VAR)
    return s, '외부데이터 위원회(분류형): 분류형 자리 = 챔피언 + 외부데이터별 분류형 6개 로짓 평균'


def e_sk_committee(s):
    _pool(s, 'logit', SK_VAR); _pool(s, 'hgb', SK_VAR)
    return s, '외부데이터 위원회(로지스틱·HGB): 두 sklearn 자리 = 챔피언 + 외부데이터별 모형 4개 로짓 평균'


def e_combo(s):
    w = s['ensemble']['weights']; w['logit'] = 0.1; w['haz'] = round(w['haz'] + 0.1, 3)
    _pool(s, 'haz', ['X09']); _pool(s, 'cls', CLS_VAR)
    return s, ('결합: 기준 미달이지만 양(+)이었던 앙상블 변경 3개를 함께 — 로지스틱 0.1/위험 0.5(X12) + 위험모형 배깅6(X09) + 분류형 외부데이터 위원회(X15). '
               '각각 +0.0011/-0.0002/+0.0001')


RS = ['rs_sci', 'rs_lncit']
NT = ['nt_closed', 'nt_cl1y', 'nt_yrs']


def x_ros(s):
    return _add(s, RS, log1p=()), ('외부F Reliance on Science(Marx&Fuegi, USPTO 특허→논문 인용): KR 특허의 US 패밀리 등록특허(토큰 yr<=L연도-1, 발행일<L)가 '
                                   '논문을 인용하는지·최대 논문 인용 수(log) — 과학기반 특허는 기술 수명이 길어 덜 포기된다는 가설 (4개 구성요소)')


def x_nts_haz(s):
    return _add(s, NT, comps=('haz',)), ('외부G 국세청 휴·폐업(출원인코드→사업자번호, 폐업일<L): 폐업 여부·직전 1년 폐업·폐업 후 경과연수. 좌측 절단 때문에 L<2016 은 NaN. '
                                          '위험모형에만(분류형 학습 기준일은 모두 2016 이전)')


def t_acd_cap(s):
    s['train']['acd_cap'] = 0.03
    return s, ('학습 가중: 대형 출원인 가중 상한 — 기준일마다 한 출원인의 학습 가중 합이 3%를 넘지 않게(min(1, 0.03·N_L/n_a)). '
               '대형 권리자의 일괄 정리(예: 학습 구간 포기율 86%인 2위 출원인)가 특허 단위 패턴 학습을 지배하는 것을 줄인다는 가설 (4개 구성요소 재학습)')


AT = ['at_rate', 'at_n', 'at_rel']


def i_at(s):
    return _add(s, AT, log1p=()), ('내부 권리자 구간별 정리 정책: 출원인 × 다음 연차료 구간(1-3/4-6/7-9/10-12/13+) 최근 2년 불납률·결정 수·출원인 전체율 대비 차이 '
                                   '— "7년차에 일괄 포기" 같은 권리자 단위 정리 규칙을 직접 잡는다 (4개 구성요소)')


def t_acd_cap15(s):
    s['train']['acd_cap'] = 0.015
    return s, '학습 가중 변형: 출원인 가중 상한 3% → 1.5%(채택된 X21 방향으로 더 강하게)'


def t_acd_cap6(s):
    s['train']['acd_cap'] = 0.06
    return s, '학습 가중 변형: 출원인 가중 상한 3% → 6%(X21보다 약하게)'


def e_cap_pool(s):
    for c in ('cls', 'haz', 'logit', 'hgb'):
        _pool(s, c, ['X21', 'X23'])
    return s, '앙상블: 출원인 가중 상한 3%(X21)와 1.5%(X23) 모형을 구성요소마다 로짓 평균 — 상한 값 선택의 분산을 줄임(재학습 없음)'


# ================================================================ (B단계) 새 수집 데이터(r3nts·r3kosis·r3kip) + A단계 후속
def t_acd_cap1(s):
    s['train']['acd_cap'] = 0.01; s['train'].pop('acd_cap_pow', None)
    return s, '학습 가중 변형(B): 출원인 가중 상한 3% → 1%(A단계 권고 1회 시험 — 6%<3%<1.5% 순으로 좋아진 방향의 끝점)'


def t_acd_pow(s):
    s['train']['acd_cap'] = 0.005; s['train']['acd_cap_pow'] = 0.5
    return s, ('학습 가중 변형(B): 하드 상한 대신 제곱근 완화 상한 — w = min(1, sqrt(0.005·N_L/n_a)). 점유 0.5% 넘는 모든 출원인을 1/sqrt(n) 로 줄임 '
               '(A단계 권고 1/sqrt(n) 가중 1회 시험, 점유 3%면 0.41·10%면 0.22)')


DM = ['dm_opay_yrs', 'dm_oreg_yrs', 'dm_opay_n1y']


def i_dm(s):
    return _add(s, DM, log1p=('dm_opay_yrs', 'dm_oreg_yrs')), (
        '국세청 폐업 → 내부 대리변수(B): 출원인의 다른 특허 마지막 납부·등록 이후 경과연수, 최근 1년 다른 특허 납부 건수(모두 pd·rd<=L). '
        '국세청 폐업일은 L>=2016 만 관측되지만 폐업 법인은 모든 특허 납부를 멈춘다 — L>=2016 평가행에서 직전 1년 폐업을 AUC 0.73~0.92(4개 중 3개 기준일)로 잡음(probe/p7). '
        '이 대리변수는 모든 원점·학습 기준일에 값이 있다 (4개 구성요소)')


KZ = ['kz_gprod', 'kz_gprod3', 'kz_invship', 'kz_dinvship', 'kz_util_g1', 'kz_cap_g3', 'kz_impdom3']


def x_kz_xgb(s):
    return _add(s, KZ, comps=XGB), ('외부H KOSIS(B): KSIC 5자리 연간 품목편 연쇄 생산성장(1·3년)·재고/출하·그 변화(Y<=L연도-3/-2), '
                                    '3자리 가동률 1년·생산능력 3년 성장, 국내공급 수입−국산 3년 성장차(2013~). 성장률은 KSIC 단면 평균 차감 — '
                                    '수요 퇴조·수입 대체 업종의 특허 포기 가설, XGB 2개에만')


def t_acd_cap05(s):
    s['train']['acd_cap'] = 0.005; s['train'].pop('acd_cap_pow', None)
    return s, '학습 가중 변형(B): 출원인 가중 하드 상한 1% → 0.5%(6%<3%<1.5%<1% 로 단조 개선된 방향의 다음 점, 반전 확인용)'


def e_cap_pool_b(s):
    for c in ('cls', 'haz', 'logit', 'hgb'):
        _pool(s, c, ['X26', 'X32'])
    return s, '앙상블(B): 출원인 가중 상한 1%(X26)와 0.5%(X32) 모형을 구성요소마다 로짓 평균 — 상한 값 선택의 분산을 줄임(재학습 없음)'


def e_equal_w(s):
    for c in ('cls', 'haz', 'logit', 'hgb'):
        s['ensemble']['weights'][c] = 0.25
    return s, ('앙상블(B): 가중치 위험 0.4/나머지 0.2 → 4개 균등 0.25. 출원인 상한 뒤 로지스틱 AUC 가 크게 올라(O16 0.638→0.663) '
               '위험모형 편중 가중의 근거가 약해졌다는 가설(재학습 없음)')


def e_b_committee(s):
    _pool(s, 'haz', ['X26', 'X28', 'X29', 'X30']); _pool(s, 'cls', ['X26', 'X28', 'X29'])
    _pool(s, 'logit', ['X26', 'X28']); _pool(s, 'hgb', ['X26', 'X28'])
    return s, ('B단계 새 데이터 위원회: 구성요소 자리마다 챔피언 + 새 데이터 모형(국세청 폐업 대리변수 X28·KOSIS X29·국세청 폐업 X30) 로짓 평균. '
               '각각은 기준 미달이지만 원점별로 강한 곳이 달라(X28 O14·O18, X29 O16, X30 O18) 오차가 상쇄된다는 가설(재학습 없음)')
