# -*- coding: utf-8 -*-
"""T4 2라운드 실험 정의: 각 함수 = 변경 1개. 입력 = 현재 챔피언 spec(깊은 복사), 출력 = (새 spec, 변경 설명)."""
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


# ---------------------------------------------------------------- 앙상블(재학습 없음)
def ens_drop_logit(s):
    s['ensemble']['weights'] = {'cls': 1 / 3, 'haz': 1 / 3, 'logit': 0.0, 'hgb': 1 / 3}
    return s, '앙상블에서 가장 약한 로지스틱 제외(분류·위험·HGB 1/3씩 로짓평균)'


def ens_rank(s):
    s['ensemble']['how'] = 'rank'
    return s, '앙상블 결합을 로짓평균 → 순위평균으로'


def ens_haz_up(s):
    w = s['ensemble']['weights']
    s['ensemble']['weights'] = {k: (0.4 if k == 'haz' else 0.2) for k in w}
    return s, '앙상블 가중치: 위험모형 0.4, 나머지 0.2'


# ---------------------------------------------------------------- 학습 설정
def rec_strong(s):
    s['train']['weight_div'] = 1.0
    return s, '최근성 가중 강화(1+(연도-2005)/2 → 1+(연도-2005)/1)'


def win8(s):
    s['train']['window_years'] = 8
    return s, '학습 기준일 창을 최근 8년으로 제한(초기 2005~ 코호트 제외)'


# ---------------------------------------------------------------- 내부 피처
def trend_ksic(s):
    return _add(s, ['ksic_lapse_1y', 'ksic_lapse_trend', 'ipc_lapse_2y']), '산업(KSIC)·IPC 최근 포기율·추세 피처 추가(4개 구성요소)'


# ---------------------------------------------------------------- 외부 데이터(KSIC 연결) — PROTOCOL_R2 §4
TR = ['tr_imp_g1', 'tr_imp_rg3', 'tr_cn_share', 'tr_cn_dshare3']
KS = ['ks_ship_rg3', 'ks_estab_g3', 'ks_emp_g3', 'ks_va_ratio']
IO = ['io_dn_g', 'io_dn_g3', 'io_dn_rel']
TM = ['tm_mg_rg', 'tm_mg_dd', 'tm_mg_rgkr', 'tm_sub_rg', 'tm_ksic_rg', 'tm_ksic_rgkr']
KA = ['ka_ksic_rel', 'ka_sub_rel']


def ext_trade(s):
    return _add(s, TR), '외부①무역: KSIC↔HS6 한국 수입 증가율(1년·3년 상대)·중국 수입비중·3년 변화 (4개 구성요소)'


def ext_kosis(s):
    return _add(s, KS), '외부②KOSIS 광업제조업조사 5자리: 출하 3년 상대증가·사업체·종사자 3년 증가·부가가치율 (L연도-2까지)'


def ext_io(s):
    return _add(s, IO), '외부③산업연관표 수요구조 × 광공업생산지수: 수요산업 생산 증감(1년·3년)·자기산업 대비'


def meta_tm(s):
    return _add(s, TM), '제공 5개국 KSIC별 특허 리스트 → 기술셀(IPC 메인그룹·서브클래스·KSIC) 출원 모멘텀·정점 대비 낙폭(공개 완결 연도까지)'


def ka_rel(s):
    return _add(s, KA), 'KSIC·IPC서브클래스 × 연차대 최근 3년 불납률(같은 연차대 전체 대비 상대위험)'


BZ = ['bz_tm_act3', 'bz_tm_share3', 'bz_pat_gr', 'bz_ext_share']


def biz_ratio(s):
    return _add(s, BZ), '권리목록(법인 전체 포트폴리오, L연도-1까지 등록분) 비율 피처: 최근3년 상표·디자인 활동·비중, 특허 등록 증가율, 제공범위 밖 특허 비중'


# ---------------------------------------------------------------- 모형·하이퍼파라미터
def hgb_big(s):
    s['components']['hgb']['params'].update(max_iter=800, learning_rate=0.04)
    return s, 'HGB 500회·lr0.05 → 800회·lr0.04'


def logit_c01(s):
    s['components']['logit']['C'] = 0.1
    return s, '로지스틱 규제 강화 C 0.5 → 0.1'


def ens_haz_05(s):
    s['ensemble']['weights'] = {'cls': 0.2, 'haz': 0.5, 'logit': 0.1, 'hgb': 0.2}
    return s, '앙상블 가중치 변형: 위험모형 0.5, 분류 0.2, HGB 0.2, 로지스틱 0.1'


def haz_semi(s):
    s['train']['haz_semi'] = True
    return s, '위험모형 학습 기준일을 연 1회 → 반기(1월·7월)로: 더 최근(C-1.5년) 결정까지 포함, 상한 100만행 추출 유지'


def ext_trade_xgb(s):
    return _add(s, TR, comps=XGB), '외부①무역 변형: 로지스틱·HGB 제외, XGB 분류·위험모형에만 무역 피처(로지스틱이 산업단위 피처를 외삽해 O12에서 0.634→0.602로 붕괴한 부분만 롤백)'


def ext_io_xgb(s):
    return _add(s, IO, comps=XGB), '외부③산업연관 변형: XGB 분류·위험모형에만'


def ens_nologit_haz(s):
    s['ensemble']['weights'] = {'cls': 0.3, 'haz': 0.4, 'logit': 0.0, 'hgb': 0.3}
    return s, '앙상블: 위험 0.4 유지, 로지스틱 제외(분류 0.3, HGB 0.3)'


def rec_weak(s):
    s['train']['weight_div'] = 4.0
    return s, '최근성 가중 약화(1+(연도-2005)/2 → 1+(연도-2005)/4) — 강화(E07)가 나빴으므로 반대 방향'


def hgb_leaf500(s):
    s['components']['hgb']['params']['min_samples_leaf'] = 500
    return s, 'HGB min_samples_leaf 200 → 500(시간 이동에 강하게 규제)'


def cls_r400(s):
    s['components']['cls']['rounds'] = 400
    return s, 'XGB 분류형 600 → 400라운드(오래된 코호트 과적합 완화)'


MAC = ['mac_prod_yoy', 'mac_ship_yoy', 'mac_inv_yoy', 'mac_prod_rel', 'mac_prod_3y']
FAM = ['fam_n', 'fam_cc', 'fam_us', 'fam_ep', 'fam_jp', 'fam_cn', 'fam_wo']


def drop_mac(s):
    return _drop(s, MAC), '제거 실험: 1라운드 외부 경기지표(KOSIS 광공업생산·출하·재고지수 5개) 빼기 — 전진검증에서도 기여하는지 확인'


def drop_fam(s):
    return _drop(s, FAM), '제거 실험: 제공 5개국 패밀리 피처(7개) 빼기'
