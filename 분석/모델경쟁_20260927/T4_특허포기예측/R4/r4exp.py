# -*- coding: utf-8 -*-
"""T4 4라운드 실험 정의(NTIS 정부R&D 기원, NTIS Open API 활용): 함수 1개 = 변경 1개. 입력 = 현재 챔피언 spec(깊은 복사)."""
ALL = ('cls', 'haz', 'logit', 'hgb')
XGB = ('cls', 'haz')


def _add(spec, cols, comps=ALL):
    for c in comps:
        n = spec['components'][c]['num']
        for x in cols:
            if x not in n:
                n.append(x)
    return spec


GV5 = ['gv_univ', 'gv_gri', 'gv_big', 'gv_sme', 'gv_oth']


def x_gov_grp(s):
    return _add(s, GV5), ('[NTIS] 수행주체별 정부R&D 기원 플래그 5개(대학·출연연·대기업/중견·중소·기타; 성과 레코드 Year ≤ L연도−1, '
                          '등록 2007년 전 특허는 NaN) — 4개 구성요소. ntis_TESTS T4-N1')


def x_gov_self(s):
    return _add(s, ['gv_self', 'gv_lnproj']), ('[NTIS] 기업 직접 수행 정부과제 특허 플래그 + log(1+연결 과제 수)(같은 시점 규칙·NaN 규칙) — '
                                               '동일 출원인 안에서도 −2.0%p였던 몫. 4개 구성요소. ntis_TESTS T4-N2')


def x_gov_grp_xgb(s):
    return _add(s, GV5, comps=XGB), '[NTIS] 수행주체별 플래그 5개를 XGB 2개(분류형·위험모형)에만 — 선형 구성요소의 NaN→중앙값 대치 부담 제거'


def x_gov_any_sk(s):
    return _add(s, ['gv_any', 'gv_self'], comps=('logit', 'hgb')), '[NTIS] 정부 기원·기업 수행 2개 플래그를 sklearn 2개(로지스틱·HGB)에만'
