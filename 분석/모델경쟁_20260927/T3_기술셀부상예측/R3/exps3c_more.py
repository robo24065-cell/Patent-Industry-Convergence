# exps3c.py 가 exec 로 읽는 추가 실험 정의 (B단계 재시도)
def add_comp(comp_, w):
    def f(cfg):
        c2 = copy.deepcopy(cfg); c2['components'].append(comp_)
        c2['weights'] = list(c2.get('weights', [1] * (len(c2['components']) - 1))) + [w]; return c2
    return f

MORE = {
 'E51': ('4번째 성분: KSIC 산업 nowcast 점수 ks_now 자체(학습 없음) 순위평균 가중 0.3',
         add_comp(dict(type='raw', features=['ks_now']), 0.3), 'KSIC 수준 산업 신호를 셀 순위에 기울기로만 반영'),
 'E52': ('KOSIS 품목편 공개연도 3피처 → 3성분', feats(KP), ''),
 'E53': ('4번째 성분: KSIC 산업 로지스틱(품목편 3 + 생산지수 2 + KSIC 특허 모멘텀 2, C=0.1) 순위평균 가중 0.3',
         add_comp(dict(type='logit', features=['kp_gprod2', 'kp_gship2', 'kp_invlvl', 'pi_prod_g2', 'pi_ship_g2', 'ksic_kr_mom', 'ksic_tot_mom'], C=0.1), 0.3),
         'E31(A단계, 물가·NBER·OpenAlex 포함 −0.0037)의 품목편 판'),
 'E54': ('ETSI 표준필수특허 선언 2피처(셀 누적 SEP 선언 레코드 수·최근 3년 신규 선언; 선언일 ≤t0, 레코드 출원 ≤t0-2) → 트리 2성분',
         feats(['sep_cum', 'sep_new3'], comps=T2), '표준화 경쟁 셀의 후속 출원. 품질 수집기 단독검정은 T3 기각(15 KSIC 표본)'),
 'E55': ('근소 미달 묶음: KSIC nowcast 층 ks_now → 3성분(E50, +0.0008 7/9) + ETSI SEP 2피처 → 트리(E54, +0.0002 7/9)',
         chain(feats(['ks_now']), feats(['sep_cum', 'sep_new3'], comps=T2)), '묶음 채택 시 선택 편향 명시'),
 'E56': ('KOSIS 품목편 연쇄 생산 2년 성장 1피처(E40 점수 통제 편상관 +0.048 9/9) → 음이항 성분에만', feats(['kp_gprod2'], comps=('nb',)), '산업 성장률을 셀 기대 건수의 승수로'),
}
