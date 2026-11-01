# -*- coding: utf-8 -*-
"""s3_checks.py 3차 수정. 한 번만 실행."""
import os
d = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(d, 's3_checks.py')
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    if a not in s:
        raise SystemExit('NOT FOUND: ' + a[:90])
    s = s.replace(a, b, 1)
rep("""    vals = []
    for r in rows:
        if re.match(r'\\d{5}', r['항목'].strip()) and r['비율'].strip():
            vals.append(F(r['비율']))""", """    vals = [F(r['nps_share']) for r in CSV('(MC)/_shared/ext/firm_nps_link_by_ksic.csv')]""")
rep("""chk(125, ['≤1e−10'], [S(lambda: '≤1e−10' if t4n5max() <= 1e-10 else f'max {t4n5max():.2e}')], '(LOOP)/C04/C04-T2/t4_rescore.json·n5_rescore.json diff_vs_official 최대 절대값', srctype='루프 산출 원천',
    note=str(S(lambda: f'{t4n5max():.3e}')))""", """chk(125, ['≤1e−10'], [S(lambda: '≤1e−10' if t4n5max() <= 1e-10 else f'max {t4n5max():.2e}')], '(LOOP)/C04/C04-T2/t4_rescore.json·n5_rescore.json diff_vs_official 최대 절대값', srctype='루프 산출 원천',
    note='AUC 차 최대 7.9e−11(≤1e−10 성립)이나 T4 skill 차 2.0e−10. 같은 칸의 T1 9.0e−8은 skill 포함 전 지표 최대 차', verdict_override='부분',
    defn='E-C04T2-04 원안은 \\'AUC 차 ≤1e−10\\'; 표에서 \\'AUC\\'가 빠져 전 지표 최대 차로 읽힘(T4 skill 2.0e−10)')""")
rep("""chk(142, ['46', '48%', '0.713', '0.523'],""", """chk(142, ['46', '48%'], [S(lambda: md_num('(MC)/T3_기술셀부상예측/VERIFY_R5/robustness_table.md', r'피처 창 KR 문서의 약 ([0-9]+)%')), S(lambda: 48.2)],
    'T3 VERIFY_R5 robustness_table(\\'약 46%\\', R2 t0 2019·2020) / (LOOP)C03-T3 meta 재계산 46.9–48.2%(t0 2016·2019–21)', srctype='문서·루프 원천',
    defn='하한 46은 트랙 문서 \\'약 46%\\'(R2 원점), 재계산 범위는 46.9–48.2%(→47–48%); \\'약\\' 표기로 허용 가능하나 출처마다 원점 집합이 다름')
chk(142, ['0.713', '0.523'],""")
rep("""[S(lambda: 46.9), S(lambda: 48.2), S(lambda: F(ROW(SP + 't3a_forecast_summary.csv', 목표='5국', L='0', 구간='개발')['쌍순서_M'])),""", """[S(lambda: F(ROW(SP + 't3a_forecast_summary.csv', 목표='5국', L='0', 구간='개발')['쌍순서_M'])),""")
rep("""    note='46–48%는 46.9–48.2%를 바깥으로 감싼 범위 표기("약")')""", """    note='split34 t3a_forecast_summary 5국 개발 L=0/L=2')""")
open(p, 'w', encoding='utf-8').write(s)
print('patched c')
