# -*- coding: utf-8 -*-
"""s3_checks.py 4차 수정: 감사 메모에서 옮겨 적은 상수 대신 루프 산출 파일을 직접 읽는다. 한 번만 실행."""
import os
d = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(d, 's3_checks.py')
s = open(p, encoding='utf-8').read()
def rep(a, b):
    global s
    if a not in s:
        raise SystemExit('NOT FOUND: ' + a[:100])
    s = s.replace(a, b, 1)
helpers = r'''
def _rs(col):
    r = [x for x in CSV('(LOOP)/C05/C05-T2/rescore_strata.csv') if x['track'] == 'N7' and x['stratum'].startswith('S2 as-of 확인')][0]
    if col == 'share':
        return F(r['n']) / F([x for x in CSV('(LOOP)/C05/C05-T2/rescore_strata.csv') if x['track'] == 'N7' and x['stratum'].startswith('공식')][0]['n']) * 100
    if col == 'skill':
        return F(r['skill'])
    lo, hi = [F(v) for v in r['skill_ci'].strip('[]').split(',')]
    return lo if col == 'lo' else hi
def _sec(which):
    j = J('(LOOP)/C07/C07-T2/cache/s3_A_n7_secret.json')
    if which == 'test':
        a = j['share_by_year']['2023']; b = j['share_by_year']['2024']
        return (a[0] + b[0]) / (a[1] + b[1]) * 100
    return j['share_dev_window_2018_22'] * 100
def _lc():
    for r in CSV('(LOOP)/C01/C01-T3/label_composition.csv'):
        try:
            n = F(r['n'])
        except Exception:
            continue
        if 9 < n < 10.5:
            return n / 108 * 100
    raise KeyError('권리자 청구 추정 행')
def _inv(metric):
    r = [x for x in CSV('(LOOP)/C01/C01-T3/rescoring_sensitivity.csv') if x['label'] == 'L_inv' and x['metric'] == metric]
    if not r:
        raise KeyError(str(sorted({x['label'] for x in CSV('(LOOP)/C01/C01-T3/rescoring_sensitivity.csv')})))
    return F(r[0]['point']) * 100
def _n6spec(i):
    v = [x[1] for x in J('(LOOP)/C07/C07-T2/cache/s2_n6_check.json', 'spec_skill').values()]
    return min(v) if i == 0 else max(v)
'''
rep("# ---------------- S 요약 ----------------", helpers + "\n# ---------------- S 요약 ----------------")
rep("""chk(275, ['70%', '0.18', '0.10', '0.26'], [S(lambda: 8647 / 12286 * 100), S(lambda: 0.1836), S(lambda: 0.103), S(lambda: 0.258)], rs + ' S2 as-of 확인(8,647/12,286, 0.1836 [0.103, 0.258]) — C08-T2 B42 재계산값', srctype='루프 산출 원천')""",
    """chk(275, ['70%', '0.18', '0.10', '0.26'], [S(lambda: _rs('share')), S(lambda: _rs('skill')), S(lambda: _rs('lo')), S(lambda: _rs('hi'))], rs + ' N7 S2 as-of 확인 행(n/공식 n, skill, skill_ci)', srctype='루프 산출 원천')""")
rep("""chk(275, ['1.6%', '1.1%'], [S(lambda: 1.591), S(lambda: 1.055)], '(LOOP)C07-T2 cache s3_A_n7_secret.json(1.591%/1.055%)', srctype='루프 산출 원천')""",
    """chk(275, ['1.6%', '1.1%'], [S(lambda: _sec('test')), S(lambda: _sec('dev'))], '(LOOP)C07-T2 cache s3_A_n7_secret.json share_by_year 2023+2024 / share_dev_window_2018_22', srctype='루프 산출 원천')""")
rep("""chk(286, ['9%', '39.3%', '33.3%'], [S(lambda: 9.63 / 108 * 100), S(lambda: 0.3929 * 100), S(lambda: 0.3333 * 100)], '(LOOP)C01-T3 label_composition(9.63/108); rescoring L_inv cap 0.3929·SB 0.3333', srctype='루프 산출 원천')""",
    """chk(286, ['9%', '39.3%', '33.3%'], [S(_lc), S(lambda: _inv('cap_champ')), S(lambda: _inv('cap_SB'))], '(LOOP)C01-T3 label_composition.csv(권리자 청구 추정 n/108); rescoring_sensitivity.csv L_inv cap_champ·cap_SB', srctype='루프 산출 원천')""")
rep("""chk(287, ['39건', '30건'], [S(lambda: J(N8J, 'champ_top1pct_hits')), S(lambda: 30.0)], N8J + '; (LOOP)C03-T3 s04 hits 39·prior 30', srctype='원천·루프')""",
    """chk(287, ['39건', '30건'], [S(lambda: J(N8J, 'champ_top1pct_hits')), S(lambda: float(J('(LOOP)/C01/C01-T3/s04_toplist.json', 'p_champ', 'prior')))], N8J + '; (LOOP)C01-T3 s04_toplist.json p_champ.prior', srctype='원천·루프')""")
rep("""chk(253, ['+6.4%', '+16.5%', '+12.6%'], [S(lambda: case('29280', 'KR출원_연성장%(08-09→16-17)')), S(lambda: 16.535), S(lambda: 12.640)],""",
    """chk(253, ['+6.4%', '+16.5%', '+12.6%'], [S(lambda: case('29280', 'KR출원_연성장%(08-09→16-17)')), S(lambda: J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json', '29280_출하_0809_1617')), S(lambda: J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json', '29280_출하_0809_1516'))],""")
rep("""chk(267, ['0.156', '0.159'], [S(lambda: 0.149496 + 0.00639), S(lambda: 0.149496 + 0.00992)], '(LOOP)C07-T2 cache s2_n6_check.json(공식 0.149496 + Δ 0.00639–0.00992)', srctype='루프 산출 원천')""",
    """chk(267, ['0.156', '0.159'], [S(lambda: _n6spec(0)), S(lambda: _n6spec(1))], '(LOOP)C07-T2 cache s2_n6_check.json spec_skill(명세 7개) 최소·최대', srctype='루프 산출 원천')""")
rep("""chk(324, ['14,872사', '11.0%', '5.5%', '+6.2%p', '+1.5', '+11.0'], [S(lambda: 14872.0), S(lambda: 1633 / 14872 * 100), S(lambda: 811 / 14872 * 100),""",
    """_U = lambda: sum(J('(LOOP)/C06/C06-T1/out/s2_meta.json', 'groups').values())
chk(324, ['14,872사', '11.0%', '5.5%', '+6.2%p', '+1.5', '+11.0'], [S(lambda: float(_U())), S(lambda: r12('NPS5', '기업') / _U() * 100), S(lambda: r12('DART5', '기업') / _U() * 100),""")
rep("""chk(73, ['48.8%'], [S(lambda: 4242947 / 8700777 * 100)], 'C03-T3 meta 직접 집계(실용신안 4,242,947/8,700,777)', srctype='루프 산출 원천')""",
    """chk(73, ['48.8%'], [S(lambda: J('(LOOP)/C03/C03-T3/_s10_cn_doctype.json', 'utility') / J('(LOOP)/C03/C03-T3/_s10_cn_doctype.json', 'cn_rows') * 100)], '(LOOP)C03-T3 _s10_cn_doctype.json utility/cn_rows', srctype='루프 산출 원천')""")
rep("""chk(223, ['5%'], [S(lambda: 5.4)], '(LOOP)C03-T3 t3_cn_ipc_recheck.json by_origin.2024: 5,328 대 비례배분 5,632 → 5.4% 적음', srctype='루프 산출 원천')""",
    """chk(223, ['5%'], [S(lambda: J('(LOOP)/C03/C03-T3/t3_cn_ipc_recheck.json', 'by_origin', '2024', 'pct_fewer_vs_imputed'))], '(LOOP)C03-T3 t3_cn_ipc_recheck.json by_origin.2024.pct_fewer_vs_imputed', srctype='루프 산출 원천')""")
rep("""[S(lambda: md_num('(MC)/T3_기술셀부상예측/VERIFY_R5/robustness_table.md', r'피처 창 KR 문서의 약 ([0-9]+)%')), S(lambda: 48.2)],""",
    """[S(lambda: md_num('(MC)/T3_기술셀부상예측/VERIFY_R5/robustness_table.md', r'피처 창 KR 문서의 약 ([0-9]+)%')), S(lambda: max(v['unseen_min_pub_rgst'] for v in J('(LOOP)/C03/C03-T3/_s8_unseen.json').values()) * 100)],""")
open(p, 'w', encoding='utf-8').write(s)
print('patched d')
