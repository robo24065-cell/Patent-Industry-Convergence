# -*- coding: utf-8 -*-
"""C09-T2 s6: 장별 커버리지 요약표(cache/chapter_table.txt)"""
import csv, os
from collections import Counter, defaultdict
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = [x for x in csv.DictReader(open(os.path.join(BASE, 'coverage_map.csv'), encoding='utf-8-sig')) if x['cls'] != 'X']
def ch(l):
    l = int(l)
    for a, b, n in [(1, 19, '요약·표지(1–19)'), (20, 99, '1장(23–99)'), (100, 144, '2장(103–144)'), (145, 362, '3장(148–362)'), (363, 415, '4장(366–415)'), (416, 442, '출처·참고(419–441)'), (443, 692, '부록 출처표(445–691)'), (693, 9999, '편집 메모(693–804)')]:
        if a <= l <= b:
            return n
T = defaultdict(Counter)
for x in r:
    c = ch(x['line'])
    T[c]['n'] += 1; T[c][x['cls']] += 1
    st = x['recompute_status']
    if st == '재계산':
        T[c]['re_' + x['recompute_verdict']] += 1
    elif st == '구조·정의':
        T[c]['정의'] += 1
    elif st.startswith('A('):
        T[c]['A유지'] += 1
    elif st.startswith('미처리(간접'):
        T[c]['간접'] += 1
    elif st.startswith('미처리'):
        T[c]['미처리'] += 1
    elif st.startswith('대상 외'):
        T[c]['대상외'] += 1
order = ['요약·표지(1–19)', '1장(23–99)', '2장(103–144)', '3장(148–362)', '4장(366–415)', '출처·참고(419–441)', '부록 출처표(445–691)', '편집 메모(693–804)']
hdr = '장 | 토큰 | A | B | C | D | 재계산 정확 | 재계산 부분 | 재계산 오류 | 구조·정의 | A 그대로 | 부록 간접 | 미처리 | 대상 외'
lines = [hdr]
tot = Counter()
for o in order:
    t = T[o]; tot.update(t)
    lines.append(f"{o} | {t['n']} | {t['A']} | {t['B']} | {t['C']} | {t['D']} | {t['re_정확']} | {t['re_부분']} | {t['re_오류']} | {t['정의']} | {t['A유지']} | {t['간접']} | {t['미처리']} | {t['대상외']}")
lines.append(f"합계 | {tot['n']} | {tot['A']} | {tot['B']} | {tot['C']} | {tot['D']} | {tot['re_정확']} | {tot['re_부분']} | {tot['re_오류']} | {tot['정의']} | {tot['A유지']} | {tot['간접']} | {tot['미처리']} | {tot['대상외']}")
open(os.path.join(BASE, 'cache', 'chapter_table.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
