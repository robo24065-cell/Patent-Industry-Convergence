# -*- coding: utf-8 -*-
"""C09-T2 s1: 현재 원고 숫자 토큰 추출 (PREREG 1절 규칙). 출력: cache/tokens.csv"""
import re, csv, hashlib, sys, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'cache', 'manuscript_now_88d77045.md')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'cache', 'tokens.csv')
txt = open(SRC, encoding='utf-8').read()
lines = txt.split('\n')
if lines and lines[-1] == '':
    lines = lines[:-1]

FRAC = re.compile(r'(?<![A-Za-z0-9_.\-])\d[\d,]*/\d[\d,]*')
NUM = re.compile(r'(?<![A-Za-z0-9_.\-/])([+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][−\-+]?\d+)?)')
UNIT = re.compile(r'^(%p|%|배|건|개|셀|사|쌍|곳|명|행|쪽|과제|칸|년|월|자|회|단계|년차|차)')
YEAR_RANGE_TAIL = re.compile(r'(?:19|20)\d\d[–\-~](\d\d)(?!\d)')
LIT_EXCL = [r'예시 1·2', r'4\.4-1·2', r'4\.4-\d', r'코로나19', r'CC BY-NC 4\.0', r'제3자', r'4분면', r'\d\.\d\.\d']
DATE = re.compile(r'(?:19|20)\d\d-\d\d(?:-\d\d)?')


def scope_of(i, line):
    n = i + 1
    if 693 <= n:
        return 'M'
    if 445 <= n <= 692:
        return 'R'
    if 433 <= n <= 442:
        return 'X'  # 참고문헌
    if line.lstrip().startswith('|') or re.match(r'^\*\*표 \d+', line.strip()):
        return 'T'
    if n <= 19:
        return 'S'
    if 366 <= n <= 415:
        return 'K'
    return 'B'


def fingerprint(line):
    s = FRAC.sub('#', line)
    s = NUM.sub('#', s)
    s = re.sub(r'[\s\*\|#`>]+', '', s)
    return s

rows = []
for i, line in enumerate(lines):
    n = i + 1
    sc = scope_of(i, line)
    fp = fingerprint(line)
    excl_line = None
    st = line.strip()
    if re.match(r'^\[그림 \d+: .*\.png\]$', st):
        excl_line = '그림 자리 표시줄'
    spans = []
    # 날짜·연도범위 꼬리 위치 기록
    date_spans = [(m.start(), m.end()) for m in DATE.finditer(line)]
    yr_tail = [(m.start(1), m.end(1)) for m in YEAR_RANGE_TAIL.finditer(line)]
    taken = []
    for m in FRAC.finditer(line):
        # 날짜 안이면 건너뜀
        taken.append((m.start(), m.end(), m.group(0), 'frac'))
    for m in NUM.finditer(line):
        s, e = m.start(1), m.end(1)
        if any(a <= s < b for a, b, _, _ in taken):
            continue
        taken.append((s, e, m.group(1), 'num'))
    taken.sort()
    lit_spans = [(m.start(), m.end()) for pat in LIT_EXCL for m in re.finditer(pat, line)]
    for s, e, tok, kind in taken:
        before = line[max(0, s - 6):s]
        after = line[e:e + 6]
        um = UNIT.match(line[e:])
        unit = um.group(1) if um else ''
        excl = excl_line
        core = tok.lstrip('+-−')
        if excl is None:
            if sc == 'X':
                excl = '참고문헌 서지'
            elif any(a <= s < b for a, b in lit_spans):
                excl = '식별자(문구)'
            elif any(a <= s < b for a, b in date_spans):
                excl = '날짜'
            elif any(a <= s < b for a, b in yr_tail):
                excl = '연도 범위 꼬리'
            elif re.fullmatch(r'(19|20)\d\d', core) and unit in ('', '년') and kind == 'num':
                excl = '연도'
            elif s > 0 and re.match(r'[A-Za-z]', line[s - 1]):
                excl = '영문 식별자'
            elif s > 1 and line[s - 1] == '-' and re.match(r'[A-Za-z0-9]', line[s - 2]) and not tok.startswith('-'):
                excl = '영문 식별자'
            elif re.search(r'(표|그림|박스|예시|Q|부록|절|항목|단계) ?$', line[max(0, s - 3):s]) and unit == '':
                excl = '표·그림 번호'
            elif unit in ('단계', '년차', '차') :
                excl = '단계·차수 식별자'
            elif re.fullmatch(r'[1-6]\.[1-6](\.[1-6])?', core) and unit == '' and (re.match(r'^[\)\,·\s\-–과와의에이을를]', line[e:e+1] or ' ') or e == len(line)):
                excl = '절 번호'
            elif re.match(r'^#+ ', line) and re.fullmatch(r'\d+(\.\d+)*', core) and s < 12:
                excl = '절 번호'
            elif re.match(r'^\d+\. ', line) and s == 0:
                excl = '목록 순번'
            elif e < len(line) and re.match(r'[A-Za-z_]', line[e]) and not re.match(r'e[−\-+]?\d', line[e:]):
                excl = '영문 식별자'
            elif re.search(r'(\\|/|\.csv|\.json|\.md|\.py|\.png)', line[max(0, s - 2):e + 6]) and sc in ('R', 'M'):
                excl = '경로'
        ctx = line[max(0, s - 40):min(len(line), e + 30)].replace('\n', ' ')
        rng = ''
        if re.match(r'^\s*[–~]', line[e:e+2]) or re.search(r'[–~]\s*$', line[max(0, s-2):s]):
            rng = 'range'
        if re.search(r'\[\s*[+\-−]?[\d.,]*\s*,?\s*$', line[max(0, s-12):s]):
            rng = (rng + ' ci').strip()
        rows.append(dict(line=n, scope=sc, pos=s, token=tok, unit=unit, kind=kind, range=rng,
                         excluded=excl or '', context=ctx, fp=fp))

with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
from collections import Counter
print('lines', len(lines), 'tokens', len(rows))
print('excluded', Counter(r['excluded'] for r in rows).most_common())
print('scope(included)', Counter(r['scope'] for r in rows if not r['excluded']).most_common())
print('sha1', hashlib.sha1(open(SRC, 'rb').read()).hexdigest())
