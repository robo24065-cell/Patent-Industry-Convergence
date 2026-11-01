# C19-T3 s1: 원고(v20) 숫자 토큰 추출 + 부록 출처표 파싱 → cache/s1_tokens.csv, cache/s1_app.json
import os, sys, re, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
ms = manu_lines()
st = next(i for i, s in enumerate(ms) if s.startswith('# [부록'))
en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
# 절 번호(본문 줄 → 절)
sec = {}; cur = '표지'
for i in range(st):
    s = ms[i]
    m = re.match(r'^#+\s+(\d+(?:\.\d+)*)', s)
    if s.startswith('## 요약'): cur = '표지·요약'
    elif m: cur = m.group(1)
    elif s.startswith('## 데이터 출처'): cur = '데이터 출처'
    elif s.startswith('## 참고문헌'): cur = '참고문헌'
    sec[i + 1] = cur
APP = {}
for i in range(st, en):
    s = ms[i]
    if s.startswith('| ') and not s.startswith('| ---'):
        c = [x.strip() for x in s.strip().strip('|').split(' | ')]
        if len(c) >= 4 and c[0] != 'id':
            key = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0])
            APP[key] = dict(line=i + 1, sec=c[1], val=' | '.join(c[2:-1]), src=c[-1])
NUM = re.compile(r'(?<![A-Za-z0-9_.])[−\-+]?\d[\d,]*(?:\.\d+)?%?')
def norm(t):
    t = t.replace('−', '-').replace('+', '').rstrip('%').rstrip(',')
    if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', t): t = t.replace(',', '')
    return t.rstrip(',').lstrip('-')
def struct_kind(s, m, line_no):
    t = m.group(0); a = max(0, m.start() - 6); pre = s[a:m.start()]; post = s[m.end():m.end() + 4]
    if s.startswith('#'): return '제목'
    if re.search(r'(표|그림|부록)\s*$', pre): return '표·그림 번호'
    if re.fullmatch(r'(19|20)\d\d', t.rstrip('%')) and not t.endswith('%'): return '연도'
    if re.match(r'^\s*\d+\.\s', s) and m.start() < 4: return '목록 번호'
    if re.fullmatch(r'\d\.\d(\.\d)?', t) and (re.match(r'^[\s(]*$', post[:1] or ' ') or post.startswith(')') or post.startswith('에') or post.startswith('·') or post.startswith('절')): return '절 번호'
    if sec.get(line_no) == '참고문헌': return '서지'
    return ''
rows = []
for i in range(en):
    s = ms[i]; ln = i + 1
    if not s.strip() or s.startswith('| ---') or s.startswith('[그림'): continue
    part = '부록' if i >= st else '본문'
    app_id = ''
    if part == '부록' and s.startswith('| '):
        c = [x.strip() for x in s.strip().strip('|').split(' | ')]
        app_id = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0]) if len(c) >= 4 else ''
        scan = ' | '.join(c[2:-1]) if app_id in APP else s
    else:
        scan = s
    for m in NUM.finditer(scan.replace('–', ' – ').replace('~', ' ~ ')):
        k = struct_kind(scan.replace('–', ' – ').replace('~', ' ~ '), m, ln) if part == '본문' else ''
        rows.append(dict(line=ln, part=part, sec=sec.get(ln, '부록'), app_id=app_id, tok=m.group(0), ntok=norm(m.group(0)), struct=k))
with open(C + r'\s1_tokens.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
jdump(dict(st=st + 1, en=en + 1, APP=APP), 's1_app.json')
from collections import Counter
out('tokens', len(rows), Counter(r['part'] for r in rows), Counter(r['struct'] for r in rows))
out('app rows', len(APP))
srcs = Counter()
for k, v in APP.items():
    for p in re.split(r';\s*', v['src']):
        srcs[p.split('(')[0].strip() if not p.startswith('(') else p.strip()] += 1
out('unique src entries', len(srcs))
