# -*- coding: utf-8 -*-
"""C09-T2 s2: 커버리지 지도 (PREREG 2절). 입력: cache/tokens.csv, 원고 판본 사슬, 이전 감사 csv.
출력: cache/coverage_raw.csv, cache/line_since.csv"""
import csv, re, os, json, difflib
from decimal import Decimal, ROUND_HALF_UP
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.abspath(os.path.join(BASE, '..', '..', '..'))  # 분석
REP = os.path.join(AN, '보고서_20261012')
LOOP = os.path.join(AN, '검증루프_20261012')
VERS = [('v2r', os.path.join(REP, 'manuscript_v2_from_response.md')),
        ('wip', os.path.join(REP, 'manuscript_v3_wip.md')),
        ('bC03', os.path.join(REP, 'drafts', 'manuscript_v3_before_C03.md')),
        ('v5', os.path.join(REP, 'drafts', 'manuscript_for_v5.md')),
        ('v6', os.path.join(REP, 'drafts', 'manuscript_for_v6.md')),
        ('v7', os.path.join(REP, 'drafts', 'manuscript_for_v7.md')),
        ('v8', os.path.join(REP, 'drafts', 'manuscript_for_v8.md')),
        ('v9', os.path.join(REP, 'drafts', 'manuscript_for_v9.md')),
        ('v10', os.path.join(BASE, 'cache', 'manuscript_now_88d77045.md'))]
VI = {k: i for i, (k, _) in enumerate(VERS)}

def rl(p):
    L = open(p, encoding='utf-8').read().split('\n')
    return L[:-1] if L and L[-1] == '' else L

def norm(s):
    return re.sub(r'[\s\*`]+', '', s)

NUMRE = re.compile(r'[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][−\-+]?\d+)?')
def fpr(s):
    s = re.sub(r'\d[\d,]*/\d[\d,]*', '#', s)
    s = NUMRE.sub('#', s)
    return re.sub(r'[\s\*\|#`>]+', '', s)

texts = {k: rl(p) for k, p in VERS}
nsets = {k: {} for k in texts}
for k, L in texts.items():
    for i, l in enumerate(L):
        nsets[k].setdefault(norm(l), []).append(i + 1)
cur = texts['v10']
# 줄별 '같아진 최초 판'
since = []
for i, l in enumerate(cur):
    n = norm(l)
    k0 = len(VERS) - 1
    for j in range(len(VERS) - 1, -1, -1):
        if n in nsets[VERS[j][0]]:
            k0 = j
        else:
            break
    since.append(k0)

fp_cur = [fpr(l) for l in cur]

_ops = {}
def _opmap(ver):
    if ver in _ops:
        return _ops[ver]
    a = [norm(l) for l in texts[ver]]; b = [norm(l) for l in cur]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    mp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for i in range(i1, i2):
            if tag == 'equal':
                mp[i + 1] = [j1 + (i - i1) + 1]
            elif tag == 'replace':
                mp[i + 1] = list(range(j1 + 1, j2 + 1))
            else:
                mp[i + 1] = []
    _ops[ver] = mp
    return mp

def map_to_current(ver, vline_text, vline=None):
    """판본 ver의 줄을 현재 줄로. 반환 (현재줄번호, 'same'|'changed', ratio)"""
    n = norm(vline_text)
    if not n:
        return None
    if n in nsets['v10']:
        ln = nsets['v10'][n][0]
        return ln, ('same' if since[ln - 1] <= VI[ver] else 'changed'), 1.0
    f = fpr(vline_text)
    cands = _opmap(ver).get(vline, []) if vline else []
    best, br = None, 0.0
    for j in cands:
        g = fp_cur[j - 1]
        r = difflib.SequenceMatcher(None, f, g, autojunk=False).ratio() if g else 0
        if r > br:
            best, br = j, r
    if best and br >= 0.3:
        return best, 'changed', br
    best, br = None, 0.0
    for i, g in enumerate(fp_cur):
        if not g or abs(len(g) - len(f)) > max(len(g), len(f)) * 0.6:
            continue
        sm = difflib.SequenceMatcher(None, f, g, autojunk=False)
        if sm.quick_ratio() < max(br, 0.5):
            continue
        r = sm.ratio()
        if r > br:
            best, br = i + 1, r
    if best and br >= 0.55:
        return best, 'changed', br
    return None

def find_frag_line(ver, frag, value=None, hint=None):
    L = texts[ver]
    if hint and 0 < hint <= len(L) and value:
        v = re.sub(r'[+\-−]', '', value.split('–')[0].split('~')[0]).strip()
        if v and v in L[hint - 1].replace('−', '-').replace('+', ''):
            nf0 = norm(frag)
            if nf0[:4] in norm(L[hint - 1]):
                return hint
    nf = norm(frag)
    pieces = [p for p in re.split(r'…|\.\.\.|·{2,}', nf) if len(p) >= 6]
    pieces.sort(key=len, reverse=True)
    for p in pieces[:2]:
        hits = [i + 1 for i, l in enumerate(L) if p in norm(l)]
        if hits:
            return hits[0]
    # 퍼지: 조각 지문과 가장 비슷한 줄
    f = fpr(frag)
    if len(f) < 6:
        return None
    best, br = None, 0
    for i, l in enumerate(L):
        g = fpr(l)
        if not g:
            continue
        # 부분 포함 비율
        sm = difflib.SequenceMatcher(None, f, g, autojunk=False)
        m = sum(b.size for b in sm.get_matching_blocks())
        r = m / max(1, len(f))
        if r > br:
            best, br = i + 1, r
    if br >= 0.8:
        return best
    if value:
        v = value.strip()
        hits = [i + 1 for i, l in enumerate(L) if v and v in l]
        if len(hits) >= 1:
            fb = None; bb = -1
            for h in hits:
                r = difflib.SequenceMatcher(None, f, fpr(L[h - 1]), autojunk=False).ratio()
                if r > bb:
                    fb, bb = h, r
            return fb
    return None

def nums(s):
    out = []
    for m in NUMRE.finditer(s or ''):
        t = m.group(0).replace(',', '').replace('−', '-').replace('+', '')
        try:
            out.append(float(t.replace('e-', 'e-')))
        except ValueError:
            pass
    return out

def tokval(tok):
    t = tok.rstrip(',').replace(',', '').replace('−', '-').replace('+', '')
    try:
        return float(t)
    except ValueError:
        return None

def decs(tok):
    t = tok.rstrip(',')
    if 'e' in t.lower():
        return None
    return len(t.split('.')[1]) if '.' in t else 0

def nmatch(x, tok):
    y = tokval(tok)
    if y is None:
        return False
    d = decs(tok)
    for c in (x, x * 100):
        c = abs(c); yy = abs(y)
        if d is None:
            if abs(c - yy) <= abs(yy) * 1e-6 + 1e-30:
                return True
            continue
        q = Decimal(1).scaleb(-d)
        if Decimal(repr(c)).quantize(q, rounding=ROUND_HALF_UP) == Decimal(repr(yy)).quantize(q):
            return True
    return False

B_PAT = re.compile(r'facts|ERRATA|FINDINGS|현재 원고|문구|\(문장\)|manuscript|원고 \d|v\d+ \d+행|^diff|^-$|그대로$')
recs = []
def add(audit, ver, vline, values_str, verdict, evid, path, extra=''):
    if vline is None:
        recs.append(dict(audit=audit, ver=ver, vline=None, cur=None, status='unmapped', values=values_str,
                         verdict=verdict, evid=evid, path=path, cls='', extra=extra)); return
    vt = texts[ver][vline - 1] if 0 < vline <= len(texts[ver]) else ''
    m = map_to_current(ver, vt, vline)
    ok_verdict = bool(re.search(r'일치|정확', verdict)) and not re.search(r'불일치|부분|오류', verdict)
    srcB = bool(B_PAT.search(path or '')) or bool(B_PAT.search(evid or '')) or not (path or '').strip()
    if not ok_verdict:
        cls = 'B'
    else:
        cls = 'B' if srcB else 'A'
    recs.append(dict(audit=audit, ver=ver, vline=vline, cur=(m[0] if m else None), status=(m[1] if m else 'gone'),
                     values=values_str, verdict=verdict, evid=evid, path=path, cls=cls, extra=extra))

# C01
p = os.path.join(LOOP, 'C01', 'C01-T2', 'audit_numbers.csv')
last_path = ''
for r in csv.DictReader(open(p, encoding='utf-8-sig')):
    path = r['원천 파일'] or ''
    if path.strip() in ('같은 파일', '') and last_path:
        path2 = last_path
    else:
        path2 = path; last_path = path
    ln = find_frag_line('v2r', r['원고 문장 조각'], r['원고 값'])
    has_src_val = bool((r['원천 값'] or '').strip() or (r['재계산 값'] or '').strip())
    ev = ('' if has_src_val else '원천값없음 ') + (r['메모'] or '')
    add('C01-T2', 'v2r', ln, r['원고 값'], r['판정'], ev if not has_src_val else (r['원천 값'] + ' ' + r['재계산 값']), path2,
        r['원고 문장 조각'][:60])
# C02
p = os.path.join(LOOP, 'C02', 'C02-T3', 'audit_numbers_C02.csv')
last_path = ''
for r in csv.DictReader(open(p, encoding='utf-8-sig')):
    path = r['원천_경로'] or ''
    if path.strip() in ('같은 파일', '') and last_path:
        path2 = last_path
    else:
        path2 = path; last_path = path
    try:
        hint = int(re.match(r'\d+', r['줄(v3)'] or '').group(0))
    except Exception:
        hint = None
    ln = find_frag_line('v2r', r['원고_조각'], r['원고_값'], hint)
    ver = 'v2r'
    if ln is None:
        ln = find_frag_line('wip', r['원고_조각'], r['원고_값'], hint); ver = 'wip'
    sv = (r['원천_값'] or '').strip()
    has = bool(sv) and sv not in ('(문장)',)
    add('C02-T3', ver, ln, r['원고_값'], r['판정'], sv if has else '원천값없음 ' + (r['메모'] or ''), path2,
        r['원고_조각'][:60])
# C02 x2 출처표 '쓰인 절' 대조(원천 값 대조 아님 → B)
p = os.path.join(LOOP, 'C02', 'C02-T3', 'x2_section_check.csv')
seen = set()
for r in csv.DictReader(open(p, encoding='utf-8-sig')):
    try:
        ln = int(r['출처표_줄'])
    except Exception:
        continue
    if ln in seen:
        continue
    seen.add(ln)
    add('C02-T3x2', 'v2r', ln, r['숫자토큰'], '문구대조', r['숫자토큰'], '출처표 쓰인 절 문구 대조(x2_section_check)')
# C03
p = os.path.join(LOOP, 'C03', 'C03-T3', 'v2_v3_diff.csv')
for r in csv.DictReader(open(p, encoding='utf-8-sig')):
    m = re.match(r'(\d+)-(\d+)', r['v3_lines'] or '')
    if not m:
        continue
    a, b = int(m.group(1)), int(m.group(2))
    for ln in range(a, b + 1):
        ev = r['evidence']
        path = 'C03-T3 evidence' if not re.search(r'ERRATA 문구', ev) else 'ERRATA 문구'
        add('C03-T3', 'bC03', ln, ev, r['verdict'].replace('반영 ', ''), ev, path)
# C07
p = os.path.join(LOOP, 'C07', 'C07-T2', 'audit_changed_lines.csv')
for r in csv.DictReader(open(p, encoding='utf-8-sig')):
    m = re.match(r'(\d+)', r['현재행(manuscript_v3 cfa99556)'] or '')
    if not m or '부록' in r['현재행(manuscript_v3 cfa99556)'] and False:
        continue
    ln = int(m.group(1))
    add('C07-T2', 'v8', ln, r['원천 재계산·근거 값'], r['판정'], r['원천 재계산·근거 값'], r['근거 파일'])
# C08
for fn in ('audit_v4_v7.csv', 'audit_v8_v9.csv'):
    p = os.path.join(LOOP, 'C08', 'C08-T2', fn)
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        cl = r['current_line(88abe4e7)'] or ''
        lns = []
        for part in re.split(r'[;,]', cl):
            m = re.match(r'(\d+)(?:-(\d+))?', part.strip())
            if m:
                a = int(m.group(1)); b = int(m.group(2) or a)
                lns += list(range(a, b + 1))
        for ln in lns:
            add('C08-T2', 'v9', ln, r['evidence_numbers'], r['verdict'], r['evidence_numbers'], r['evidence_path'] + ' ' + r['note'][:40], r['block'])

json.dump([{k: v for k, v in x.items()} for x in recs], open(os.path.join(BASE, 'cache', 'audit_records.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)

# 토큰 분류
toks = list(csv.DictReader(open(os.path.join(BASE, 'cache', 'tokens.csv'), encoding='utf-8-sig')))
by_cur = {}
for r in recs:
    if r['cur']:
        by_cur.setdefault(r['cur'], []).append(r)
out = []
for t in toks:
    ln = int(t['line'])
    cls, match = '', ''
    if not t['excluded']:
        cand = by_cur.get(ln, [])
        hits = []
        for r in cand:
            vals = nums(r['values'])
            if t['kind'] == 'frac':
                if t['token'].rstrip(',') in (r['values'] or ''):
                    hits.append(r)
                continue
            if any(nmatch(x, t['token']) for x in vals):
                hits.append(r)
        same = [h for h in hits if h['status'] == 'same']
        chg = [h for h in hits if h['status'] == 'changed']
        if same:
            cls = 'A' if any(h['cls'] == 'A' for h in same) else 'B'
            use = [h for h in same if h['cls'] == cls]
        elif chg:
            cls = 'C'; use = chg
        else:
            cls = 'D'; use = []
        match = ' ; '.join(sorted({f"{h['audit']}@{h['ver']}:{h['vline']}({h['verdict'][:10]})" for h in use}))[:300]
        evid = ' ; '.join(sorted({(h['path'] or '')[:90] for h in use}))[:300]
    else:
        evid = ''
    out.append(dict(line=ln, scope=t['scope'], token=t['token'].rstrip(','), unit=t['unit'], range=t['range'],
                    excluded=t['excluded'], line_since=VERS[since[ln - 1]][0], cls=cls if not t['excluded'] else 'X',
                    matched=match, evidence=evid, context=t['context']))
with open(os.path.join(BASE, 'cache', 'coverage_raw.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
from collections import Counter
print('records', len(recs), Counter(r['status'] for r in recs))
print(Counter((r['audit'], r['status']) for r in recs))
inc = [o for o in out if o['cls'] != 'X']
print('included', len(inc))
for sc in 'STKBRM':
    c = Counter(o['cls'] for o in inc if o['scope'] == sc)
    print(sc, dict(c))
print('since dist', Counter(VERS[s][0] for s in since))
