# -*- coding: utf-8 -*-
# 반박 검증: 원본 facts + facts_patch_v20 을 csv 모듈로 독립 적용 -> facts_work 와 행·칸 단위 비교
import csv, io, sys, json, collections, hashlib, re
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석'
ORIG = B + r'\보고서_20261012\facts'
T = B + r'\검증루프_20261012\C19\C19-T3'
FILES = ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']
out = {}

def rd(p):
    b = open(p, 'rb').read()
    bom = b.startswith(b'\xef\xbb\xbf')
    s = b.decode('utf-8-sig')
    crlf = s.count('\r\n'); lf = s.count('\n') - crlf
    rows = list(csv.reader(io.StringIO(s, newline='')))
    return rows, dict(bom=bom, crlf=crlf, lf=lf, endnl=s.endswith('\n'), sha1=hashlib.sha1(b).hexdigest())

orig = {f: rd(ORIG + '\\' + f) for f in FILES}
work = {f: rd(T + r'\facts_work\\' + f) for f in FILES}
for f in FILES:
    out[f + '_fmt'] = dict(orig=orig[f][1], work=work[f][1],
                           orig_fieldcounts=dict(collections.Counter(len(r) for r in orig[f][0])),
                           work_fieldcounts=dict(collections.Counter(len(r) for r in work[f][0])),
                           orig_n=len(orig[f][0]), work_n=len(work[f][0]))
patch = list(csv.reader(open(T + r'\facts_patch_v20.csv', encoding='utf-8-sig', newline='')))
hdr, patch = patch[0], patch[1:]

# 적용
st = {f: [list(r) for r in orig[f][0]] for f in FILES}
problems = []
for i, p in enumerate(patch, 2):
    f, pid, col, cur, new, ev, src, kind = p
    rows = st[f]; H = rows[0]
    idx = [k for k, r in enumerate(rows) if r and r[0] == pid]
    if col in H:
        c = H.index(col)
        if len(idx) != 1:
            problems.append((i, pid, 'id count', len(idx))); continue
        r = rows[idx[0]]
        if r[c] != cur:
            problems.append((i, pid, col, 'cur mismatch', r[c][:80], cur[:80]))
        r[c] = new
    elif col == '(행 추가)':
        nr = next(csv.reader([new]))
        if len(nr) != 8: problems.append((i, pid, 'add fields', len(nr)))
        if idx: problems.append((i, pid, 'add but exists', len(idx)))
        rows.append(nr)
    elif col == '(행 대체)':
        a, b2 = pid.split('→')
        idx = [k for k, r in enumerate(rows) if r and r[0] == a]
        nr = next(csv.reader([new]))
        cr = next(csv.reader([cur]))
        if len(idx) != 1: problems.append((i, pid, 'repl idcount', len(idx))); continue
        if rows[idx[0]] != cr: problems.append((i, pid, 'repl cur mismatch', rows[idx[0]], cr))
        if len(nr) != 8: problems.append((i, pid, 'repl fields', len(nr)))
        rows[idx[0]] = nr
    elif col == '(형식)':
        r = rows[idx[0]]
        problems.append(('info', i, pid, 'format row orig fields', len(r), r))
    elif col == '-':
        pass
    else:
        problems.append((i, pid, 'unknown col', col))
out['problems'] = problems

# facts_work 와 비교 (행 순서 무시 / 유지 둘 다)
cmp = {}
for f in FILES:
    a = st[f]; w = work[f][0]
    ka = collections.Counter(tuple(r) for r in a); kw = collections.Counter(tuple(r) for r in w)
    onlyA = list((ka - kw).elements()); onlyW = list((kw - ka).elements())
    cmp[f] = dict(n_mine=len(a), n_work=len(w), same_order=(a == w), onlyMine=onlyA, onlyWork=onlyW)
out['cmp'] = cmp

# 패치 대상 밖 변경: 원본 vs work, 패치에 없는 (id,col) 변경
touched = set((p[0], p[1], p[2]) for p in patch)
added = set((p[0], p[1]) for p in patch if p[2] in ('(행 추가)',))
outside = []
for f in FILES:
    H = orig[f][0][0]
    om = {r[0]: r for r in orig[f][0][1:] if r}
    wm = {r[0]: r for r in work[f][0][1:] if r}
    for k, r in wm.items():
        if k not in om:
            if (f, k) not in added and not any(p[1].endswith('→' + k) for p in patch): outside.append((f, k, 'new id'))
            continue
        for c, h in enumerate(H):
            ov = om[k][c] if c < len(om[k]) else None; wv = r[c] if c < len(r) else None
            if ov != wv and (f, k, h) not in touched: outside.append((f, k, h, (ov or '')[:60], (wv or '')[:60]))
    for k in om:
        if k not in wm and not any(p[1].startswith(k + '→') for p in patch): outside.append((f, k, 'removed'))
    # 중복 id
    ids = [r[0] for r in work[f][0][1:] if r]
    d = [k for k, v in collections.Counter(ids).items() if v > 1]
    out[f + '_dupids_work'] = d
    out[f + '_dupids_orig'] = [k for k, v in collections.Counter(r[0] for r in orig[f][0][1:] if r).items() if v > 1]
out['outside'] = outside

# 행 추가 22행 수치·단위 점검: 수치 토큰이 조각 안에 있는가
num_chk = []
for p in patch:
    if p[2] in ('(행 추가)', '(행 대체)'):
        nr = next(csv.reader([p[4]]))
        val, unit, frag = nr[2], nr[3], nr[4]
        toks = re.split(r'[/–]', val.replace('−', '-'))
        fr = frag.replace('−', '-')
        miss = [t for t in toks if t.strip() and t.strip().lstrip('+') not in fr]
        num_chk.append((p[1], val, unit, miss, bool(val.strip()), bool(unit.strip())))
out['num_chk'] = num_chk
json.dump(out, open(T + r'\verify_0\r1_apply.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'cmp'}, ensure_ascii=False, indent=1)[:12000])
for f in FILES:
    c = cmp[f]; print(f, c['n_mine'], c['n_work'], c['same_order'], len(c['onlyMine']), len(c['onlyWork']))
    for r in c['onlyMine'][:5]: print('  MINE', [x[:100] for x in r])
    for r in c['onlyWork'][:5]: print('  WORK', [x[:100] for x in r])
