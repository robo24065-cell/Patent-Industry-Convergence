# 반박 검증 r1: 독립 재계산(부록 파싱·토큰화 독자 구현), 패치 독자 적용, 축약/복사 점검
import csv, io, re, json, sys, hashlib
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; T = AN + r"\검증루프_20261012\C18\C18-T3"; V = T + r"\verify_2"
msb = open(R + r"\manuscript_v3.md", 'rb').read()
assert hashlib.sha1(msb).hexdigest().startswith('b316df8e')
ms = msb.decode('utf-8').replace('\r\n', '\n').split('\n')
a0 = [i for i, s in enumerate(ms) if s.startswith('# [부록')][0]
a1 = [i for i, s in enumerate(ms) if s.startswith('# [편집 메모')][0]
APP = {}
for i in range(a0, a1):
    s = ms[i].strip()
    if not s.startswith('|') or set(s) <= set('|- :'): continue
    cells = [c.strip() for c in s[1:-1].split('|')]
    if cells[0] in ('id', ''): continue
    # 값 칸에 '|'가 있을 수 있음: 마지막 칸=출처, 둘째=절, 나머지=값
    key = re.sub(r'\s*\(보충( R12)?\)$', '', cells[0])
    APP.setdefault(key, []).append(dict(line=i+1, val='|'.join(cells[2:-1])))
dup = {k: v for k, v in APP.items() if len(v) > 1}
def toks(s):
    s = s.replace('−', '-').replace('–', ' ').replace('—', ' ')
    out = set()
    for m in re.finditer(r'(?<![A-Za-z0-9_])\d[\d,]*(?:\.\d+)?', s):
        t = m.group(0).rstrip(',')
        if re.fullmatch(r'\d{1,3}(,\d{3})+(\.\d+)?', t): t = t.replace(',', '')
        out.add(t)
    return out
def rd(p):
    t = open(p, 'rb').read().decode('utf-8-sig')
    rows = list(csv.reader(io.StringIO(t)))
    h = rows[0]; res = {}
    order = []
    for r in rows[1:]:
        if not r: continue
        d = dict(zip(h, r)); res[r[0]] = d; order.append(r[0])
    return h, res, order, rows
NAMES = ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']
def metric(fdir, tokf):
    allrows = {}
    for n in NAMES:
        h, res, order, rows = rd(fdir + chr(92) + n)
        for k in order: allrows[k] = (n, res[k])
    nA = nB = nAB = n = 0; det = []
    for k, (n_, d) in allrows.items():
        if k not in APP: continue
        n += 1
        av = ' '.join(x['val'] for x in APP[k])
        ft = tokf(d.get('정확한 문장 조각', '')); at = tokf(av)
        A = ft - at; B = at - (ft | tokf(d.get('조건·경고', '')))
        nA += bool(A); nB += bool(B); nAB += bool(A or B)
        if A or B: det.append((k, sorted(A), sorted(B)))
    miss = [k for k in APP if k not in allrows]
    return dict(n=n, A=nA, B=nB, AorB=nAB, app_ids=len(APP), app_missing_in_facts=miss), det
out = {}
out['dup_app_ids'] = {k: [x['line'] for x in v] for k, v in dup.items()}
for lab, fd in [('src', R + r'\facts'), ('work', T + r'\facts_work')]:
    m, det = metric(fd, toks); out[lab] = m; out[lab + '_resid'] = det
# 패치 독자 적용
P = list(csv.DictReader(open(T + r'\facts_patch_v19.csv', encoding='utf-8-sig')))
mine = {}; probs = []
for n in NAMES:
    h, res, order, rows = rd(R + r'\facts' + chr(92) + n)
    mine[n] = dict(h=h, res={k: dict(v) for k, v in res.items()}, order=list(order))
for p in P:
    f = mine[p['파일']]; k = p['id']; kind = p['종류']; col = p['열']
    if kind in ('설정', '치환', '덧붙임'):
        if k not in f['res']: probs.append(('no id', k)); continue
        cur = f['res'][k].get(col)
        if cur is None: probs.append(('no col', k, col)); continue
        if kind == '치환':
            if p['현재값'] not in cur: probs.append(('치환 현재값 없음', k, col)); continue
            f['res'][k][col] = cur.replace(p['현재값'], p['새값'])
        else:
            if cur != p['현재값']: probs.append((kind + ' 현재값 불일치', k, col))
            f['res'][k][col] = p['새값']
    elif kind in ('행 추가', '행 대체'):
        vals = [x.strip() for x in p['새값'].split(' | ')]
        new = dict(zip(f['h'], vals)) if len(vals) <= len(f['h']) else None
        if new is None: probs.append(('fields>', k, len(vals))); continue
        nid = vals[0]
        if kind == '행 대체':
            old = k.split('→')[0]; idx = f['order'].index(old); f['order'][idx] = nid; del f['res'][old]
        else:
            f['order'].append(nid)
        f['res'][nid] = {c: new.get(c, '') for c in f['h']}
# 내 적용본 ↔ facts_work 필드 비교
cmp = {}
for n in NAMES:
    h, res, order, rows = rd(T + r'\facts_work' + chr(92) + n)
    m = mine[n]; diffs = []
    if order != m['order']: diffs.append(('order', len(order), len(m['order'])))
    for k in set(order) | set(m['order']):
        a = res.get(k); b = m['res'].get(k)
        if a is None or b is None: diffs.append(('missing', k)); continue
        for c in h:
            if (a.get(c) or '') != (b.get(c) or ''): diffs.append((k, c, (a.get(c) or '')[:60], (b.get(c) or '')[:60]))
    cmp[n] = diffs
out['apply_problems'] = probs; out['mine_vs_work'] = cmp
# 설정·치환으로 facts 조각이 부록 셀과 완전히 같아진 행(원고→facts 복사), '…' 포함 행
_, wk, _, _ = rd(T + r'\facts_work\facts.csv'); _, sr, _, _ = rd(R + r'\facts\facts.csv')
copy = []; ell = []
for p in P:
    if p['열'] == '정확한 문장 조각' and p['id'] in APP:
        if p['새값'].strip() == APP[p['id']][0]['val'].strip(): copy.append(p['id'])
    if '…' in p['새값'] and '…' not in p['현재값']: ell.append((p['id'], p['열'], p['종류']))
out['frag_set_equal_appendix'] = copy; out['ellipsis_introduced'] = ell
for n in NAMES:
    _, w, _, _ = rd(T + r'\facts_work' + chr(92) + n)
    out['ellipsis_in_work_frag_' + n] = [k for k, d in w.items() if '…' in d.get('정확한 문장 조각', '')]
    _, s, _, _ = rd(R + r'\facts' + chr(92) + n)
    out['ellipsis_in_src_frag_' + n] = [k for k, d in s.items() if '…' in d.get('정확한 문장 조각', '')]
# 부록 셀 '…' 축약 행 전체 및 남은 3행의 누락 토큰이 축약 뒤에 오는지
trunc = {k: v[0]['val'] for k, v in APP.items() if v[0]['val'].rstrip().endswith('…')}
out['app_truncated_ids'] = sorted(trunc)
chk = {}
for k in ['MAP-KR-TOP', 'DATA-REGISTRY', 'LINK-T1-R3']:
    d = wk[k]; frag = d['정확한 문장 조각']; pre = trunc[k].rstrip('…').rstrip()
    # 부록 셀(…앞)이 facts 조각의 접두인지(마지막 미완 토큰 허용)
    cut = pre[:-8] if len(pre) > 8 else pre
    chk[k] = dict(prefix=frag.startswith(cut), A_missing=sorted(toks(frag) - toks(trunc[k])), tail=frag[len(cut):][:200])
out['truncation_check'] = chk
json.dump(out, open(V + r'\r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in out if not k.endswith('_resid')}, ensure_ascii=False, indent=1)[:9000])
print('work_resid', out['work_resid']); print('src_resid count', len(out['src_resid']))
