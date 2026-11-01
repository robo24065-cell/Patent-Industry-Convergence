import csv, io, re, sys, json
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; T = AN + r"\검증루프_20261012\C16\C16-T4"
FILES = ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']
lines = open(R + r'\manuscript_v3.md', encoding='utf-8').read().splitlines()
st = [i for i, s in enumerate(lines) if s.startswith('# [부록')][0]
en = [i for i, s in enumerate(lines) if s.startswith('# [편집 메모')][0]
app = {}; dups = []; allrows = []
for i in range(st, en):
    s = lines[i].strip()
    if not s.startswith('|') or re.match(r'^\|\s*-', s): continue
    cells = [c.strip() for c in s[1:-1].split('|')] if s.endswith('|') else [c.strip() for c in s[1:].split('|')]
    if cells[0] == 'id': continue
    rid = re.sub(r'\s*\(보충( R12)?\)\s*$', '', cells[0])
    val = '|'.join(cells[2:-1])
    allrows.append((i + 1, rid, len(cells)))
    if rid in app: dups.append((rid, app[rid]['line'], i + 1))
    app[rid] = dict(line=i + 1, val=val)
def tk(s, signed):
    s = s.replace('−', '-').replace('–', ' ').replace('~', ' ')
    res = set()
    for m in re.finditer(r'(?<![\w.])([+-]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?', s):
        sign, ip, fp = m.group(1), m.group(2), m.group(3) or ''
        # a leading '-' right after a digit/letter range is a dash, not sign; treat '-' preceded by space or '(' or '[' as sign
        pre = s[max(0, m.start() - 1):m.start()]
        if sign == '-' and m.start() > 0 and re.match(r'[\w)]', s[m.start() - 1]): sign = ''
        if sign == '+': sign = ''
        t = ip.replace(',', '') + fp
        res.add((sign if signed else '') + t)
    return res
def load(p):
    rr = list(csv.reader(io.StringIO(open(p, encoding='utf-8-sig').read()))); h = rr[0]
    return [dict(zip(h, r[:len(h) - 1] + [','.join(r[len(h) - 1:])])) for r in rr[1:] if r]
res = {}
for lab, d in [('orig', R + r'\facts'), ('work', T + r'\facts_work')]:
    rows = []
    for f in FILES:
        for r in load(d + '\\' + f): r['_f'] = f; rows.append(r)
    out = dict(n_app_match=0, A_unsigned=[], B_unsigned=[], A_signed=[], B_signed=[], NUM_col_not_in_app=[])
    ids = set(r['id'] for r in rows)
    out['app_ids_without_facts'] = sorted(k for k in app if k not in ids)
    for r in rows:
        if r['id'] not in app: continue
        out['n_app_match'] += 1
        v = app[r['id']]['val']
        for sg in (False, True):
            ft = tk(r['정확한 문장 조각'], sg); at = tk(v, sg); ct = tk(r['조건·경고'], sg)
            a = sorted(ft - at); b = sorted(at - ft - ct)
            key = 'signed' if sg else 'unsigned'
            if a: out['A_' + key].append((r['id'], app[r['id']]['line'], ' '.join(a)))
            if b: out['B_' + key].append((r['id'], app[r['id']]['line'], ' '.join(b)))
        nt = tk(r['수치'], False) - tk(v, False)
        if nt: out['NUM_col_not_in_app'].append((r['id'], r['수치'], ' '.join(sorted(nt))))
    for k in list(out):
        if isinstance(out[k], list) and k != 'app_ids_without_facts': out['n_' + k] = len(out[k])
    res[lab] = out
res['app_rows'] = len(allrows); res['app_unique_ids'] = len(app); res['app_dup_ids'] = dups
res['cellcount_dist'] = {n: sum(1 for x in allrows if x[2] == n) for n in set(x[2] for x in allrows)}
json.dump(res, open(T + r'\verify_1\v1_b_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for lab in ('orig', 'work'):
    o = res[lab]; print(lab, {k: v for k, v in o.items() if k.startswith('n_')}, 'no facts:', o['app_ids_without_facts'])
print('app rows', res['app_rows'], 'unique', res['app_unique_ids'], 'dups', res['app_dup_ids'], res['cellcount_dist'])
w = res['work']
for k in ('A_unsigned', 'B_unsigned', 'A_signed', 'B_signed'):
    print('==', k); [print('  ', x) for x in w[k]]
print('== NUM col (work)'); [print('  ', x) for x in w['NUM_col_not_in_app']]
