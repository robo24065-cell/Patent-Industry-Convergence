# -*- coding: utf-8 -*-
"""C19-T4 s1: 원 등급 추출 + 층화 표본 + 눈가림 목록. 항목별 상태는 cache\sample_key.csv에만 쓴다(화면 출력 없음)."""
import csv, re, os, json, random, collections, hashlib
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(os.path.dirname(D))
SEED = 20261019
ST = os.path.join(B, 'C18', 'C18-T3', 'errata_status_v18.csv')
L = open(os.path.join(B, 'ERRATA.txt'), encoding='utf-8').read().split('\n')
rows = list(csv.DictReader(open(ST, encoding='utf-8-sig')))

def bubble_of_line(i):
    for j in range(i, -1, -1):
        m = re.search(r'C(\d\d)', L[j]) if L[j].startswith('====') else None
        if m:
            return 'C' + m.group(1)
    return ''

def header_lines(r):
    out = []
    lns = [x for x in re.split(r'[,; ]+', r['ERRATA줄'] or '') if x.strip().isdigit()]
    for x in lns:
        i = int(x)
        if 0 < i <= len(L) and r['ID'] in L[i - 1]:
            out.append(L[i - 1])
    if not out:
        for t in L:
            if re.match(r'\s*' + re.escape(r['ID']) + r'\b', t):
                out.append(t); break
    if not out:
        for x in lns:
            i = int(x)
            if 0 < i <= len(L):
                out.append(L[i - 1])
    return out

def grade_of(h, idv):
    k = h.find(idv)
    tag = h[k + len(idv):k + len(idv) + 200] if k >= 0 else h[:200]
    br = ' '.join(re.findall(r'\[([^\]]*)\]', tag))
    src = br if br else tag[:60]
    s2 = re.sub(r'재빌드\s*필수|확인\s*필수|재빌드로[^\]]*?필수', '', src)
    g = set()
    if '필수' in s2:
        g.add('필수')
    if '권장' in src:
        g.add('권장')
    if '오류 아님' not in src and (re.search(r'확인된 오류|확인 오류|확인·오류', src) or (re.search(r'(^|[\s\[·(])오류', src) and '확인' in src)):
        g.add('확인된오류')
    return g, src

# 우선1
prio1 = set()
for i, t in enumerate(L):
    if t.strip().startswith('반영 우선순위'):
        seg = t.split(':', 1)[1] if ':' in t else ''
        if not seg.strip():
            # 다음 줄들에서 (1)
            for j in range(i + 1, min(i + 6, len(L))):
                if re.match(r'\s*\(1\)', L[j]):
                    seg = L[j]; break
        if '(1)' in seg:
            seg = seg.split('(1)', 1)[1].split('(2)', 1)[0]
        else:
            seg = seg.split('>', 1)[0]
        bub = bubble_of_line(i)
        ids = []
        last_prefix = None
        for m in re.finditer(r'(E-C\d\d[-A-Za-z0-9]*\d[a-z]?)|(?:(?<=[·/])\s*(\d\d[a-z]?)\b)|\b(T\d-\d\d)\b', seg):
            if m.group(1):
                ids.append(m.group(1)); last_prefix = m.group(1).rsplit('-', 1)[0]
            elif m.group(2) and last_prefix:
                ids.append(last_prefix + '-' + m.group(2))
            elif m.group(3) and bub == 'C03':
                ids.append('E-C03-' + m.group(3)); last_prefix = 'E-C03-' + m.group(3).rsplit('-', 1)[0]
        prio1.update(ids)

allid = {r['ID'] for r in rows}
info = {}
for r in rows:
    hs = header_lines(r)
    g = set(); srcs = []
    for h in hs:
        gg, s = grade_of(h, r['ID']); g |= gg; srcs.append(s[:100])
    if r['ID'] in prio1:
        g.add('우선1')
    info[r['ID']] = {'원등급': '+'.join(sorted(g)), 'G': bool(g), 'Gstar': bool(g & {'필수', '확인된오류'}), '꼬리표': ' / '.join(srcs)}

rng = random.Random(SEED)
by = collections.defaultdict(list)
for r in sorted(rows, key=lambda r: r['ID']):
    by[r['상태']].append(r)
memb = collections.defaultdict(set)
A = rng.sample(by['반영됨'], 25)
for r in A: memb[r['ID']].add('A')
for r in by['부분']: memb[r['ID']].add('B')
cg = [r for r in by['미반영'] if info[r['ID']]['G']]
cn = [r for r in by['미반영'] if not info[r['ID']]['G']]
Cr = rng.sample(cn, 35 - len(cg))
for r in cg: memb[r['ID']].add('C_G')
for r in Cr: memb[r['ID']].add('C_rand')
Dd = rng.sample(by['폐기·대체'], 20)
for r in Dd: memb[r['ID']].add('D')
for r in rows:
    if info[r['ID']]['Gstar']: memb[r['ID']].add('E')

sel = [r for r in sorted(rows, key=lambda r: r['ID']) if r['ID'] in memb]
key_rows = []
for r in sel:
    d = dict(r); d.update({'층': '+'.join(sorted(memb[r['ID']])), '원등급': info[r['ID']]['원등급'], '꼬리표': info[r['ID']]['꼬리표']})
    key_rows.append(d)
os.makedirs(os.path.join(D, 'cache'), exist_ok=True)
with open(os.path.join(D, 'cache', 'sample_key.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys())); w.writeheader(); w.writerows(key_rows)
blind = [{'ID': r['ID'], '바퀴': r['바퀴'], 'ERRATA줄': r['ERRATA줄'], '범위': r['범위'], '부록여부': r['부록여부'],
          '원등급': info[r['ID']]['원등급']} for r in sel]
rng2 = random.Random(SEED + 1)
rng2.shuffle(blind)
for i, b in enumerate(blind):
    b['판정순번'] = i + 1
with open(os.path.join(D, 'blind_list.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['판정순번', 'ID', '바퀴', 'ERRATA줄', '범위', '부록여부', '원등급']); w.writeheader(); w.writerows(blind)
# 전체 원등급표(상태 없음)
with open(os.path.join(D, 'cache', 'orig_grades_v2.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', '원등급', 'G', 'Gstar', '꼬리표'])
    for r in rows:
        x = info[r['ID']]; w.writerow([r['ID'], x['원등급'], int(x['G']), int(x['Gstar']), x['꼬리표']])
summ = {'seed': SEED, 'n_selected': len(sel), 'strata_sizes': {k: sum(1 for r in sel if k in memb[r['ID']]) for k in ['A', 'B', 'C_G', 'C_rand', 'D', 'E']},
        'pop_G_by_status': {s: sum(1 for r in by[s] if info[r['ID']]['G']) for s in by},
        'pop_Gstar_by_status': {s: sum(1 for r in by[s] if info[r['ID']]['Gstar']) for s in by},
        'pop_by_status': {s: len(by[s]) for s in by}, 'prio1_ids_n': len(prio1), 'prio1_not_in_table': sorted(prio1 - allid),
        'sample_key_sha1': hashlib.sha1(open(os.path.join(D, 'cache', 'sample_key.csv'), 'rb').read()).hexdigest()}
json.dump(summ, open(os.path.join(D, 'cache', 's1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
