# -*- coding: utf-8 -*-
# C19-T4 s2: 판정 보조 자료. 눈가림 목록(blind_list.csv)만 읽는다(sample_key.csv 읽지 않음).
# 항목별 ERRATA 원문 블록, 따옴표 문자열의 v19/v20 검색 결과, ERRATA 반영 기록 중 ID가 나오는 줄을 cache\aid_NN.txt로 쓴다.
import csv, re, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(os.path.dirname(D))
R = os.path.join(os.path.dirname(B), '보고서_20261012')
L = open(os.path.join(B, 'ERRATA.txt'), encoding='utf-8').read().split('\n')
V19 = open(os.path.join(R, 'drafts', 'manuscript_for_v19.md'), encoding='utf-8').read().split('\n')
V20 = open(os.path.join(R, 'manuscript_v3.md'), encoding='utf-8').read().split('\n')

def norm(s):
    s = s.replace('**', '').replace('*', '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', s)
N19 = [norm(x) for x in V19]; N20 = [norm(x) for x in V20]
hdr = re.compile(r'^\s*(E-C\d\d[-A-Za-z0-9]*)\s*(\[|=|→|\+)')
rec_lines = []
inrec = False
for i, t in enumerate(L):
    if t.startswith('---- [메인 세션 반영 기록'):
        inrec = True
    elif t.startswith('====') or hdr.match(t):
        inrec = False
    if inrec:
        rec_lines.append(i)

def block(start):
    out = [L[start - 1]]
    for j in range(start, min(start + 14, len(L))):
        t = L[j]
        if hdr.match(t) or t.startswith('====') or t.startswith('----') or t.startswith('반영 우선순위'):
            break
        out.append(t)
    return out

def find(s, NN):
    s2 = norm(s).strip()
    if len(s2) < 4:
        return []
    return [i + 1 for i, x in enumerate(NN) if s2 in x][:6]

blind = list(csv.DictReader(open(os.path.join(D, 'blind_list.csv'), encoding='utf-8-sig')))
lo, hi = int(sys.argv[1]), int(sys.argv[2])
o = open(os.path.join(D, 'cache', f'aid_{lo:03d}_{hi:03d}.txt'), 'w', encoding='utf-8')
for b in blind:
    k = int(b['판정순번'])
    if not (lo <= k <= hi):
        continue
    idv = b['ID']
    o.write(f"\n######## [{k}] {idv} | {b['바퀴']} | 범위={b['범위']} | 원등급={b['원등급']} | ERRATA줄={b['ERRATA줄']}\n")
    lns = [int(x) for x in re.split(r'[,; ]+', b['ERRATA줄'] or '') if x.strip().isdigit()]
    if not lns:
        lns = [i + 1 for i, t in enumerate(L) if idv in t][:3]
    seen = set()
    for ln in lns:
        for t in block(ln):
            if t in seen: continue
            seen.add(t)
            o.write('  E| ' + t[:900] + '\n')
    qs = []
    for ln in lns:
        for t in block(ln):
            qs += re.findall(r"'([^']{4,160})'", t)
    done = set()
    for q in qs:
        if q in done: continue
        done.add(q)
        h19 = find(q, N19); h20 = find(q, N20)
        o.write(f"  Q| '{q[:90]}' v19={h19} v20={h20}\n")
    for i in rec_lines:
        if idv in L[i] and i + 1 not in lns:
            m = L[i].find(idv)
            o.write(f"  R| ERRATA {i+1}: …{L[i][max(0,m-60):m+260]}\n")
o.close()
