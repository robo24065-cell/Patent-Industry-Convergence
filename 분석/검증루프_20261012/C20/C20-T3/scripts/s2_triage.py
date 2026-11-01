# C20-T3 s2: s1 플래그(C3v·C4·C3x) 행의 '원천에서 못 찾은 토큰'을 확장 경로(출처 칸 + 패치 근거 경로 + C19 circular_evidence 원천)에서 다시 찾는다.
# 출력: cache/s2_triage.csv (행·토큰·어디서 찾았는가)
import os, sys, csv, io, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib20 import *

rows = list(csv.DictReader(open(C + r'\s1_scan_rows.csv', encoding='utf-8-sig')))
S = json.load(open(C + r'\s1_state_S.json', encoding='utf-8'))
rec = {}
for f in FILES:
    H = S[f][0]
    for r in S[f][1:]:
        rec[(f, r[0])] = dict(zip(H, r))
patch = read_rows(PATCH20)[1:]
ev = {}
for p in patch:
    ev.setdefault((p[0], p[1].split('→')[-1]), []).append(p[5])
circ = read_rows(C19T3 + r'\facts_work\circular_evidence.csv')[1:]
cpaths = {}
for r in circ:
    for s in (r[3] if len(r) > 3 else '').split(';'):
        s = s.strip()
        if s and not s.startswith('<'):
            cpaths.setdefault(r[0], set()).add(os.path.join(AN, s))

INTER = ('facts_sync.csv', 'facts_patch.csv', 'facts_patch_v19.csv', 'facts_patch_v20.csv', 'circular_evidence.csv')
STRUCT_YEAR = re.compile(r'^(19|20)\d\d$')
out = []
for r in rows:
    fl = r['플래그'].split()
    if not ({'C3v', 'C4', 'C3x'} & set(fl)):
        continue
    f, i = r['파일'], r['id']
    d = rec[(f, i)]
    p_src = [p for _, p in resolve_paths(d['출처 파일 경로']) if p]
    p_ev = []
    for e in ev.get((f, i), []):
        for m in re.finditer(r'((?:검증루프_20261012|모델경쟁_20260927|보고서_20261012)\\[^\s;,\]\[(]+)', e):
            p_ev.append(os.path.join(AN, m.group(1)))
    p_c = sorted(cpaths.get(i, []))
    toks_need = []
    for t in r['C3_수치원천없음'].split():
        toks_need.append(('수치', t))
    for t in r['C3_조각원천없음'].split():
        toks_need.append(('조각', t))
    if 'C3x' in fl:
        for t in toks(d['수치']):
            toks_need.append(('수치(C3x)', t))
    seen = set()
    for kind, t in toks_need:
        if (kind, t) in seen:
            continue
        seen.add((kind, t))
        res = {}
        for lab, plist in (('출처칸', p_src), ('패치근거', p_ev), ('circ원천', p_c)):
            hit = []
            for p in dict.fromkeys(plist):
                if not os.path.exists(p):
                    continue
                n = src_numbers(p)
                if n[0] != '읽음':
                    continue
                h = src_has(t, n)
                if h:
                    hit.append(os.path.relpath(p, AN) + ':' + h)
            res[lab] = hit
        raw = [x for x in res['패치근거'] + res['circ원천'] if not any(k in x for k in INTER)]
        inter = [x for x in res['패치근거'] + res['circ원천'] if any(k in x for k in INTER)]
        if STRUCT_YEAR.match(t.lstrip('-')):
            cls = '구조(연도)'
        elif res['출처칸']:
            cls = '출처칸에서 찾음'
        elif raw:
            cls = '다른 원천에서만(출처 칸 낡음 후보)'
        elif inter:
            cls = '중간표에서만'
        else:
            cls = '못 찾음'
        out.append(dict(파일=f, id=i, 종류=kind, 토큰=t, 분류=cls, 출처칸=' ; '.join(res['출처칸'])[:300], 원천=' ; '.join(raw)[:400], 중간표=' ; '.join(inter)[:200],
                        출처칸경로=' ; '.join(os.path.relpath(p, AN) if p.startswith(AN) else p for p in p_src)))
with open(C + r'\s2_triage.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
import collections
print(collections.Counter(o['분류'] for o in out))
for o in out:
    if o['분류'] not in ('출처칸에서 찾음', '구조(연도)'):
        print(o['id'], o['종류'], o['토큰'], o['분류'], '|', o['원천'][:160], '|', o['중간표'][:100])
