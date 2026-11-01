# C20-T3 s3: s2에서 '중간표에서만'·'못 찾음'·'다른 원천에서만'으로 남은 토큰을 가진 행마다,
# 분석 폴더 전체 텍스트 파일(중간표·원고·facts·원장 제외)에서 행 토큰을 가장 많이 담은 파일을 찾는다(반올림 구간 이분 탐색).
# 사용: python s3_index.py <part> <nparts>   출력: cache/s3_part{part}.json
import os, sys, csv, json, re, bisect, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib20 import *
from decimal import Decimal

part, nparts = int(sys.argv[1]), int(sys.argv[2])
tri = list(csv.DictReader(open(C + r'\s2_triage.csv', encoding='utf-8-sig')))
need = {}
for t in tri:
    if t['분류'] in ('구조(연도)', '출처칸에서 찾음'):
        continue
    need.setdefault(t['파일'] + '|' + t['id'], set()).add(t['토큰'].lstrip('-'))
EXCL = ('facts_sync', 'facts_patch', 'ERRATA', 'FINDINGS', 'LEDGER', 'manuscript', 'circular_evidence', 'mismatch', 'proposed_',
        r'\facts\facts', r'보고서_20261012\facts', r'\drafts', r'C20\C20-T3', r'C19\C19-T3\facts_work', 'diff_numbers', 'line_conflicts', 'source_map',
        'PREREG', 'USER_REQUESTS')
files = []
for base in (VL, MC, REP):
    for dp, dn, fn in os.walk(base):
        for f in fn:
            if f.lower().endswith(('.json', '.csv', '.txt', '.md', '.tsv', '.log')):
                p = os.path.join(dp, f)
                if any(e in p for e in EXCL):
                    continue
                try:
                    if os.path.getsize(p) < 20_000_000:
                        files.append(p)
                except OSError:
                    pass
files.sort()
files = files[part::nparts]
intervals = {}
for k, ts in need.items():
    for t in ts:
        T = Decimal(t); d = ndp(t); h = Decimal(5).scaleb(-d - 1)
        intervals[t] = [(float(T - h), float(T + h)), (float((T - h) / 100), float((T + h) / 100))]
NUMRE = re.compile(r'\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
res = {}
t0 = time.time()
for p in files:
    try:
        s = open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    s2 = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)
    vals = sorted({float(m) for m in NUMRE.findall(s2) if len(m) < 40})
    if not vals:
        continue
    hitc = {}
    for t, ivs in intervals.items():
        for lo, hi in ivs:
            j = bisect.bisect_left(vals, lo - 1e-12)
            if j < len(vals) and vals[j] <= hi + 1e-12:
                hitc[t] = 1; break
    for k, ts in need.items():
        got = [t for t in ts if t in hitc]
        if got:
            res.setdefault(k, []).append((len(got), os.path.relpath(p, AN), sorted(got)))
    if time.time() - t0 > 480:
        print('time limit', p); break
for k in res:
    res[k] = sorted(res[k], key=lambda x: -x[0])[:15]
json.dump(dict(need={k: sorted(v) for k, v in need.items()}, res=res, nfiles=len(files)), open(C + rf'\s3_part{part}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('done', len(files), round(time.time() - t0))
