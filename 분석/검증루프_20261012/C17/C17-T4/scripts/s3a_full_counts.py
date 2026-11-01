# C17-T4 s3a: 전체 USPTO 파일 재집계(개발 구역만). python -I. 인자: <zip> <tag>
# (1) 숫자 특허 등록연도별 고유 수·번호 최소·최대  (2) 단계별 EXP./납부 이벤트 수(이벤트연도) 전체 대 제공연결
import sys, io, json, zipfile, collections
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C17\C17-T4\scripts')
from lib17 import *
import pyarrow.parquet as pq
zp, tag = sys.argv[1], sys.argv[2]
APPL = set(pq.read_table(C + r'\us_docs.parquet', columns=['appl8']).column('appl8').to_pylist())
z = zipfile.ZipFile(zp)
name = [n for n in z.namelist() if n.startswith('MaintFeeEvents_') and n.endswith('.txt')][0]
f = io.TextIOWrapper(z.open(name), encoding='ascii', errors='replace', newline='')
seen = {}
pref = collections.Counter()
ev = collections.Counter()   # (grp, eventyear, s, kind)
zc = {}
for line in f:
    pat = line[0:13]; appl = line[14:22]
    G = int(line[34:42]); e = int(line[43:51]); code = line[52:57].strip()
    if e >= 20240101:
        k = (G, e)
        lz = zc.get(k)
        if lz is None:
            lz = lock_zone(G, e); zc[k] = lz
        if lz:
            continue
    if not pat.isdigit():
        pref[pat.lstrip('0')[:2]] += 1
        continue
    n = int(pat)
    if n not in seen:
        seen[n] = G // 10000
    grp = 'link' if appl in APPL else 'other'
    if code == 'EXP.':
        dy = (e // 10000) - (G // 10000)
        s = 1 if dy <= 5 else (2 if dy <= 9 else 3)
        ev[(grp, e // 10000, s, 'EXP')] += 1
    elif code in PAY:
        ev[(grp, e // 10000, PAY[code][0], 'PAY')] += 1
by = collections.defaultdict(list)
for n, gy in seen.items():
    by[gy].append(n)
res = {'tag': tag, 'nonnumeric_prefix_lines': dict(pref.most_common(20)), 'by_grant_year': {}}
for gy in sorted(by):
    v = by[gy]
    res['by_grant_year'][str(gy)] = {'n': len(v), 'min': min(v), 'max': max(v), 'span': max(v) - min(v) + 1}
res['events'] = {f'{a}|{b}|{c}|{d}': n for (a, b, c, d), n in sorted(ev.items())}
json.dump(res, open(C + rf'\s3a_full_{tag}.json', 'w'), indent=1)
out({k: res['by_grant_year'][k] for k in res['by_grant_year'] if '2009' <= k <= '2024'})
