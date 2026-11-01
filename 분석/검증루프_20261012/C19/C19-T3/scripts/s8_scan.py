# C19-T3 s8: P0→V1E에서 바뀐 모든 수치(s6_compare_*.json)를 원고(1–662행)의 표기와 대조해, 원고에 나오는 숫자 중 표기가 바뀌는 후보를 전수로 뽑는다.
# 표기 후보: 원값을 소수 0–4자리로 반올림한 문자열, ×100(%) 0–2자리, 천 단위 쉼표 정수. 같은 자리수에서 V1E 표기와 다르면 '바뀜 후보'.
# 사용: python s8_scan.py map|lo
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
grp = sys.argv[1]
ms = manu_lines()
en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
TXT = [(i + 1, s.replace('−', '-')) for i, s in enumerate(ms[:en])]
D = json.load(open(C + rf'\s6_compare_{grp}.json', encoding='utf-8'))
def f(x):
    try:
        v = float(str(x).replace(',', ''))
        return v if v == v else None
    except Exception:
        return None
def reps(v):
    out = []
    for d in range(0, 5):
        out.append((f'{v:.{d}f}', d, 1.0))
    for d in range(0, 3):
        out.append((f'{v * 100:.{d}f}', d, 100.0))
    if abs(v) >= 1000 and float(v).is_integer():
        out.append((f'{int(v):,}', 0, 1.0))
    return out
hits = []
for r in D['v1e']:
    for k, a, b in r['diffs']:
        # 구간 문자열 [x, y]도 분해
        pa = re.findall(r'-?\d+\.?\d*', str(a)) if str(a).startswith('[') else [a]
        pb = re.findall(r'-?\d+\.?\d*', str(b)) if str(b).startswith('[') else [b]
        for x, y in zip(pa, pb):
            va, vb = f(x), f(y)
            if va is None or vb is None or va == vb: continue
            for (sa, d, mult) in reps(va):
                sb = f'{vb * mult:.{d}f}' if ',' not in sa else f'{int(round(vb)):,}'
                if sa == sb: continue
                if len(sa.replace('-', '').replace('.', '').lstrip('0')) < 2: continue   # 한 자리 숫자는 우연 일치가 많아 제외
                pat = re.compile(r'(?<![\d.,])' + re.escape(sa) + r'(?![\d])')
                for ln, s in TXT:
                    if pat.search(s):
                        hits.append(dict(file=r['file'], key=k, p0=x, v1e=y, shown=sa, new=sb, line=ln, ctx=s[max(0, pat.search(s).start() - 40):pat.search(s).end() + 30]))
jdump(hits, f's8_scan_{grp}.json')
seen = set()
for h in hits:
    key = (h['line'], h['shown'], h['file'])
    if key in seen: continue
    seen.add(key)
    out(h['line'], h['file'], h['key'][:50], h['p0'], '->', h['v1e'], '| shown', h['shown'], '->', h['new'], '|', h['ctx'])
out('n hits', len(hits), 'unique', len(seen))
