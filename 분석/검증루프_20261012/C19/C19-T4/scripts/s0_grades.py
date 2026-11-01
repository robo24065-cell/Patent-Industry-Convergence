# -*- coding: utf-8 -*-
"""C19-T4 s0: ERRATA 원 등급(헤더 꼬리표) 추출. 상태 칸은 집계(교차표)만 출력하고 항목별 상태는 출력하지 않는다."""
import csv, re, sys, json, collections, os
BASE = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012'
OUT = os.path.join(BASE, r'C19\C19-T4\cache')
os.makedirs(OUT, exist_ok=True)
ST = os.path.join(BASE, r'C18\C18-T3\errata_status_v18.csv')
ER = os.path.join(BASE, 'ERRATA.txt')
lines = open(ER, encoding='utf-8').read().split('\n')
rows = list(csv.DictReader(open(ST, encoding='utf-8-sig')))

def header_text(r):
    try:
        ln = int(r['ERRATA줄'])
    except Exception:
        return ''
    t = lines[ln - 1] if 0 < ln <= len(lines) else ''
    return t

def grade(h):
    # 헤더(첫 줄)의 꼬리표 부분만 본다: ID 뒤 첫 대괄호 묶음들 + 직후 120자
    m = re.search(r'(E-[A-Za-z0-9\-]+)\s*(.*)', h)
    tag = m.group(2)[:160] if m else h[:160]
    br = ' '.join(re.findall(r'\[([^\]]*)\]', tag))
    g = set()
    src = br if br else tag[:60]
    if '필수' in src:
        g.add('필수')
    if '권장' in src:
        g.add('권장')
    # 확인된 오류: '확인된 오류', '확인 오류', '[확인][오류]', '확인·오류', '[오류]'+확인
    if re.search(r'확인된 오류|확인 오류|확인·오류', src) or (re.search(r'(^|[\s\[·])오류', src) and '확인' in src and '오류 아님' not in src):
        g.add('확인된오류')
    return sorted(g), br, tag

out = []
for r in rows:
    h = header_text(r)
    g, br, tag = grade(h)
    out.append({'ID': r['ID'], '원등급': '+'.join(g) if g else '', '헤더꼬리표': br[:120], '헤더': h[:200]})
with open(os.path.join(OUT, 'orig_grades.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
# 집계 교차표(상태 × 원등급 유무)만
ct = collections.Counter()
for r, o in zip(rows, out):
    ct[(r['상태'], 'G' if o['원등급'] else '-')] += 1
gc = collections.Counter(o['원등급'] for o in out)
res = {'crosstab_status_x_hasgrade': {f'{k[0]}|{k[1]}': v for k, v in ct.items()}, 'grade_counts': dict(gc),
       'empty_header': sum(1 for o in out if not o['헤더'])}
json.dump(res, open(os.path.join(OUT, 's0_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=1))
