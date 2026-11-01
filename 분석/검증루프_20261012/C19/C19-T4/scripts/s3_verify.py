# -*- coding: utf-8 -*-
# C19-T4 s3(반박): 불일치 항목에 대해 (a) 상태표(C18-T3) 근거 문자열 'NNN행: 문자열'이 v19 그 줄에 실제로 있는지 기계 검색,
# (b) 원 작업자 독립 판정 파일(verify\v1b_status_all.csv) 해당 행을 붙인다. 결과 cache\s3_verify.json.
import csv, os, re, json
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(os.path.dirname(D))
R = os.path.join(os.path.dirname(B), '보고서_20261012')
V19 = open(os.path.join(R, 'drafts', 'manuscript_for_v19.md'), encoding='utf-8').read().split('\n')
V20 = open(os.path.join(R, 'manuscript_v3.md'), encoding='utf-8').read().split('\n')
def nows(s):
    return re.sub(r'\s+', '', s.replace('**', '').replace('*', ''))
dis = list(csv.DictReader(open(os.path.join(D, 'cache', 'disagreements_blind.csv'), encoding='utf-8-sig')))
key = {r['ID']: r for r in csv.DictReader(open(os.path.join(D, 'cache', 'sample_key.csv'), encoding='utf-8-sig'))}
v1b = {}
p = os.path.join(B, 'C18', 'C18-T3', 'verify', 'v1b_status_all.csv')
if os.path.exists(p):
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        v1b[r.get('ID', '')] = r
out = []
for r in dis:
    k = key[r['ID']]
    ev = []
    for m in re.finditer(r'(\d+)행:\s*([^/;]+)', k['근거']):
        ln = int(m.group(1)); s = m.group(2).strip()
        ok19 = 0 < ln <= len(V19) and nows(s) in nows(V19[ln - 1])
        any20 = [i + 1 for i, x in enumerate(V20) if nows(s) in nows(x)][:3]
        ev.append({'line': ln, 'str': s[:80], 'in_v19_line': ok19, 'v20_lines': any20})
    out.append({'ID': r['ID'], '상태표': k['상태'], '상태표근거': k['근거'][:300], '근거검색': ev,
                'C18_v1b': {kk: vv[:120] for kk, vv in v1b.get(r['ID'], {}).items()} if r['ID'] in v1b else None})
json.dump(out, open(os.path.join(D, 'cache', 's3_verify.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
