# -*- coding: utf-8 -*-
# 반박 검증: 잔여 불일치 독립 재계산 (C18 지표 재현 + 부호 보존 변형)
import csv, re, sys, io, json
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석'
T = B + r'\검증루프_20261012\C19\C19-T3'
MS = B + r'\보고서_20261012\manuscript_v3.md'
lines = open(MS, encoding='utf-8').read().replace('\r\n', '\n').split('\n')
st = next(i for i, s in enumerate(lines) if s.startswith('# [부록'))
en = next(i for i, s in enumerate(lines) if s.startswith('# [편집 메모'))
APP = {}
for i in range(st, en):
    s = lines[i]
    if s.startswith('| ') and not s.startswith('| ---'):
        c = [x.strip() for x in s.strip().strip('|').split(' | ')]
        if len(c) >= 4 and c[0] != 'id':
            k = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0])
            if k in APP: print('DUP app id', k)
            APP[k] = (i + 1, ' | '.join(c[2:-1]))
NUM = re.compile(r'(?<![A-Za-z0-9_])[−\-+]?\d[\d,]*(?:\.\d+)?')
def toks(s, keep_sign):
    out = set()
    for m in NUM.finditer(s.replace('–', ' – ')):
        t = m.group(0).replace('−', '-').replace('+', '')
        t = t.replace(',', '') if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', t) else t.rstrip(',')
        if not keep_sign: t = t.lstrip('-')
        out.add(t)
    return out
def rows(p):
    return list(csv.reader(io.StringIO(open(p, encoding='utf-8-sig').read())))
res = {}
for label, d in [('work', T + r'\facts_work'), ('orig', B + r'\보고서_20261012\facts')]:
    for ks in (False, True):
        bad = []; seen = set()
        for f in ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']:
            R = rows(d + '\\' + f); H = R[0]
            for r in R[1:]:
                rec = dict(zip(H, r[:8]))
                i = rec['id']; seen.add(i)
                if i in APP:
                    ft = toks(rec['정확한 문장 조각'], ks); at = toks(APP[i][1], ks)
                    A = ft - at; Bm = at - (ft | toks(rec['조건·경고'], ks))
                    if A or Bm: bad.append((i, sorted(A)[:12], sorted(Bm)[:12]))
        res[f'{label}_sign{int(ks)}'] = dict(nAPP=len(APP), nbad=len(bad), missing=[k for k in APP if k not in seen], bad=bad)
for k, v in res.items():
    print(k, v['nAPP'], v['nbad'], v['missing'])
    for b in v['bad']: print('   ', b)
json.dump(res, open(T + r'\verify_0\r2_residual.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
