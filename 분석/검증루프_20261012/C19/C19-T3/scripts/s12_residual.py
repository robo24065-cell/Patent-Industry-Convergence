# C19-T3 s12: facts ↔ 원고 부록 숫자 토큰 잔여 불일치(C18-T3 s7_residual.py 지표 그대로: A = facts 조각 토큰 중 부록 같은 ID 값 칸에 없는 것, B = 부록 값 칸 토큰 중 facts 조각+조건에 없는 것)
# 사용: python s12_residual.py <facts 폴더> <원고 md> <라벨>  → cache/s12_<라벨>.json
import csv, re, sys, json, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
fdir, mpath, label = sys.argv[1], sys.argv[2], sys.argv[3]
ms = open(mpath, encoding='utf-8').read().replace('\r\n', '\n').split('\n')
st = next(i for i, s in enumerate(ms) if s.startswith('# [부록')); en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
APP = {}
for i in range(st, en):
    s = ms[i]
    if s.startswith('| ') and not s.startswith('| ---'):
        c = [x.strip() for x in s.strip().strip('|').split(' | ')]
        if len(c) >= 4 and c[0] != 'id':
            key = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0])
            APP[key] = dict(line=i + 1, val=' | '.join(c[2:-1]))
NUM = re.compile(r'(?<![A-Za-z0-9_])[−\-+]?\d[\d,]*(?:\.\d+)?')
def toks(s):
    out = set()
    for m in NUM.finditer(s.replace('–', ' – ')):
        t = m.group(0).replace('−', '-').replace('+', '')
        t = t.replace(',', '') if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', t) else t.rstrip(',')
        t = t.lstrip('-') if t.startswith('-') else t
        out.add(t)
    return out
def load(p):
    rows = list(csv.reader(io.StringIO(open(p, encoding='utf-8-sig').read())))
    h = rows[0]; out = []
    for r in rows[1:]:
        if len(r) >= len(h): out.append(dict(zip(h, r[:len(h) - 1] + [','.join(r[len(h) - 1:])])))
    return out
res = []; ids = set()
for name in ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']:
    for r in load(os.path.join(fdir, name)):
        i = r['id']; ids.add(i); frag = r.get('정확한 문장 조각', ''); cond = r.get('조건·경고', '')
        if i in APP:
            ft = toks(frag); at = toks(APP[i]['val'])
            mA = sorted(ft - at); mB = sorted(at - (ft | toks(cond)))
            res.append(dict(파일=name, id=i, 부록행=APP[i]['line'], nA=len(mA), A=' '.join(mA), nB=len(mB), B=' '.join(mB)))
app_rows = [r for r in res]
S = dict(label=label, 부록ID=len(APP), 대조행=len(app_rows), A행=sum(1 for r in app_rows if r['nA']), B행=sum(1 for r in app_rows if r['nB']),
         A또는B=sum(1 for r in app_rows if r['nA'] or r['nB']), 부록ID_facts없음=[k for k in APP if k not in ids],
         잔여=[(r['id'], r['A'][:120], r['B'][:120]) for r in app_rows if r['nA'] or r['nB']])
json.dump(S, open(os.path.join(C, f's12_{label}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(os.path.join(C, f's12_{label}.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print(json.dumps({k: v for k, v in S.items() if k != '잔여'}, ensure_ascii=False))
for x in S['잔여']: print('  ', x)
