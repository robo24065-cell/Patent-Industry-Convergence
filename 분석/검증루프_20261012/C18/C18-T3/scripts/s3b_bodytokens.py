# C18-T3 s3b: 본문 숫자 토큰 중 부록·facts 어디에도 없는 것 목록(확인된 오류 후보 탐색용, 읽기 전용)
import re, sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; W = AN + r"\검증루프_20261012\C18\C18-T3"
ms = open(R + r"\manuscript_v3.md", encoding='utf-8').read().split('\n')
st = next(i for i, s in enumerate(ms) if s.startswith('# [부록')); en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
NUM = re.compile(r'(?<![A-Za-z0-9_])[−\-+]?\d[\d,]*(?:\.\d+)?')
def toks(s):
    out = set()
    for m in NUM.finditer(s.replace('–', ' – ').replace('~', ' ~ ')):
        t = m.group(0).replace('−', '').replace('-', '').replace('+', '')
        if re.fullmatch(r'\d{1,3}(,\d{3})+(\.\d+)?', t): t = t.replace(',', '')
        out.add(t.rstrip(','))
    return out
app = set()
for s in ms[st:en]: app |= toks(s)
facts = set()
for f in ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']:
    facts |= toks(open(os.path.join(R, 'facts', f), encoding='utf-8-sig').read())
miss = {}
for i, s in enumerate(ms[:st], 1):
    if s.startswith('[그림') or i >= 433: continue
    for t in toks(s):
        if t in app or t in facts: continue
        if re.fullmatch(r'(19|20)\d\d', t) or len(t.replace('.', '')) <= 1: continue
        miss.setdefault(t, []).append(i)
json.dump(miss, open(W + r"\cache\s3b_body_tokens_unmatched.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(len(miss))
for t, l in sorted(miss.items(), key=lambda x: x[1][0]): print(t, l)
