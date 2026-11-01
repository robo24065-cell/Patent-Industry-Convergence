# -*- coding: utf-8 -*-
# C19-T4 s8: 눈가림 해제(sample_key.csv) 뒤 일치율·Cohen kappa·불일치 표·숨은 필수 상한. 반박 전(blind) 판정 기준.
import csv, os, json, collections, math, hashlib
from scipy.stats import beta
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
key = {r['ID']: r for r in csv.DictReader(open(os.path.join(D, 'cache', 'sample_key.csv'), encoding='utf-8-sig'))}
jud = list(csv.DictReader(open(os.path.join(D, 'blind_judgments.csv'), encoding='utf-8-sig')))
final_fn = os.path.join(D, 'cache', 'final_overrides.json')
overrides = json.load(open(final_fn, encoding='utf-8')) if os.path.exists(final_fn) else {}
J4 = {'반영': '반영', '부분': '부분', '미반영-여전히 필요': '미반영', '폐기 부당': '미반영', '미반영-불필요(다른 항목이 대체)': '폐기', '폐기 타당': '폐기'}
S4 = {'반영됨': '반영', '부분': '부분', '미반영': '미반영', '폐기·대체': '폐기'}
CATS = ['반영', '부분', '미반영', '폐기']

def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in CATS) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else None

def cp_upper(x, n, a=0.05):
    return 1.0 if x >= n else float(beta.ppf(1 - a / 2, x + 1, n - x))

recs = []
for j in jud:
    k = key[j['ID']]
    jb = j['판정']
    jf = overrides.get(j['ID'], {}).get('최종판정', jb)
    recs.append({'ID': j['ID'], '층': k['층'], '원등급': k['원등급'], '범위': k['범위'], '상태표': k['상태'], '상태표분류': k['분류'], '상태표근거': k['근거'][:300],
                 '상태표4': S4[k['상태']], '눈가림판정': jb, '눈가림4': J4[jb], '최종판정': jf, '최종4': J4[jf],
                 '재등급': overrides.get(j['ID'], {}).get('재등급', j['재등급']), '숫자대조': j['숫자대조'], 'v19근거줄': j['v19근거줄'], 'v20상태': j['v20상태'], '메모': j['메모'],
                 '반박메모': overrides.get(j['ID'], {}).get('사유', '')})

def subset(f):
    return [r for r in recs if f(r)]
groups = {
 '전체(A–E)': recs,
 '무작위 층만(A·D·C_rand)': subset(lambda r: any(s in r['층'].split('+') for s in ['A', 'D', 'C_rand'])),
 'A 반영됨': subset(lambda r: 'A' in r['층'].split('+')),
 'B 부분': subset(lambda r: 'B' in r['층'].split('+')),
 'C 미반영': subset(lambda r: any(s in r['층'].split('+') for s in ['C_rand', 'C_G'])),
 'D 폐기': subset(lambda r: 'D' in r['층'].split('+')),
 'E 핵심 G*': subset(lambda r: 'E' in r['층'].split('+')),
 '본문·부록·그림만(원고 밖·facts 제외)': subset(lambda r: r['범위'] not in ('원고 밖', 'facts')),
}
res = {'blind_sha1': hashlib.sha1(open(os.path.join(D, 'blind_judgments.csv'), 'rb').read()).hexdigest()}
for g, rs in groups.items():
    po_b, k_b = kappa([(r['상태표4'], r['눈가림4']) for r in rs])
    po_f, k_f = kappa([(r['상태표4'], r['최종4']) for r in rs])
    res[g] = {'n': len(rs), 'agree_blind': po_b, 'kappa_blind': k_b, 'agree_final': po_f, 'kappa_final': k_f}
cm = collections.Counter((r['상태표4'], r['눈가림4']) for r in recs)
res['confusion_blind(상태표→판정)'] = {f'{a}→{b}': cm[(a, b)] for a in CATS for b in CATS if cm[(a, b)]}
cmf = collections.Counter((r['상태표4'], r['최종4']) for r in recs)
res['confusion_final'] = {f'{a}→{b}': cmf[(a, b)] for a in CATS for b in CATS if cmf[(a, b)]}
# (i) 숨은 필수·권장, (ii) 반영인데 숫자 불일치
need = ('미반영-여전히 필요', '폐기 부당', '부분')
res['i_hidden_must_blind'] = [r['ID'] for r in recs if r['상태표'] in ('미반영', '폐기·대체') and r['눈가림판정'] in need and r['재등급'] in ('필수', '권장')]
res['i_hidden_must_final'] = [r['ID'] for r in recs if r['상태표'] in ('미반영', '폐기·대체') and r['최종판정'] in need and r['재등급'] in ('필수', '권장')]
res['ii_reflected_number_mismatch'] = [r['ID'] for r in recs if r['상태표'] == '반영됨' and r['숫자대조'].startswith('불일치')]
res['regrade_counts'] = dict(collections.Counter(r['재등급'] or '-' for r in recs))
# CP 상한
popC = 159; popD = 66
xC = sum(1 for r in recs if 'C_rand' in r['층'] and r['최종판정'] in need and r['재등급'] in ('필수', '권장'))
nC = sum(1 for r in recs if 'C_rand' in r['층'])
xD = sum(1 for r in recs if 'D' in r['층'].split('+') and r['최종판정'] in need and r['재등급'] in ('필수', '권장'))
nD = sum(1 for r in recs if 'D' in r['층'].split('+'))
res['cp_hidden_must'] = {'미반영(G 밖 159, G 0개)': {'x': xC, 'n': nC, 'p_upper': cp_upper(xC, nC), 'N_upper': cp_upper(xC, nC) * popC},
                         '폐기(66, G* 2개는 E로 전수)': {'x': xD, 'n': nD, 'p_upper': cp_upper(xD, nD), 'N_upper': cp_upper(xD, nD) * (popD - 2)}}
# 반영됨인데 부분·미반영 판정(선택 수준 불완전 반영)
xA = sum(1 for r in recs if 'A' in r['층'].split('+') and r['최종4'] != '반영')
nA = sum(1 for r in recs if 'A' in r['층'].split('+'))
res['cp_reflected_incomplete'] = {'x': xA, 'n': nA, 'p_upper': cp_upper(xA, nA), 'N_upper_of_209': cp_upper(xA, nA) * 209}
json.dump(res, open(os.path.join(D, 'agreement.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(os.path.join(D, 'sample.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
dis = [r for r in recs if r['상태표4'] != r['눈가림4']]
with open(os.path.join(D, 'cache', 'disagreements_blind.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(dis)
