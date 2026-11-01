# -*- coding: utf-8 -*-
"""C09-T2 s4: 커버리지 토큰 ↔ 재계산 결과 결합, 구조·정의 상수 표시, coverage_map.csv / recompute.csv 작성."""
import csv, os, re, json
from collections import Counter, defaultdict
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cov = list(csv.DictReader(open(os.path.join(BASE, 'cache', 'coverage_raw.csv'), encoding='utf-8-sig')))
rec = list(csv.DictReader(open(os.path.join(BASE, 'cache', 'recompute_raw.csv'), encoding='utf-8-sig')))
CUR = open(os.path.join(BASE, 'cache', 'manuscript_now_88d77045.md'), encoding='utf-8').read().split('\n')

def norm_tok(t):
    t = t.strip().rstrip(',')
    t = re.sub(r'[^0-9.,/eE+\-−]+$', '', t)  # 단위 제거
    return t.replace('+', '').replace('−', '-').replace(',', '').lstrip('≤')

rmap = defaultdict(list)
for r in rec:
    rmap[(int(r['line']), norm_tok(r['token']))].append(r)

TOKS = list(csv.DictReader(open(os.path.join(BASE, 'cache', 'tokens.csv'), encoding='utf-8-sig')))
def struct_rule(t, line_text):
    tok = t['token']; unit = t['unit']; ctx = t['context']
    p0 = int(t['pos'])
    after = line_text[p0 + len(tok):][:14]
    before = line_text[max(0, p0 - 12):p0]
    class _L(str):
        pass
    line_text = line_text  # 위치 기반 판정
    v = tok.replace(',', '')
    if v == '5' and unit == '개' and after.startswith('개국'):
        return '5개국(제공 데이터 국가 수: c1_label_agreement 국가 5개)'
    if v == '50' and unit == '개' and re.match(r'개 ?(산업|KSIC|풀|세세분류|제조업)', after):
        return '제공 KSIC 50개(concordance_resolution_check n_provided_ksic=50)'
    if v == '5' and unit == '자':
        return 'KSIC 5자리(분류 정의)'
    if v == '3' and unit == '자':
        return 'KSIC 3자리(분류 정의)'
    if v == '9' and unit == '개' and re.match(r'개 ?(트랙|사건|조기경보|모두)', after):
        return '9개 트랙(R6·R10 결과 문서 트랙 목록)'
    if v == '1' and unit == '회':
        return '최종시험 1회·연 1회(프로토콜 정의)'
    if v == '95' and after.startswith('% CI'):
        return '95% 신뢰구간(정의)'
    if v == '1' and after.startswith('위'):
        return '서수(1위)'
    if v == '2' and after.startswith('위'):
        return '서수(2위)'
    if unit in ('년', '개월') and re.search(r'(안|내|창|모멘텀|이력|채점|늦춰|옮겨|지연|전향|3년 내|안에|합)', line_text[max(0, p0 - 10):p0 + 14]):
        return '기간 정의(창 길이·시차)'
    if v in ('10', '1') and unit == '%' and re.search(r'상위 ?$', before[-4:]):
        return '상위 k% 정의'
    if v in ('100', '1,000', '1000') and re.search(r'상위 ?$|후보 ?(셀 )?$|진입 후보 ?$', before[-8:]):
        return '상위 k 정의'
    if v == '1/100' or v == '1/3':
        return '분율 정의(상위 1%·상위 1/3)'
    if v == '10' and after.startswith('인'):
        return 'KOSIS 10인 이상(조사 범위 정의; C07-T1 data_check)'
    if v in ('0', '1', '2', '3', '4', '5') and unit == '건' and re.search(r'(직전|셀은|이상|이하|0건|KR)', line_text[max(0, p0 - 8):p0 + 8]):
        return '건수 문턱 정의'
    if v in ('300',) and unit == '명':
        return '규모 구간 정의(res_sub >=300)'
    if v in ('20',) and unit == '건':
        return '구간 정의(res_sub 특허 >=20)'
    if v == '1' and re.search(r'RTA(가|>) ?$', before[-6:]):
        return 'RTA 기준값 1(정의)'
    if v == '0' and (after.startswith('보다') or after.startswith(' 포함') or after.startswith('에 닿') or after.startswith('을 포함') or after.startswith('이 0')):
        return '0 기준(정의)'
    if v == '0.7' and unit == '배':
        return '격차 기준값(정의)'
    if v == '15' and unit == '%':
        return 'WIPO 대조 문턱(정의)'
    if v == '35' and unit == '개':
        return 'WIPO 35개 기술분야(분류 정의)'
    if v == '48' and unit == '개' and 'KSIC' in after:
        return '사분면 대상 48개 KSIC(m4 rho n=48)'
    if v == '34' and unit == '개':
        return '연계표 비구분 34개(50−16, concordance)'
    if v == '15' and unit == '개':
        return '그림 6 상위 15개(그림 정의)'
    if v == '1' and unit == '행':
        return '1행 = (출원, KSIC) 쌍(데이터 구조 정의)'
    if v == '1' and re.search(r'\(1 ?$|÷ ?\(1 ?$', before):
        return 'skill 식의 상수 1'
    if v == '2' and unit == '개' and after.startswith('개 이상'):
        return '다대다 정의(2개 이상 산업)'
    if v == '5' and re.match(r'–10%', after):
        return '상위 k% 구간 정의'
    if v == '5' and unit == '개' and re.search(r'이 $', before):
        return '상위 5개 합계(정의)'
    if v in ('1년', '1') and unit == '년':
        return '시차·연차 정의'
    if unit == '년' and re.search(r'(단계|확장|시스템|차)', line_text):
        return '연차 정의'
    if unit == '건' and re.search(r'키당 ?$', before):
        return 'H-mg-3 정의(키당 3건)'
    return None

def scope_name(s):
    return {'S': '요약·표지', 'T': '표', 'K': '4장 결론', 'B': '본문', 'R': '부록·출처표', 'M': '편집 메모', 'X': '참고문헌'}.get(s, s)

out = []
assert len(TOKS) == len(cov)
for t, tk in zip(cov, TOKS):
    t = dict(t); t['pos'] = tk['pos']
    ln = int(t['line'])
    key = (ln, norm_tok(t['token']))
    rr = rmap.get(key, [])
    rv = rs = rsrc = rval = ''
    if t['cls'] in ('X',):
        status = '제외'
    else:
        if rr:
            r = rr[0]
            rv = r['verdict']; rsrc = r['source']; rval = r['src_value']; rs = r['srctype']
            status = '재계산'
        else:
            st = struct_rule(t, CUR[ln - 1])
            if st and t['scope'] in 'STKB':
                rv = '정확'; rsrc = st; rs = '구조·정의 상수'; status = '구조·정의'
            elif t['scope'] == 'M':
                status = '대상 외(편집 메모)'
            elif t['cls'] == 'A':
                status = 'A(이전 원천 확인)'
            else:
                status = '미처리'
    out.append(dict(line=ln, scope=t['scope'], scope_name=scope_name(t['scope']), token=t['token'], unit=t['unit'], range=t['range'],
                    excluded=t['excluded'], cls=t['cls'], line_since=t['line_since'], prior_audit=t['matched'], prior_evidence=t['evidence'],
                    recompute_status=status, recompute_verdict=rv, recompute_srctype=rs, recompute_source=rsrc[:200], src_value=rval,
                    context=t['context']))
# 부록(R): 같은 표기의 본문 토큰이 원천 재계산 '정확'이면 '간접 확인'(원천 재계산 아님)
okvals = {norm_tok(o['token']) for o in out if o['scope'] in 'STKB' and o['recompute_status'] == '재계산' and o['recompute_verdict'] == '정확' and len(norm_tok(o['token'])) >= 3}
for o in out:
    if o['scope'] == 'R' and o['recompute_status'] == '미처리':
        if norm_tok(o['token']) in okvals:
            o['recompute_status'] = '미처리(간접: 본문 같은 표기 재계산 정확)'
        else:
            o['recompute_status'] = '미처리(부록·제출본 제외, 우선순위 최하)'
with open(os.path.join(BASE, 'coverage_map.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

# recompute.csv: 재계산 행(검증 대상) + 구조·정의 + 미처리 목록
covkey = defaultdict(list)
for o in out:
    covkey[(o['line'], norm_tok(o['token']))].append(o)
rows = []
for r in rec:
    k = (int(r['line']), norm_tok(r['token']))
    cl = sorted({o['cls'] for o in covkey.get(k, [])})
    sc = sorted({o['scope'] for o in covkey.get(k, [])})
    rows.append(dict(line=r['line'], scope=''.join(sc), coverage_cls=''.join(cl) or '(토큰 매칭 없음)', token=r['token'], src_value=r['src_value'], rounded=r['rounded'],
                     verdict=r['verdict'], srctype=r['srctype'], source=r['source'], note=r['note']))
for o in out:
    if o['recompute_status'] == '구조·정의':
        rows.append(dict(line=o['line'], scope=o['scope'], coverage_cls=o['cls'], token=o['token'] + o['unit'], src_value='', rounded='', verdict='정확(정의)',
                         srctype='구조·정의 상수', source=o['recompute_source'], note=''))
    if o['recompute_status'].startswith('미처리') and o['cls'] in 'BCD':
        rows.append(dict(line=o['line'], scope=o['scope'], coverage_cls=o['cls'], token=o['token'] + o['unit'], src_value='', rounded='', verdict='미처리',
                         srctype='', source=o['recompute_status'], note=o['context'][:120]))
rows.sort(key=lambda x: (int(x['line']), x['token']))
with open(os.path.join(BASE, 'recompute.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

inc = [o for o in out if o['cls'] != 'X']
print('included tokens', len(inc))
for sc in 'STKBRM':
    c = Counter(o['cls'] for o in inc if o['scope'] == sc)
    st = Counter(o['recompute_status'] for o in inc if o['scope'] == sc)
    print(sc, dict(sorted(c.items())), dict(st))
print('unprocessed STKB B/C/D:')
for o in inc:
    if o['scope'] in 'STKB' and o['recompute_status'] == '미처리' and o['cls'] in 'BCD':
        print(o['line'], o['scope'], o['cls'], o['token'] + o['unit'], '|', o['context'][15:75])
