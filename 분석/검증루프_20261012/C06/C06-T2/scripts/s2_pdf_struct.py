# C06-T2 s2: pdf 쪽수·텍스트 포함 여부, 부록/편집메모 혼입, 표·그림 번호 순서, 캡션 출처
import re, json, os, sys, hashlib
import fitz
sys.stdout.reconfigure(encoding='utf-8')
REP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T2"
CACHE = os.path.join(OUT, 'cache')

def sq(s):
    return re.sub(r'\s+', '', s)

res = {}
for ver in ['v5', 'v6', 'v7']:
    pdf = os.path.join(REP, f'보고서_초안_{ver}.pdf')
    doc = fitz.open(pdf)
    pages = [p.get_text() for p in doc]
    full = sq(''.join(pages))
    r = {'pdf': pdf, 'sha1': hashlib.sha1(open(pdf, 'rb').read()).hexdigest(), 'n_pages': len(pages),
         'chars_per_page': [len(sq(t)) for t in pages]}
    # 마지막 쪽 채움: 마지막 텍스트 블록의 y1 / 쪽 높이
    last = doc[-1]; blocks = [b for b in last.get_text('blocks') if b[4].strip()]
    r['last_page_fill'] = round(max(b[3] for b in blocks) / last.rect.height, 3) if blocks else 0
    r['last_page_tail'] = pages[-1].strip()[-120:]
    # docx 문단 포함 여부
    key = {'v5': 's1_v5docx_vs_for_v5', 'v6': 's1_v6docx_vs_for_v6', 'v7': 's1_v7docx_vs_md_v3'}[ver]
    j = json.load(open(os.path.join(CACHE, key + '.json'), encoding='utf-8'))
    miss = []
    for k, t in j['got']:
        s = sq(t)
        if not s: continue
        if s not in full:
            # 표 셀 등 줄바꿈으로 갈린 경우: 앞 15자·뒤 15자 모두 있으면 부분 일치로 본다
            if s[:15] in full and s[-15:] in full:
                continue
            miss.append(t[:80])
    r['n_docx_paras'] = len(j['got']); r['n_missing_in_pdf'] = len(miss); r['missing_examples'] = miss[:10]
    bad = ['[부록', '편집 메모', '숫자 출처표', 'facts.csv', 'E-C0', 'ERRATA', '원고 아님', '제출본에서 삭제', '[사후 결정', 'TODO']
    r['appendix_leak_pdf'] = {b: (sq(b) in full) for b in bad}
    docx_txt = sq(' '.join(t for _, t in j['got']))
    r['appendix_leak_docx'] = {b: (sq(b) in docx_txt) for b in bad}
    res[ver] = r

# 표·그림 번호 순서(현재 원고 제출 범위)
md = open(os.path.join(REP, 'manuscript_v3.md'), encoding='utf-8').read()
sub = md[:re.search(r'^# \[(부록|편집 메모)', md, re.M).start()]
lines = sub.split('\n')
caps = []; refs = []
for i, l in enumerate(lines, 1):
    m = re.match(r'^\*\*표\s*(\d+)\.', l.strip())
    if m: caps.append(('표', int(m.group(1)), i, l.strip()[:60]))
    m = re.match(r'^그림\s*(\d+)\.', l.strip())
    if m: caps.append(('그림', int(m.group(1)), i, l.strip()))
    m = re.match(r'^\[그림\s*(\d+)\s*[:：]\s*([^\]\s]+\.png)', l)
    if m: caps.append(('그림자리', int(m.group(1)), i, m.group(2)))
    for mm in re.finditer(r'(표|그림)\s*(\d+)', l):
        if not re.match(r'^\*\*표|^그림\s*\d+\.|^\[그림', l.strip()):
            refs.append((mm.group(1), int(mm.group(2)), i))
order = {}
for kind in ['표', '그림']:
    nums = [n for k, n, i, _ in caps if k == kind]
    order[kind] = {'sequence': nums, 'is_1_to_n': nums == list(range(1, len(nums) + 1))}
first_ref = {}
for k, n, i in refs:
    first_ref.setdefault((k, n), i)
cap_line = {(k if k != '그림자리' else '그림', n): i for k, n, i, _ in caps if k in ('표', '그림자리')}
ref_check = []
for (k, n), i in sorted(first_ref.items()):
    ref_check.append({'kind': k, 'num': n, 'first_ref_line': i, 'caption_line': cap_line.get((k, n)),
                      'exists': (k, n) in cap_line})
fig_caps = [{'num': n, 'line': i, 'caption': t, 'has_source': ('출처' in t)} for k, n, i, t in caps if k == '그림']
res['numbering'] = order; res['refs'] = ref_check; res['fig_captions'] = fig_caps
res['fig_files'] = [(n, t) for k, n, i, t in caps if k == '그림자리']
json.dump(res, open(os.path.join(CACHE, 's2_pdf_struct.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for v in ['v5', 'v6', 'v7']:
    r = res[v]; print(v, r['n_pages'], 'fill', r['last_page_fill'], 'miss', r['n_missing_in_pdf'], 'of', r['n_docx_paras'],
                      'leak_pdf', [k for k, x in r['appendix_leak_pdf'].items() if x], 'leak_docx', [k for k, x in r['appendix_leak_docx'].items() if x])
    print('  missing ex:', r['missing_examples'][:5])
print(order)
print([x for x in ref_check if not x['exists']])
print([ (x['kind'],x['num'],x['first_ref_line'],x['caption_line']) for x in ref_check])
print([ (c['num'], c['has_source']) for c in fig_caps])
print(res['fig_files'])
