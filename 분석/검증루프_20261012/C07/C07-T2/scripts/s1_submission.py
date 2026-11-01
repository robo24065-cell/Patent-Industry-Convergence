# C07-T2 s1: (a) manuscript_for_v7 -> manuscript_v3 diff 재생성, (b) docx(v7, v8) vs 현재 원고 문단 diff,
# (c) PDF 쪽수·문단 포함, (d) docx 내장 PNG 해시 v7 vs v8, (e) 낡은 값 재유입 검사. 읽기 전용, 출력은 C07-T2 안.
import os, sys, re, json, hashlib, difflib, zipfile, csv
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c06t2_s1_docx_diff_copy as C  # C06-T2 s1 사본(같은 폴더). expected_paras/docx_paras/compare 재사용
import fitz
REP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
OUT = os.path.dirname(HERE); CACHE = os.path.join(OUT, 'cache')
sha1 = lambda p: hashlib.sha1(open(p, 'rb').read()).hexdigest()
MD3 = os.path.join(REP, 'manuscript_v3.md'); MD7 = os.path.join(REP, 'drafts', 'manuscript_for_v7.md')
res = {'inputs': {p: sha1(p) for p in [MD3, MD7, os.path.join(REP, 'drafts', 'manuscript_for_v8.md'),
       os.path.join(REP, '보고서_초안_v8.docx'), os.path.join(REP, '보고서_초안_v8.pdf'), os.path.join(REP, 'drafts', 'test_v8.docx'),
       os.path.join(REP, 'drafts', 'test_v8.pdf'), os.path.join(REP, '보고서_초안_v7.docx'), os.path.join(REP, '보고서_초안_v7.pdf')]}}
# (a) 줄 diff
a = open(MD7, encoding='utf-8').read().split('\n'); b = open(MD3, encoding='utf-8').read().split('\n')
hunks = []
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if op != 'equal':
        hunks.append({'op': op, 'v7_lines': f'{i1+1}-{i2}' if i2 > i1 else '', 'cur_lines': f'{j1+1}-{j2}' if j2 > j1 else '',
                      'v7_text': a[i1:i2], 'cur_text': b[j1:j2]})
res['line_hunks'] = hunks
res['n_lines'] = {'v7': len(a) - (1 if a[-1] == '' else 0), 'cur': len(b) - (1 if b[-1] == '' else 0)}
# (b) docx diff
comp = {}
for lab, md, dx in [('v8docx_vs_cur', MD3, os.path.join(REP, '보고서_초안_v8.docx')),
                    ('test_v8docx_vs_cur', MD3, os.path.join(REP, 'drafts', 'test_v8.docx')),
                    ('v7docx_vs_cur', MD3, os.path.join(REP, '보고서_초안_v7.docx')),
                    ('v7docx_vs_for_v7', MD7, os.path.join(REP, '보고서_초안_v7.docx'))]:
    s, d, ef, gf = C.compare(md, dx, lab)
    s.pop('fig_files_md', None)
    comp[lab] = {'summary': s, 'diffs': d}
    if lab == 'v8docx_vs_cur':
        json.dump({'exp': ef, 'got': gf}, open(os.path.join(CACHE, 's1_v8_paras.json'), 'w', encoding='utf-8'), ensure_ascii=False)
res['docx_compare'] = comp
# (c) PDF
def sq(s): return re.sub(r'\s+', '', s)
pdfr = {}
for v in ['v7', 'v8']:
    doc = fitz.open(os.path.join(REP, f'보고서_초안_{v}.pdf'))
    pages = [p.get_text() for p in doc]; full = sq(''.join(pages))
    last = doc[-1]; blocks = [bb for bb in last.get_text('blocks') if bb[4].strip()]
    r = {'n_pages': len(pages), 'last_page_fill': round(max(bb[3] for bb in blocks) / last.rect.height, 3) if blocks else 0,
         'pdf_images': sum(len(p.get_images()) for p in doc)}
    if v == 'v8':
        got = json.load(open(os.path.join(CACHE, 's1_v8_paras.json'), encoding='utf-8'))['got']
        miss = [t[:80] for k, t in got if sq(t) and sq(t) not in full and not (sq(t)[:15] in full and sq(t)[-15:] in full)]
        r['n_docx_paras'] = len(got); r['n_missing_in_pdf'] = len(miss); r['missing_examples'] = miss[:10]
        bad = ['[부록', '편집 메모', '숫자 출처표', 'facts.csv', 'E-C0', 'ERRATA', '원고 아님', '[사후 결정', 'TODO', '검증루프']
        r['appendix_leak_pdf'] = [x for x in bad if sq(x) in full]
        # 변경 문장들이 PDF에 실제로 들어갔는지
        keys = ['동명법인의공란출원을주소로되살려도skill은0.156–0.159다', '가입자300명이상기업·그룹안5자리', '평가문서법인14,872사중NPS11.0%,DART5.5%',
                'N6+0.006–0.010', 'DART우위는2010–14년문서에서도남는다', '권리자청구추정약9%포함', '(초록·청구항미사용)', '공개최강배정보다DART업종에서만']
        r['changed_sentences_in_pdf'] = {k: (k in full) for k in keys}
        # 낡은 문구 재유입
        stale = ['원점2019기준NPS9,054사의26.8%', '로낮아진다', '공식대비4.8%→8.6%', '대기업·그룹안5자리', '제공라벨을만든모델의입력변수는확인하지못했다',
                 '확정사유로추정하면', '0.157–0.159', '+0.008–0.010', '시차가우위를만들지않았다', '업종변경기업1%미만', 'R12표본의비율은계산하지않았다']
        r['stale_in_pdf'] = {k: (k in full) for k in stale}
        r['summary13_대기업구간'] = ('NPS특허가중과대기업구간' in full)
    pdfr[v] = r
res['pdf'] = pdfr
# (d) 내장 PNG
def media(dx):
    z = zipfile.ZipFile(dx); doc = z.read('word/document.xml').decode('utf-8'); rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
    rid2t = {}
    for m in re.finditer(r'<Relationship [^>]*>', rels):
        tag = m.group(0); i = re.search(r'Id="([^"]+)"', tag); t = re.search(r'Target="([^"]+)"', tag)
        if i and t: rid2t[i.group(1)] = t.group(1)
    return [hashlib.sha1(z.read('word/' + rid2t[r])).hexdigest() for r in re.findall(r'r:embed="(rId\d+)"', doc)]
m7 = media(os.path.join(REP, '보고서_초안_v7.docx')); m8 = media(os.path.join(REP, '보고서_초안_v8.docx'))
import glob
figs = {hashlib.sha1(open(f, 'rb').read()).hexdigest(): os.path.basename(f) for f in glob.glob(os.path.join(REP, 'figs', '*.png'))}
res['media'] = {'v7': [(h, figs.get(h)) for h in m7], 'v8': [(h, figs.get(h)) for h in m8], 'same_sequence': m7 == m8,
                'n_v7': len(m7), 'n_v8': len(m8), 'all_v8_in_current_figs': all(h in figs for h in m8)}
json.dump(res, open(os.path.join(CACHE, 's1_submission.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('lines', res['n_lines'], 'hunks', len(hunks))
for h in hunks: print(h['op'], h['v7_lines'], '->', h['cur_lines'])
for k, v in comp.items(): print(k, {x: v['summary'][x] for x in ['n_expected', 'n_docx', 'n_img_docx', 'n_fig_md', 'n_diff_rows', 'ratio']})
for k, v in comp.items():
    for d in v['diffs'][:12]: print('  ', k, d['op'], d['md_line'], d['md_text'][:60], '|', str(d['docx_text'])[:60])
print(json.dumps(pdfr, ensure_ascii=False, indent=1))
print('media same', res['media']['same_sequence'], res['media']['n_v7'], res['media']['n_v8'], res['media']['all_v8_in_current_figs'])
