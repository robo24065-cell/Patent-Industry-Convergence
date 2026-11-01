# -*- coding: utf-8 -*-
# C19-T4 s5: 제출본(docx·pdf)과 md 원고의 같은 줄 대조. 본문(445행 앞) 근거 줄이 있는 판정 항목에서 20개 무작위(시드 20261019).
# 근거 문자열을 ' / '·'…'로 쪼갠 조각(6자 이상)이 md 해당 줄, docx 텍스트, pdf 텍스트(공백 제거 비교)에 있는지 본다. v19와 v20 둘 다.
import csv, os, re, random, json, hashlib
import docx, fitz
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(os.path.dirname(D))
R = os.path.join(os.path.dirname(B), '보고서_20261012')
SEED = 20261019
rows = list(csv.DictReader(open(os.path.join(D, 'cache', 'judg_partial.tsv'), encoding='utf-8'), delimiter='\t'))

def body_line(r):
    ls = [int(x) for x in re.findall(r'\d+', r['v19근거줄'].split('(')[0])] if r['v19근거줄'] else []
    return [x for x in ls if x < 445]
cand = [r for r in rows if body_line(r) and r['근거문자열'] and not r['근거문자열'].startswith('(') and 'figs' not in r['근거문자열']]
cand.sort(key=lambda r: r['ID'])
pick = random.Random(SEED).sample(cand, 20)

def nows(s):
    s = s.replace('**', '').replace('*', '').replace(' ', ' ')
    return re.sub(r'\s+', '', s)

def docx_text(p):
    d = docx.Document(p)
    parts = [x.text for x in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    return nows('\n'.join(parts))

def pdf_text(p):
    doc = fitz.open(p)
    s = ''.join(pg.get_text() for pg in doc)
    s = s.replace('­', '')
    return nows(s)

res = {}
for ver, md, dx, pf in [('v19', os.path.join(R, 'drafts', 'manuscript_for_v19.md'), os.path.join(R, '보고서_초안_v19.docx'), os.path.join(R, '보고서_초안_v19.pdf')),
                        ('v20', os.path.join(R, 'manuscript_v3.md'), os.path.join(R, '보고서_초안_v20.docx'), os.path.join(R, '보고서_초안_v20.pdf'))]:
    mdl = open(md, encoding='utf-8').read().split('\n')
    DX = docx_text(dx); PF = pdf_text(pf)
    res[ver] = {'md_sha1': hashlib.sha1(open(md, 'rb').read()).hexdigest(), 'docx_sha1': hashlib.sha1(open(dx, 'rb').read()).hexdigest(),
                'pdf_sha1': hashlib.sha1(open(pf, 'rb').read()).hexdigest(), 'items': []}
    for r in pick:
        frags = []
        for a in r['근거문자열'].split(' / '):
            for b in re.split('…|\\|', a):
                b = b.strip(' |')
                b = re.sub(r'^\d+\s', '', b)  # '515 DATA-…' 같은 줄 번호 접두 제거
                if len(nows(b)) >= 6:
                    frags.append(b)
        md_all = nows('\n'.join(mdl[:445]))
        fr = []
        for f in frags:
            nf = nows(f)
            fr.append({'frag': f[:60], 'md': nf in md_all, 'docx': nf in DX, 'pdf': nf in PF})
        res[ver]['items'].append({'ID': r['ID'], 'line': body_line(r), 'n_frag': len(fr),
                                  'md_all': all(x['md'] for x in fr), 'docx_all': all(x['docx'] for x in fr), 'pdf_all': all(x['pdf'] for x in fr), 'frags': fr})
json.dump(res, open(os.path.join(D, 'cache', 's5_docx_pdf.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(os.path.join(D, 'docx_pdf_check.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['판' , 'ID', 'md줄', '조각수', 'md일치', 'docx일치', 'pdf일치', '불일치조각'])
    for ver in res:
        for it in res[ver]['items']:
            bad = '; '.join(f"{x['frag']}(md={int(x['md'])},docx={int(x['docx'])},pdf={int(x['pdf'])})" for x in it['frags'] if not (x['md'] and x['docx'] and x['pdf']))
            w.writerow([ver, it['ID'], ';'.join(map(str, it['line'])), it['n_frag'], int(it['md_all']), int(it['docx_all']), int(it['pdf_all']), bad])
summ = {ver: {k: sum(1 for it in res[ver]['items'] if it[k]) for k in ['md_all', 'docx_all', 'pdf_all']} for ver in res}
summ['picked'] = [r['ID'] for r in pick]
json.dump(summ, open(os.path.join(D, 'cache', 's5_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
