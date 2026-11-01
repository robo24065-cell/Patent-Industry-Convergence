# 제출본(보고서_초안_v9 = drafts\test_v9) 점검: docx 문단 vs 원고 제출 범위, PDF 쪽수·마지막 쪽 채움, 내장 PNG 해시 vs figs\
import sys, os, re, zipfile, hashlib, json, difflib
import fitz
sys.stdout.reconfigure(encoding="utf-8")
MS, DOCX, PDF, FIGS, OUT = sys.argv[1:6]
md = open(MS, encoding='utf-8').read().replace('\r\n', '\n')
m = re.search(r'^# \[(부록|편집 메모)', md, re.M); md = md[:m.start()]
def norm(s):
    s = re.sub(r'[\s*`#|]', '', s)
    s = s.replace('-', '').replace('—', '').replace('–', '').replace('−', '')
    return s
md_paras = []
for l in md.split('\n'):
    t = l.strip()
    if not t or re.fullmatch(r'-{3,}', t) or re.fullmatch(r'\|?[\s:\-|]+\|?', t) and '-' in t: continue
    fm = re.match(r'^\[그림\s*\d+\s*[:：]\s*([^\]\s]+\.png)\s*\]?(.*)$', t)
    if fm:
        md_paras.append(('IMG', fm.group(1)))
        if fm.group(2).strip(): md_paras.append(('P', fm.group(2).strip()))
        continue
    t = re.sub(r'^\d+\.\s+', '', t); t = re.sub(r'^[-*]\s+', '', t)
    if t.startswith('|'):
        for c in t.strip('|').split('|'): md_paras.append(('P', c.strip()))
    else: md_paras.append(('P', t))
z = zipfile.ZipFile(DOCX)
xml = z.read('word/document.xml').decode('utf-8')
dx_paras = []
for p in re.findall(r'<w:p[ >].*?</w:p>', xml, re.S):
    txt = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
    txt = txt.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&apos;', "'")
    if '<w:drawing' in p or '<pic:pic' in p: dx_paras.append(('IMG', ''))
    if txt.strip(): dx_paras.append(('P', txt))
A = norm(''.join(t for k, t in md_paras if k == 'P')); B = norm(''.join(t for k, t in dx_paras if k == 'P'))
sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
diffs = [(op, A[i1:i2][:200], B[j1:j2][:200]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']
# 문단 단위 대조(정규화 후)
pa = [norm(t) for k, t in md_paras if k == 'P' and norm(t)]; pb = [norm(t) for k, t in dx_paras if k == 'P' and norm(t)]
para_ops = [(op, i2-i1, j2-j1) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pa, pb, autojunk=False).get_opcodes() if op != 'equal']
# 이미지: 문서 순서대로 rId → media
rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels)); rid2t.update({a: b for b, a in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)})
order = re.findall(r'r:embed="(rId\d+)"', xml)
md_imgs = [t for k, t in md_paras if k == 'IMG']
imgs = []
for n, rid in enumerate(order):
    data = z.read('word/' + rid2t[rid]); h = hashlib.sha1(data).hexdigest()
    exp = md_imgs[n] if n < len(md_imgs) else None
    fh = hashlib.sha1(open(os.path.join(FIGS, exp), 'rb').read()).hexdigest() if exp and os.path.exists(os.path.join(FIGS, exp)) else None
    imgs.append(dict(order=n+1, media=rid2t[rid], sha1=h[:12], expected=exp, figs_sha1=(fh or '')[:12], match=(h == fh)))
doc = fitz.open(PDF)
pages = doc.page_count
last = doc[pages-1]
blocks = [b for b in last.get_text('blocks') if b[4].strip()]
ph = last.rect.height
maxy = max(b[3] for b in blocks) if blocks else 0
# 쪽 번호(바닥글) 블록 제외한 본문 최하단
body = [b for b in blocks if b[1] < ph * 0.93]
body_maxy = max(b[3] for b in body) if body else 0
last_text_tail = last.get_text()[-300:]
fig5_page = [i+1 for i in range(pages) if '그림 5.' in doc[i].get_text()]
res = dict(manuscript=MS, docx=DOCX, pdf=PDF,
           docx_sha1=hashlib.sha1(open(DOCX, 'rb').read()).hexdigest(), pdf_sha1=hashlib.sha1(open(PDF, 'rb').read()).hexdigest(),
           ms_chars_norm=len(A), docx_chars_norm=len(B), char_diff_ops=len(diffs), char_diffs=diffs[:20],
           md_paras=len(pa), docx_paras=len(pb), para_diff_ops=para_ops[:20],
           images=imgs, n_images_docx=len(order), n_images_md=len(md_imgs),
           pdf_pages=pages, last_page_body_bottom_frac=round(body_maxy/ph, 3), last_page_any_bottom_frac=round(maxy/ph, 3),
           last_page_tail=last_text_tail, fig5_caption_page=fig5_page)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('char_diffs',)}, ensure_ascii=False, indent=1)[:4000])
print('char_diffs', diffs[:10])
