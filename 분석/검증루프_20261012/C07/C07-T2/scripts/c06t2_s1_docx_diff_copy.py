# C06-T2 s1: 원고(md) -> 기대 문단(md2docx.js 규칙 재현) vs docx 본문 문단(python-docx) diff
# 읽기 전용. 출력은 C06-T2 폴더 안.
import re, sys, json, hashlib, difflib, csv, os
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

REP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T2"
CACHE = os.path.join(OUT, "cache"); os.makedirs(CACHE, exist_ok=True)

def sha1(p):
    return hashlib.sha1(open(p, 'rb').read()).hexdigest()

def runs_text(t):
    # md2docx runs(): **bold** 처리, 굵게 아닌 조각에서 *ital* 처리, ` 제거
    out = []
    for p in re.split(r'(\*\*[^*]+\*\*)', t):
        if not p: continue
        bold = p.startswith('**') and p.endswith('**') and len(p) >= 4
        subs = [p[2:-2]] if bold else re.split(r'(\*[^*\s][^*]*\*)', p)
        for q in subs:
            if not q: continue
            ital = (not bold) and len(q) > 2 and q.startswith('*') and q.endswith('*')
            out.append((q[1:-1] if ital else q).replace('`', ''))
    return ''.join(out)

def parse_row(l):
    s = l.strip()
    s = re.sub(r'^\|', '', s); s = re.sub(r'\|$', '', s)
    return [c.strip() for c in s.split('|')]

def expected_paras(md_path):
    md = open(md_path, encoding='utf-8').read().replace('\r\n', '\n')
    m = re.search(r'^# \[(부록|편집 메모)', md, re.M)
    cutline = None
    if m and m.start() > 0:
        cutline = md[:m.start()].count('\n') + 1
        md = md[:m.start()]
    lines = md.split('\n')
    res = []  # (md_line_no(1-based), kind, text)
    title_done = False; after_title = 0
    i = 0
    while i < len(lines):
        raw = lines[i]; l = raw.rstrip()
        ln = i + 1
        if not l.strip():
            i += 1; continue
        if l.strip().startswith('|') and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?-{2,}', lines[i + 1]):
            block = []; start = i
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append((i + 1, lines[i])); i += 1
            header = parse_row(block[0][1]); ncol = len(header)
            for c in header:
                res.append((block[0][0], 'table_head', runs_text(c)))
            for (bl, row) in block[2:]:
                r = parse_row(row)
                for k in range(ncol):
                    res.append((bl, 'table_cell', runs_text(r[k] if k < len(r) else '')))
            continue
        if re.match(r'^-{3,}$', l.strip()):
            i += 1; continue
        fm = re.match(r'^\[그림\s*(\d+)\s*[:：]\s*([^\]\s]+\.png)\s*\]?(.*)$', l)
        if fm:
            res.append((ln, 'figure', '[IMG:' + fm.group(2) + ']'))
            if fm.group(3) and fm.group(3).strip():
                res.append((ln, 'fig_caption_inline', runs_text(fm.group(3).strip())))
            i += 1; continue
        h = re.match(r'^(#{1,3})\s+(.*)$', l)
        if h:
            text = h.group(2).strip()
            if not title_done and len(h.group(1)) == 1:
                res.append((ln, 'title', text)); title_done = True; after_title = 2
            else:
                res.append((ln, 'heading%d' % len(h.group(1)), runs_text(text)))
            i += 1; continue
        if after_title > 0:
            res.append((ln, 'subtitle', runs_text(l.strip()))); after_title -= 1; i += 1; continue
        t = l.strip()
        if re.match(r'^\*\*(표|그림|박스)\s*\d+', t) or re.match(r'^(그림|표)\s*\d+\.', t):
            res.append((ln, 'caption', runs_text(t))); i += 1; continue
        if re.match(r'^(주[:：]|¹|²|³|출처[:：])', t):
            res.append((ln, 'note', runs_text(t))); i += 1; continue
        b = re.match(r'^(\s*)[-*]\s+(.*)$', raw)
        if b:
            res.append((ln, 'bullet', runs_text(b.group(2)))); i += 1; continue
        n = re.match(r'^(\s*)(\d+)\.\s+(.*)$', raw)
        if n:
            res.append((ln, 'numbered', runs_text(n.group(3)))); i += 1; continue
        res.append((ln, 'para', runs_text(t)))
        i += 1
    return res, cutline

def docx_paras(path):
    d = Document(path)
    body = d.element.body
    out = []
    for child in body.iterchildren():
        tag = child.tag.split('}')[1]
        if tag == 'p':
            p = Paragraph(child, d)
            has_img = bool(child.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip'))
            if has_img:
                out.append(('figure', '[IMG]'))
            txt = p.text
            if txt.strip():
                out.append(('p', txt))
        elif tag == 'tbl':
            t = Table(child, d)
            for row in t.rows:
                seen = set()
                for cell in row.cells:
                    if id(cell._tc) in seen: continue
                    seen.add(id(cell._tc))
                    out.append(('cell', cell.text))
    return out

def norm(s):
    return re.sub(r'\s+', ' ', s).strip()

def compare(md_path, docx_path, label):
    exp, cut = expected_paras(md_path)
    # 빈 셀·빈 문단은 비교 제외(docx에도 텍스트가 없음)
    exp_f = [(ln, k, norm(t)) for (ln, k, t) in exp if norm(t) and k != 'figure']
    exp_figs = [t for (ln, k, t) in exp if k == 'figure']
    got = docx_paras(docx_path)
    got_f = [(k, norm(t)) for (k, t) in got if norm(t) and k != 'figure']
    n_img = sum(1 for (k, t) in got if k == 'figure')
    a = [t for (_, _, t) in exp_f]; b = [t for (_, t) in got_f]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    diffs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal': continue
        L = max(i2 - i1, j2 - j1)
        for k in range(L):
            ea = exp_f[i1 + k] if i1 + k < i2 else None
            gb = got_f[j1 + k] if j1 + k < j2 else None
            diffs.append({'pair': label, 'op': op,
                          'md_line': ea[0] if ea else '', 'md_kind': ea[1] if ea else '',
                          'md_text': ea[2] if ea else '', 'docx_index': (j1 + k) if gb else '',
                          'docx_text': gb[1] if gb else ''})
    return {'label': label, 'md': md_path, 'md_sha1': sha1(md_path), 'docx': docx_path, 'docx_sha1': sha1(docx_path),
            'cut_line': cut, 'n_expected': len(exp_f), 'n_docx': len(got_f), 'n_img_docx': n_img,
            'n_fig_md': len(exp_figs), 'fig_files_md': exp_figs, 'ratio': sm.ratio(), 'n_diff_rows': len(diffs)}, diffs, exp_f, got_f

if __name__ == '__main__':
    MD3 = os.path.join(REP, 'manuscript_v3.md')
    pairs = [
        (MD3, os.path.join(REP, '보고서_초안_v7.docx'), 'v7docx_vs_md_v3(current)'),
        (os.path.join(REP, 'drafts', 'manuscript_for_v7.md'), os.path.join(REP, '보고서_초안_v7.docx'), 'v7docx_vs_for_v7(build_src)'),
        (MD3, os.path.join(REP, '보고서_초안_v6.docx'), 'v6docx_vs_md_v3(current)'),
        (os.path.join(REP, 'drafts', 'manuscript_for_v6.md'), os.path.join(REP, '보고서_초안_v6.docx'), 'v6docx_vs_for_v6(build_src)'),
        (os.path.join(REP, 'drafts', 'manuscript_for_v5.md'), os.path.join(REP, '보고서_초안_v5.docx'), 'v5docx_vs_for_v5(build_src)'),
    ]
    summ = []; alld = []
    for md, dx, lab in pairs:
        s, d, ef, gf = compare(md, dx, lab)
        summ.append(s); alld += d
        json.dump({'exp': ef, 'got': gf}, open(os.path.join(CACHE, 's1_' + lab.split('(')[0] + '.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(summ, open(os.path.join(CACHE, 's1_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    with open(os.path.join(CACHE, 's1_diffs_raw.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair', 'op', 'md_line', 'md_kind', 'md_text', 'docx_index', 'docx_text'])
        w.writeheader(); w.writerows(alld)
    for s in summ:
        print(json.dumps({k: v for k, v in s.items() if k not in ('fig_files_md',)}, ensure_ascii=False))
