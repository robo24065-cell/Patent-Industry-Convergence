import re, csv, difflib, sys
sys.stdout.reconfigure(encoding='utf-8')
B = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/"
md = open(B+"보고서_20261012/manuscript_v2_from_response.md", encoding='utf-8').read().split('\n')[:450]
dx = open(B+"검증루프_20261012/C01/C01-T2/docx_v3_text.txt", encoding='utf-8').read().split('\n')
NUM = re.compile(r'[−\-+]?\d[\d,]*(?:\.\d+)?')
def norm(s):
    s = re.sub(r'\*\*|__|`', '', s); s = s.replace('\u2212','-').replace('–','-').replace('~','-')
    return re.sub(r'\s+', ' ', s).strip()
def nums(s): return [n.replace(',','').lstrip('+') for n in NUM.findall(norm(s))]
# md units: paragraphs; table rows -> cells joined
mdu = []
for i, l in enumerate(md, 1):
    t = l.strip()
    if not t or t.startswith('---') or re.match(r'^\|[-| ]+\|$', t): continue
    if t.startswith('|'):
        cells = [c.strip() for c in t.strip('|').split('|')]
        mdu.append((i, ' '.join(cells), 'table'))
    else:
        mdu.append((i, re.sub(r'^#+\s*|^[-*]\s+|^\d+\.\s+', '', t), 'para'))
dxn = [norm(d) for d in dx]
used = set(); rows = []
# docx tables: cells are separate paragraphs; build sliding concatenations of up to 8 consecutive paragraphs
for (i, t, kind) in mdu:
    tn = norm(t)
    best, bj, bspan = 0, None, 1
    for j in range(len(dxn)):
        for span in ((1,) if kind=='para' else range(1, 9)):
            cand = ' '.join(dxn[j:j+span])
            r = difflib.SequenceMatcher(None, tn, cand, autojunk=False).quick_ratio()
            if r > best - 0.0001 and r >= best:
                if r > best: best, bj, bspan = r, j, span
    if bj is not None:
        cand = ' '.join(dxn[bj:bj+bspan]); best = difflib.SequenceMatcher(None, tn, cand, autojunk=False).ratio()
    a = nums(t); b = nums(' '.join(dxn[bj:bj+bspan])) if bj is not None else []
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    only_md = list((ca - cb).elements()); only_dx = list((cb - ca).elements())
    if bj is not None: used.update(range(bj, bj+bspan))
    rows.append(dict(md_line=i, kind=kind, match_ratio=round(best,3), docx_para=(bj+1 if bj is not None else ''), n_md=len(a), n_docx=len(b),
                     only_md=' '.join(only_md), only_docx=' '.join(only_dx), md_text=t[:160], docx_text=(' '.join(dx[bj:bj+bspan])[:160] if bj is not None else '')))
out = B+"검증루프_20261012/C01/C01-T2/md_vs_docx_all.csv"
with open(out,'w',encoding='utf-8-sig',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
unused = [(j+1, dx[j]) for j in range(len(dx)) if j not in used]
with open(B+"검증루프_20261012/C01/C01-T2/docx_unmatched_paras.txt",'w',encoding='utf-8') as f:
    for j,t in unused: f.write(f"{j}\t{t}\n")
bad = [r for r in rows if r['only_md'] or r['only_docx'] or r['match_ratio']<0.9]
print('md units', len(rows), 'with diff', len(bad), 'docx unmatched paras', len(unused))
for r in bad: print(r['md_line'], r['kind'], r['match_ratio'], '| md-only:', r['only_md'], '| docx-only:', r['only_docx'], '|', r['md_text'][:70])
