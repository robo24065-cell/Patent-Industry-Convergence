# 제출본 형식 읽기 전용 확인: docx 문단 수·media 해시·document.xml 비교, PDF 쪽수
import zipfile, hashlib, re, json, sys, os
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = BASE + r"\보고서_20261012"
BT = BASE + r"\검증루프_20261012\C15\C15-T4\build_test"
OUT = BASE + r"\검증루프_20261012\C16\C16-T4"
def dx(p):
    z = zipfile.ZipFile(p)
    doc = z.read('word/document.xml')
    paras = re.findall(rb'<w:p[ >].*?</w:p>', doc, flags=re.S)
    texts = [re.sub(rb'<[^>]+>', b'', x).decode('utf-8') for x in paras]
    media = {n: hashlib.sha1(z.read(n)).hexdigest()[:8] for n in z.namelist() if n.startswith('word/media/')}
    return dict(docxml_sha1=hashlib.sha1(doc).hexdigest()[:8], n_para=len(paras), n_para_nonempty=sum(1 for t in texts if t.strip()), media=media, texts=texts)
def pdfpages(p):
    try:
        import fitz
        d = fitz.open(p); n = d.page_count; d.close(); return n
    except Exception as e:
        b = open(p,'rb').read()
        m = re.findall(rb'/Type\s*/Pages\b[^>]*?/Count\s+(\d+)', b)
        return int(max(m, key=int)) if m else ('err '+str(e))
res = {}
files = {'v15': R+r"\보고서_초안_v15.docx", 'v16': R+r"\보고서_초안_v16.docx", 'v17': R+r"\보고서_초안_v17.docx",
         'test_v17': R+r"\drafts\test_v17.docx", 'C15T4_rec': BT+r"\F1b1_F1b2_F2z_F3_F4a_F4b_F4c_F5b_F6_F7.docx", 'C15T4_N0': BT+r"\N0.docx"}
D = {k: dx(v) for k, v in files.items()}
for k, v in files.items():
    pdf = v[:-5]+'.pdf'
    res[k] = dict(file=os.path.basename(v), docxml_sha1=D[k]['docxml_sha1'], n_para=D[k]['n_para'], n_para_nonempty=D[k]['n_para_nonempty'],
                  media=D[k]['media'], pdf_pages=pdfpages(pdf) if os.path.exists(pdf) else None)
# figs referenced
import glob
figdir = R + r"\figs"
figs = {}
for p in sorted(glob.glob(figdir + r"\*.png")):
    figs[os.path.basename(p)] = hashlib.sha1(open(p,'rb').read()).hexdigest()[:8]
res['figs_dir_png'] = figs
ms = open(R+r"\manuscript_v3.md", encoding='utf-8').read()
cut = re.search(r'^# \[(부록|편집 메모)', ms, flags=re.M).start()
body = ms[:cut]
res['ms_img_refs'] = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', body)
# text diff v17 vs C15T4_rec, v17 vs test_v17
import difflib
def tdiff(a, b):
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, D[a]['texts'], D[b]['texts'], autojunk=False).get_opcodes():
        if tag != 'equal':
            out.append(dict(tag=tag, a=D[a]['texts'][i1:i2], b=D[b]['texts'][j1:j2]))
    return out
res['diff_C15T4rec_vs_v17'] = tdiff('C15T4_rec', 'v17')
res['diff_testv17_vs_v17'] = tdiff('test_v17', 'v17')
res['diff_v16_vs_C15T4N0'] = tdiff('v16', 'C15T4_N0')
json.dump(res, open(OUT+r"\format_check.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k in files: print(k, res[k]['docxml_sha1'], res[k]['n_para'], res[k]['n_para_nonempty'], res[k]['pdf_pages'], sorted(res[k]['media'].values()))
print('figs', figs); print('refs', res['ms_img_refs'])
for k in ['diff_C15T4rec_vs_v17','diff_testv17_vs_v17','diff_v16_vs_C15T4N0']:
    print(k, json.dumps(res[k], ensure_ascii=False)[:800])
