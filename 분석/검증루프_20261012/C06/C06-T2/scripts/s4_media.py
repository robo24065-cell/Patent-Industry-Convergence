# C06-T2 s4: docx 내장 그림과 figs PNG 해시 대조, pdf 그림 수
import zipfile, hashlib, glob, os, sys, json, re
import fitz
sys.stdout.reconfigure(encoding='utf-8')
REP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T2"
h = lambda b: hashlib.sha1(b).hexdigest()
figs = {h(open(f, 'rb').read()): os.path.basename(f) for f in glob.glob(os.path.join(REP, 'figs', '*.png'))}
old = {}
for f in glob.glob(os.path.join(REP, 'figs', '_*', '*.png')):
    old[h(open(f, 'rb').read())] = os.path.relpath(f, os.path.join(REP, 'figs'))
res = {}
for v in ['v5', 'v6', 'v7']:
    z = zipfile.ZipFile(os.path.join(REP, f'보고서_초안_{v}.docx'))
    doc = z.read('word/document.xml').decode('utf-8')
    rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
    rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels)) | {a: b for b, a in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)}
    order = re.findall(r'r:embed="(rId\d+)"', doc)
    seq = []
    for rid in order:
        t = rid2t.get(rid, ''); b = z.read('word/' + t)
        hh = h(b); seq.append({'rid': rid, 'target': t, 'current_figs': figs.get(hh), 'old_version': old.get(hh)})
    pdf = fitz.open(os.path.join(REP, f'보고서_초안_{v}.pdf'))
    res[v] = {'images_in_order': seq, 'pdf_images_per_page': [len(p.get_images()) for p in pdf]}
json.dump(res, open(os.path.join(OUT, 'cache', 's4_media.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for v, r in res.items():
    print(v, [(x['current_figs'] or ('OLD:' + str(x['old_version']))) for x in r['images_in_order']], 'pdf imgs', sum(r['pdf_images_per_page']))
