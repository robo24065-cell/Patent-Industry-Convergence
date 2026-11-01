import zipfile, re, sys, html
D = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/보고서_20261012/보고서_초안_v3.docx"
OUT = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C01/C01-T2/docx_v3_text.txt"
x = zipfile.ZipFile(D).read('word/document.xml').decode('utf-8')
paras = []
for p in re.findall(r'<w:p[ >].*?</w:p>', x, flags=re.S):
    t = ''.join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', p, flags=re.S))
    t = html.unescape(t)
    if t.strip(): paras.append(t)
open(OUT, 'w', encoding='utf-8').write('\n'.join(paras))
print(len(paras))
