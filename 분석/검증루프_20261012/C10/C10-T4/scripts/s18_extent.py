import sys, zipfile, re
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\보고서_초안_v11.docx"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test\V0.docx"
xa = zipfile.ZipFile(R).read("word/document.xml").decode(); xb = zipfile.ZipFile(T).read("word/document.xml").decode()
print(re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"', xa)); print(re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"', xb))
pa = re.findall(r'<w:p[ >].*?</w:p>', xa, re.S); pb = re.findall(r'<w:p[ >].*?</w:p>', xb, re.S)
print(len(pa), len(pb)); d = [i for i, (a, b) in enumerate(zip(pa, pb)) if a != b]
print(len(d), d[:5])
if d:
    a, b = pa[d[0]], pb[d[0]]
    k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i]); print(a[max(0, k-80):k+80]); print(b[max(0, k-80):k+80])
