import sys, fitz, hashlib, zipfile
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
a = fitz.open(R + r"\보고서_초안_v11.pdf"); b = fitz.open(T + r"\V0.pdf")
for i in range(20):
    ta = a[i].get_text().strip().splitlines(); tb = b[i].get_text().strip().splitlines()
    print(i + 1, len(ta), len(tb), "| v11 end:", ta[-2][:30] if len(ta) > 1 else "", "| V0 end:", tb[-2][:30] if len(tb) > 1 else "", "| same_text:", ta == tb)
za = zipfile.ZipFile(R + r"\보고서_초안_v11.docx"); zb = zipfile.ZipFile(T + r"\V0.docx")
print("document.xml same:", za.read("word/document.xml") == zb.read("word/document.xml"))
print(len(za.read("word/document.xml")), len(zb.read("word/document.xml")))
