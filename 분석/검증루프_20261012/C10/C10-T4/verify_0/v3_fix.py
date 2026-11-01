import sys, os, fitz, zipfile, hashlib
from PIL import Image
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
V = os.path.join(B, r"..\verify_0")
a = Image.open(os.path.join(B, "fig01_framework.png")); b = Image.open(os.path.join(R:=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012", r"figs\fig01_framework.png"))
print(a.size, b.size, a.info.get("dpi"), b.info.get("dpi"))
from PIL import ImageChops
diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox(); print("diff bbox", diff)
a.crop((diff[0]-300, diff[1]-30, diff[2]+30, diff[3]+30)).save(os.path.join(V, "v3_fix_crop.png"))
d = fitz.open(os.path.join(B, "N5.pdf")); print("N5 pages", d.page_count)
z = zipfile.ZipFile(os.path.join(B, "N5.docx")); print([n for n in z.namelist() if n.startswith("word/media/") and hashlib.sha1(z.read(n)).hexdigest().startswith("2db9d37f")])
