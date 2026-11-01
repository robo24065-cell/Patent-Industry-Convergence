import sys, hashlib
from PIL import Image, ImageChops
sys.stdout.reconfigure(encoding="utf-8")
a = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\figs\fig01_framework.png"
b = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test\fig01_framework.png"
A, B = Image.open(a), Image.open(b)
print(A.size, B.size, A.info.get("dpi"), B.info.get("dpi"))
print("diff bbox", ImageChops.difference(A.convert("RGB"), B.convert("RGB")).getbbox())
print("new sha1", hashlib.sha1(open(b, "rb").read()).hexdigest())
w, h = B.size
B.crop((int(w * 0.34), int(h * 0.88), int(w * 0.66), int(h * 0.97))).save(b.replace(".png", "_crop.png"))
