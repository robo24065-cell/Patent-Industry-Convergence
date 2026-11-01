# PDF 9쪽 내장 그림을 꺼내 figs\fig04(현행)·_v3_backup\fig04_before_C07와 같은 크기로 줄여 평균 절대 픽셀 차 비교
import sys, fitz, numpy as np, io, json
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
pdf, cur, old, out = sys.argv[1:5]
d = fitz.open(pdf); xref = d[8].get_images(full=True)[0][0]
pix = fitz.Pixmap(d, xref)
if pix.n - pix.alpha >= 4: pix = fitz.Pixmap(fitz.csRGB, pix)
a = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
res = {"pdf_img": [a.width, a.height]}
A = np.asarray(a).astype(float)
for nm, p in [("current", cur), ("before_C07", old)]:
    b = Image.open(p).convert("L").resize(a.size, Image.LANCZOS)
    res[nm + "_MAE"] = round(float(np.abs(A - np.asarray(b).astype(float)).mean()), 3)
print(res); json.dump(res, open(out, "w"), indent=1)
