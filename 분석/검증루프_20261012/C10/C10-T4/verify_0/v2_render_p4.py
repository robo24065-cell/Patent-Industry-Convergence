import sys, os, fitz
A = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
V = os.path.join(A, r"검증루프_20261012\C10\C10-T4\verify_0")
d = fitz.open(os.path.join(A, r"보고서_20261012\보고서_초안_v11.pdf"))
p = d[3]
print([ (i[0], i[2], i[3]) for i in p.get_images(full=True)])
for b in p.get_image_info(): print(b["bbox"], b.get("width"), b.get("height"))
p.get_pixmap(dpi=110).save(os.path.join(V, "v2_pdf_p4.png"))
# zoom into figure region
bb = p.get_image_info()[0]["bbox"] if p.get_image_info() else p.rect
r = fitz.Rect(bb); clip = fitz.Rect(r.x0 + r.width*0.3, r.y0 + r.height*0.6, r.x0 + r.width*0.7, r.y1)
p.get_pixmap(dpi=300, clip=clip).save(os.path.join(V, "v2_pdf_p4_control.png"))
