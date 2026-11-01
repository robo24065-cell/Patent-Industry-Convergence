# verify_0 2차 반박 시도: docx 문서 순서상 그림 1 확인, PDF 4쪽 이미지 추출·해시/픽셀 비교
import sys, os, zipfile, hashlib, re, json
sys.stdout.reconfigure(encoding="utf-8")
A = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = os.path.join(A, "보고서_20261012"); V = os.path.join(A, r"검증루프_20261012\C10\C10-T4\verify_0")
out = {}
png = open(os.path.join(R, r"figs\fig01_framework.png"), "rb").read()
out["figs_fig01_sha1"] = hashlib.sha1(png).hexdigest()
for v in ("v10", "v11"):
    z = zipfile.ZipFile(os.path.join(R, f"보고서_초안_{v}.docx"))
    doc = z.read("word/document.xml").decode("utf-8")
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
    rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels)) | {a: b for b, a in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)}
    order = re.findall(r'r:embed="(rId\d+)"', doc)
    first = rid2t.get(order[0]); data = z.read("word/" + first)
    out[v] = {"first_image": first, "sha1": hashlib.sha1(data).hexdigest(), "n_images": len(order)}
try:
    import fitz
    d = fitz.open(os.path.join(R, "보고서_초안_v11.pdf")); out["pdf_pages"] = d.page_count
    found = []
    for pno in range(d.page_count):
        for img in d[pno].get_images(full=True):
            pix = fitz.Pixmap(d, img[0])
            if pix.width > 1500:
                found.append((pno + 1, img[0], pix.width, pix.height))
            if pno == 3 and pix.width > 1000:
                pix2 = fitz.Pixmap(fitz.csRGB, pix) if pix.n - pix.alpha > 3 or pix.alpha else pix
                pix2.save(os.path.join(V, "v1_pdf_p4_img.png"))
    out["pdf_large_images"] = found[:20]
    out["pdf_p4_text_head"] = d[3].get_text()[:300]
except Exception as e:
    out["fitz_err"] = repr(e)
from PIL import Image
im = Image.open(os.path.join(R, r"figs\fig01_framework.png")); out["fig_size"] = im.size; out["fig_dpi"] = im.info.get("dpi")
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(V, "v1_recheck_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
