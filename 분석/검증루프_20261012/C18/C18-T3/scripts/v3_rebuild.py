# C18-T3 verify 3: 독립 재빌드(PREREG 8절). s5_build.py를 쓰지 않는다. verify_build\ 에 도구를 새로 복사하고,
# 작업 폴더의 산출 원고(manuscript_v19_candidate/min/option.md)와 v19 원고(N0)를 그대로 빌드한다.
# 측정은 다른 방식: document.xml 전체 sha1, 정규식 <w:p[ >] 수, media 전체 sha1, PDF 쪽수(fitz)·Word 쪽수, 마지막 쪽 바닥은 get_text('words') 기준.
import os, sys, re, json, shutil, hashlib, zipfile, subprocess
import fitz
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; W = AN + r"\검증루프_20261012\C18\C18-T3"; VB = W + r"\verify_build"
os.makedirs(VB, exist_ok=True)
for f in ["md2docx.js", "render.ps1"]:
    shutil.copyfile(os.path.join(R, "build", f), os.path.join(VB, f))
    assert open(os.path.join(VB, f), "rb").read() == open(os.path.join(R, "build", f), "rb").read()
SRC = {"N0": R + r"\manuscript_v3.md", "C": W + r"\manuscript_v19_candidate.md", "M": W + r"\manuscript_v19_min.md", "O": W + r"\manuscript_v19_option.md"}
env = dict(os.environ, FIG_NATIVE="1", NODE_PATH=r"C:\Users\PC\AppData\Roaming\npm\node_modules")
def docx_info(p):
    z = zipfile.ZipFile(p); x = z.read("word/document.xml")
    med = {n: hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/") and n.endswith(".png")}
    z.close()
    return x, len(re.findall(rb"<w:p[ >]", x)), med
def pdf_info(p):
    d = fitz.open(p); bots = []
    for pg in d:
        ys = [w[3] for w in pg.get_text("words") if w[4].strip() and w[1] < pg.rect.height * 0.93]
        bots.append(round(max(ys), 1) if ys else 0.0)
    n = d.page_count; d.close(); return n, bots
x19, np19, med19 = docx_info(R + r"\보고서_초안_v19.docx"); n19, b19 = pdf_info(R + r"\보고서_초안_v19.pdf")
out = {"_v19": dict(docxml_sha1=hashlib.sha1(x19).hexdigest(), n_para=np19, n_png=len(med19), pdf_pages=n19, last_bottom=b19[-1], p5=b19[4], p6=b19[5])}
for k in (sys.argv[1:] or list(SRC)):
    md = os.path.join(VB, f"ms_{k}.md"); shutil.copyfile(SRC[k], md)
    docx = os.path.join(VB, f"{k}.docx")
    r1 = subprocess.run(["node", os.path.join(VB, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(VB, "render.ps1"), docx], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    m = re.search(r"PAGES:\s*(\d+)", r2.stdout or "")
    x, npara, med = docx_info(docx); n, b = pdf_info(docx[:-5] + ".pdf")
    out[k] = dict(md_sha1=hashlib.sha1(open(md, "rb").read()).hexdigest(), docxml_sha1=hashlib.sha1(x).hexdigest(), docxml_same_as_v19=(x == x19),
                  n_para=npara, n_png=len(med), png_same_as_v19=(sorted(med.values()) == sorted(med19.values())), pdf_pages=n, word_pages=int(m.group(1)) if m else None,
                  last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1), p5=b[4], p6=b[5], page_bottoms=b, node_err=(r1.stderr or "")[-200:])
    print(k, {kk: vv for kk, vv in out[k].items() if kk != "page_bottoms"}, flush=True)
prev = W + r"\verify\v3_rebuild_out.json"
old = json.load(open(prev, encoding="utf-8")) if os.path.exists(prev) else {}
old.update(out)
json.dump(old, open(prev, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
