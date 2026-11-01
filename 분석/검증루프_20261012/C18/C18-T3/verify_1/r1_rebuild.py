# C18-T3 verify_1: 반박용 독립 재빌드. s5_build.py / v3_rebuild.py 미사용.
# 원고(v3)와 option 원고는 읽기만. 단일 수정판(O6, O7)은 v3에 직접 치환해 새로 만든다.
# 측정: PDF 쪽수(fitz), Word 쪽수, 쪽별 바닥(문자 단위 rawdict의 char bbox 최대 y), 쪽별 첫 줄 텍스트, 쪽별 전체 텍스트 해시.
import os, sys, re, json, shutil, hashlib, zipfile, subprocess
import fitz
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; W = AN + r"\검증루프_20261012\C18\C18-T3"; V = W + r"\verify_1"
for f in ["md2docx.js", "render.ps1"]:
    shutil.copyfile(os.path.join(R, "build", f), os.path.join(V, f))
v3 = open(R + r"\manuscript_v3.md", encoding="utf-8").read()
assert hashlib.sha1(v3.encode("utf-8")).hexdigest().startswith("b316df8e")
opt = open(W + r"\manuscript_v19_option.md", encoding="utf-8").read()
def rep(t, pairs):
    for o, n in pairs:
        assert t.count(o) == 1, o
        t = t.replace(o, n)
    return t
SRCS = {
 "N0": v3,
 "O": opt,
 "S6": rep(v3, [("| KR 출원의 9.34%, 등록의 10.95% |", "| KR 출원(실용신안 포함)의 9.34%, 등록의 10.95% |")]),
 "S7": rep(v3, [("0에 닿는 T4(−0.003)", "0 밑인 T4(−0.003)"), ("CI 하한이 0에 닿고(−0.003)", "CI 하한이 0 밑이고(−0.003)")]),
}
env = dict(os.environ, FIG_NATIVE="1", NODE_PATH=r"C:\Users\PC\AppData\Roaming\npm\node_modules")
def pdfinfo(p):
    d = fitz.open(p); out = []
    for pg in d:
        H = pg.rect.height; ymax = 0.0; lines = []
        for b in pg.get_text("rawdict")["blocks"]:
            for l in b.get("lines", []):
                chars = [c for s in l["spans"] for c in s["chars"] if c["c"].strip()]
                if not chars: continue
                y1 = max(c["bbox"][3] for c in chars); y0 = min(c["bbox"][1] for c in chars)
                if y0 >= H * 0.93: continue  # 꼬리말 제외
                ymax = max(ymax, y1)
                lines.append((round(y0, 1), "".join(c["c"] for c in l["spans"][0]["chars"])[:30]))
        lines.sort()
        txt = pg.get_text("text")
        out.append(dict(bottom=round(ymax, 1), first=lines[0][1] if lines else "", n_lines=len(lines), txt_sha=hashlib.sha1(txt.encode()).hexdigest()[:8]))
    n = d.page_count; d.close(); return n, out
res = {}
d19 = R + r"\보고서_초안_v19.pdf"; n19, i19 = pdfinfo(d19)
z = zipfile.ZipFile(R + r"\보고서_초안_v19.docx"); x19 = z.read("word/document.xml"); z.close()
res["v19"] = dict(pages=n19, info=i19, docxml=hashlib.sha1(x19).hexdigest()[:8])
for k in (sys.argv[1:] or list(SRCS)):
    md = os.path.join(V, f"r_{k}.md"); open(md, "w", encoding="utf-8", newline="\n").write(SRCS[k])
    docx = os.path.join(V, f"r_{k}.docx")
    r1 = subprocess.run(["node", os.path.join(V, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(V, "render.ps1"), docx], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    m = re.search(r"PAGES:\s*(\d+)", r2.stdout or "")
    z = zipfile.ZipFile(docx); x = z.read("word/document.xml")
    png = sorted(hashlib.sha1(z.read(nm)).hexdigest()[:8] for nm in z.namelist() if nm.startswith("word/media/") and not nm.endswith("/")); z.close()
    n, info = pdfinfo(docx[:-5] + ".pdf")
    res[k] = dict(md_sha1=hashlib.sha1(SRCS[k].encode("utf-8")).hexdigest()[:8], docxml=hashlib.sha1(x).hexdigest()[:8], n_para=len(re.findall(rb"<w:p[ >]", x)),
                  png=png, pages=n, word_pages=int(m.group(1)) if m else None, info=info, node_err=(r1.stderr or "")[-200:], r2=(r2.stdout or "")[-80:] + (r2.stderr or "")[-200:])
    print(k, res[k]["md_sha1"], res[k]["docxml"], res[k]["n_para"], len(png), n, res[k]["word_pages"], [p["bottom"] for p in info], flush=True)
    json.dump(res, open(V + r"\r1_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(res, open(V + r"\r1_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
