# C12-T2 s5: 변형 빌드(node md2docx.js 사본, FIG_NATIVE=1) → Word PDF(render.ps1 사본) → 쪽수·마지막 쪽 하단
import os, sys, json, subprocess, hashlib, zipfile, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T2\build_test"
meta = json.load(open(os.path.join(T, "variants.json"), encoding="utf-8"))["variants"]
env = dict(os.environ, FIG_NATIVE="1", NODE_PATH=r"C:\Users\PC\AppData\Roaming\npm\node_modules")
outp = os.path.join(T, "test_builds.json")
res = json.load(open(outp, encoding="utf-8")) if os.path.exists(outp) else {}
v13 = fitz.open(R + r"\보고서_초안_v13.pdf")
def pagebottoms(d):
    out = []
    for pg in d:
        ph = pg.rect.height; ys = []
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                if "".join(s["text"] for s in l["spans"]).strip() and l["bbox"][1] < ph * 0.93: ys.append(l["bbox"][3])
        out.append(round(max(ys), 1) if ys else 0)
    return out
base_b = pagebottoms(v13)
for k in sys.argv[1:]:
    m = meta[k]; fd = R + r"\figs" if m["figs"] == "figs" else os.path.join(T, m["figs"])
    docx = os.path.join(T, f"{k}.docx")
    r1 = subprocess.run(["node", os.path.join(T, "md2docx.js"), os.path.join(T, f"ms_{k}.md"), fd, docx], env=env, capture_output=True, text=True, encoding="utf-8")
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(T, "render.ps1"), docx], capture_output=True, text=True)
    d = fitz.open(os.path.join(T, f"{k}.pdf")); b = pagebottoms(d)
    xml = zipfile.ZipFile(docx).read("word/document.xml")
    xml13 = zipfile.ZipFile(R + r"\보고서_초안_v13.docx").read("word/document.xml")
    res[k] = dict(m, pages=d.page_count, last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1),
                  pages_differing_from_v13=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v13=(xml == xml13), docx_sha1=hashlib.sha1(open(docx, "rb").read()).hexdigest()[:12],
                  node_err=r1.stderr[-300:], render_out=r2.stdout[-200:])
    print(k, res[k]["pages"], res[k]["last_bottom"], res[k]["slack_pt"], res[k]["pages_differing_from_v13"], "xml_same", res[k]["docxml_same_as_v13"], r1.stderr[-200:])
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
