# C16-T4 s6(선택): 현재 원고(4cbf3d55) 사본 시험 빌드 — N0 무변경 재현 + 표 11(388행) 창 의존 단서 압축안 지면 실측
# 원고는 읽기만. 산출은 build_test\ 안. FIG_NATIVE=1, C15-T4와 같은 md2docx.js·render.ps1 사본.
import os, sys, json, subprocess, hashlib, zipfile, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C16\C16-T4\build_test"
MS = R + r"\manuscript_v3.md"
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("4cbf3d55")
src = open(MS, encoding="utf-8").read()
O = "정부 R&D 기원 특허(등록 2007–16년) 5년 내 포기 14.2%(비기원 21.0%; 출원인 유형을 맞추면 격차 1.4–3.5%p)"
E = {
 "G1": (O, "정부 R&D 기원 특허(등록 2007–16년) 5년 내 포기 14.2%(비기원 21.0%; 유형 표준화 격차 1.4–3.5%p, 10년 내 역전)"),
 "G2": (O, "정부 R&D 기원(등록 2007–16년) 5년 내 포기 14.2%(비기원 21.0%; 유형 맞춤 격차 1.4–3.5%p, 10년 내 역전)"),
 "G3": (O, "정부 R&D 기원(등록 2007–16년) 5년 내 포기 14.2%(비기원 21.0%, 10년 내는 55.5% 대 52.2%로 역전)"),
}
V = {"N0": [], "B1": ["G1"], "B2": ["G2"], "B3": ["G3"]}
env = dict(os.environ, FIG_NATIVE="1", NODE_PATH=r"C:\Users\PC\AppData\Roaming\npm\node_modules")
outp = os.path.join(T, "test_builds.json")
res = json.load(open(outp, encoding="utf-8")) if os.path.exists(outp) else {}
def pagebottoms(d):
    out = []
    for pg in d:
        ph = pg.rect.height; ys = []
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                if "".join(s["text"] for s in l["spans"]).strip() and l["bbox"][1] < ph * 0.93: ys.append(l["bbox"][3])
        out.append(round(max(ys), 1) if ys else 0)
    return out
base_pdf = fitz.open(R + r"\보고서_초안_v17.pdf"); base_b = pagebottoms(base_pdf)
z17 = zipfile.ZipFile(R + r"\보고서_초안_v17.docx"); xml17 = z17.read("word/document.xml")
med17 = sorted(hashlib.sha1(z17.read(n)).hexdigest()[:8] for n in z17.namelist() if n.startswith("word/media/") and not n.endswith("/"))
for k in sys.argv[1:]:
    t = src; dch = {}
    for e in V[k]:
        o, n = E[e]; assert t.count(o) == 1, (e, t.count(o)); t = t.replace(o, n); dch[e] = len(n) - len(o)
    md = os.path.join(T, f"ms_{k}.md"); open(md, "w", encoding="utf-8", newline="\n").write(t)
    docx = os.path.join(T, f"{k}.docx")
    r1 = subprocess.run(["node", os.path.join(T, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8")
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(T, "render.ps1"), docx], capture_output=True, text=True)
    d = fitz.open(docx.replace(".docx", ".pdf")); b = pagebottoms(d)
    z = zipfile.ZipFile(docx); xml = z.read("word/document.xml")
    med = sorted(hashlib.sha1(z.read(n)).hexdigest()[:8] for n in z.namelist() if n.startswith("word/media/") and not n.endswith("/"))
    npara = xml.count(b"<w:p>") + xml.count(b"<w:p ")
    res[k] = dict(edits=V[k], d_chars=sum(dch.values()), per_edit=dch, pages=d.page_count, last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1),
                  page_bottoms=b, pages_differing_from_v17=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v17=(xml == xml17), n_para=npara, media_same_as_v17=(med == med17), node_err=r1.stderr[-300:], render_out=r2.stdout[-80:])
    print(k, res[k]["pages"], res[k]["last_bottom"], res[k]["slack_pt"], res[k]["d_chars"], res[k]["pages_differing_from_v17"], "xml_same", res[k]["docxml_same_as_v17"], "para", npara, "media", res[k]["media_same_as_v17"], r1.stderr[-200:])
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
res["_v17_base"] = dict(pages=base_pdf.page_count, page_bottoms=base_b, last_bottom=base_b[-1], slack_pt=round(785.2 - base_b[-1], 1),
                        n_para=xml17.count(b"<w:p>") + xml17.count(b"<w:p "), media=med17)
json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("v17 base", res["_v17_base"])
