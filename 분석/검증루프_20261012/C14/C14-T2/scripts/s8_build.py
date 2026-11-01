# C14-T2 s8: 원고 v15 사본에 수정안 적용 → build_test\md2docx.js(보고서 build 사본, FIG_NATIVE=1) → Word PDF(render.ps1 사본) → 쪽수·마지막 쪽 여유.
# 원고·build 폴더는 읽기만. 모든 산출은 이 작업 폴더 build_test\ 안.
import os, sys, json, subprocess, hashlib, zipfile, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T2\build_test"
MS = R + r"\manuscript_v3.md"
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("d3727aa0")
src = open(MS, encoding="utf-8").read()
E = {
 "P079z": ("결과는 외부 사건)(등록원부도", "결과는 외부 사건; 등록원부도"),
 "P079": ("모든 트랙의 모집단·단위는 제공 데이터에서 정의한다(N4 폐업·N7 과제 진입 결과는 외부 사건)(등록원부도",
          "모든 트랙은 모집단이나 결과를 제공 데이터로 정의한다(N6·N9 모집단은 NTIS, N4·N5·N7 결과는 외부 사건; 등록원부도"),
 "P143": ("각각 빼도 T1 skill 변화 0.0022 이하)", "각각 빼도 T1·T3 skill 하락 0.0022 이하)"),
}
V = {"N0": [], "Z79": ["P079z"], "F79": ["P079"], "F143": ["P143"], "FALL": ["P079", "P143"]}
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
base_b = pagebottoms(fitz.open(R + r"\보고서_초안_v15.pdf"))
xml15 = zipfile.ZipFile(R + r"\보고서_초안_v15.docx").read("word/document.xml")
for k in sys.argv[1:]:
    t = src; dch = {}
    for e in V[k]:
        o, n = E[e]; assert t.count(o) == 1, (e, t.count(o)); t = t.replace(o, n); dch[e] = len(n) - len(o)
    md = os.path.join(T, f"ms_{k}.md"); open(md, "w", encoding="utf-8", newline="\n").write(t)
    docx = os.path.join(T, f"{k}.docx")
    r1 = subprocess.run(["node", os.path.join(T, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8")
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(T, "render.ps1"), docx], capture_output=True, text=True)
    d = fitz.open(os.path.join(T, f"{k}.pdf")); b = pagebottoms(d)
    xml = zipfile.ZipFile(docx).read("word/document.xml")
    res[k] = dict(edits=V[k], d_chars=sum(dch.values()), per_edit=dch, pages=d.page_count, last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1),
                  page_bottoms=b, pages_differing_from_v15=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v15=(xml == xml15), node_err=r1.stderr[-300:], render_out=r2.stdout[-120:])
    d.close()
    print(k, res[k]["pages"], res[k]["last_bottom"], res[k]["slack_pt"], res[k]["d_chars"], res[k]["pages_differing_from_v15"], "xml_same", res[k]["docxml_same_as_v15"], r1.stderr[-200:])
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
