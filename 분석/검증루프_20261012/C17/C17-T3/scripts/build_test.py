# C17-T3 s4: v18 원고 사본 시험 빌드(PREREG 6절). C16-T4 s6_build.py 절차 그대로(FIG_NATIVE=1, md2docx.js·render.ps1 사본), 기준만 v18로 바꿈.
# 원고·빌드 도구는 읽기만(사본은 build_test\ 안). 사용: python -I build_test.py N0 P1 P2
import os, sys, json, subprocess, hashlib, zipfile, shutil, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C17\C17-T3\build_test"
MS = R + r"\manuscript_v3.md"
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("39452163")
for f in ["md2docx.js", "render.ps1"]:
    if not os.path.exists(os.path.join(T, f)): shutil.copy2(os.path.join(R, "build", f), os.path.join(T, f))
if not os.path.exists(os.path.join(T, "manuscript_v18_copy.md")): shutil.copy2(MS, os.path.join(T, "manuscript_v18_copy.md"))
assert open(os.path.join(T, "md2docx.js"), "rb").read() == open(os.path.join(R, "build", "md2docx.js"), "rb").read()
src = open(MS, encoding="utf-8").read()
E = {
 "E1": ("·T1 위험쌍 정의, 2.5)", "·T1 위험쌍·T3 신설 IPC 셀 정의, 2.5·4.4)"),
 "E2": ("입력에 미래 정보는 쓰지 않는다.", "입력에 미래 정보는 쓰지 않는다(예외 1.4)."),
 "E2a": ("원칙은 세 가지다. 입력에 미래 정보는 쓰지 않는다.", "원칙은 셋이다. 입력에 미래 정보는 쓰지 않는다(예외 1.4)."),
 "E2f": ("원칙은 세 가지다. 입력에 미래 정보는 쓰지 않는다.", "원칙은 셋이다. 입력에 미래 정보는 쓰지 않는다(1.4)."),
 "E2o": ("입력에 미래 정보는 쓰지 않는다.", "입력에 미래 정보는 원칙적으로 쓰지 않는다."),
 "E4c": ("2026년 스냅샷은 법인 연결표를 빼면 시점을 맞추거나(T1 권리목록은 넘긴 권리가 빠짐, 영향 0.007 이하) 해석층으로만 썼으며, 국민연금 사업장은 상호 변경 키를 별칭 보정했다.", "2026년 스냅샷은 법인 연결표·DART 업종을 빼면 시점을 맞추거나(T1 권리목록은 넘긴 권리가 빠짐, 영향 0.007 이하) 해석층으로만 썼으며, 국민연금은 상호 변경 키를 별칭 보정했다."),
 "E3m": ("(빼거나 합쳐도 skill 0.39–0.42)", "(빼거나 합쳐도 skill 0.38–0.42)"),
 "E3r": ("(빼거나 합쳐도 skill 0.39–0.42)", "(신설 분류 셀을 빼거나 합쳐도 skill 0.38–0.42)"),
 "E4": ("법인 연결표를 빼면", "법인 연결표·DART 업종을 빼면"),
 "C04": ("출원인–법인 연결은 이후 활동한 기업 쪽으로 기운다(연결률", "출원인–법인 연결은 이후 활동한 기업 쪽으로 기운다(N6 미연결 기업은 결과 0; 연결률"),
 "C06": ("물가·산업연관표·생산지수는", "물가·산업연관표·광공업지수는"),
 "C3S": ("| KR 출원의 9.34%, 등록의 10.95% |", "| KR 출원(실용신안 포함)의 9.34%, 등록의 10.95% |"),
}
V = {"N0": [], "P1": ["E1", "E2f", "E3m"], "P2": ["E1", "E2f", "E3r", "E4c", "C04", "C06", "C3S"], "P2a": ["E1", "E2f", "E3r", "E4", "C04", "C06", "C3S"], "P1x": ["E1", "E2", "E3m"], "P2x": ["E1", "E2", "E3r", "E4", "C04", "C06", "C3S"]}
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
base_pdf = fitz.open(R + r"\보고서_초안_v18.pdf"); base_b = pagebottoms(base_pdf)
z18 = zipfile.ZipFile(R + r"\보고서_초안_v18.docx"); xml18 = z18.read("word/document.xml")
med18 = sorted(hashlib.sha1(z18.read(n)).hexdigest()[:8] for n in z18.namelist() if n.startswith("word/media/") and not n.endswith("/"))
res["_v18_base"] = dict(pages=base_pdf.page_count, page_bottoms=base_b, last_bottom=base_b[-1], slack_pt=round(785.2 - base_b[-1], 1),
                        n_para=xml18.count(b"<w:p>") + xml18.count(b"<w:p "), media=med18, n_media=len(med18))
for k in sys.argv[1:]:
    if "=" in k: k, spec = k.split("=", 1); V[k] = spec.split("+")
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
                  page_bottoms=b, pages_differing_from_v18=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v18=(xml == xml18), n_para=npara, media_same_as_v18=(med == med18), n_media=len(med), media=med,
                  node_err=r1.stderr[-300:], render_out=r2.stdout[-80:])
    print(k, res[k]["pages"], res[k]["last_bottom"], res[k]["slack_pt"], res[k]["d_chars"], dch, res[k]["pages_differing_from_v18"], "xml_same", res[k]["docxml_same_as_v18"], "para", npara, "media", res[k]["media_same_as_v18"], len(med), r1.stderr[-200:], flush=True)
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("v18 base", {k: v for k, v in res["_v18_base"].items() if k != 'page_bottoms'})
