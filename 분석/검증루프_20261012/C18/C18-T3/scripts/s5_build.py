# C18-T3 s5: 시험 빌드(PREREG 7절, build_plan.txt). C17-T3 build_test.py 절차(FIG_NATIVE=1, md2docx.js·render.ps1 사본) 그대로, 기준만 v19.
# 원고·빌드 도구·docx 원본은 읽기만(사본은 build_test\ 안). 사용: python -I s5_build.py N0 C M S_O1 ... O
import os, sys, json, subprocess, hashlib, zipfile, shutil, re
import fitz
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; W = AN + r"\검증루프_20261012\C18\C18-T3"
T = os.environ.get("C18_BUILD_DIR", W + r"\build_test")
os.makedirs(T, exist_ok=True)
MS = R + r"\manuscript_v3.md"
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("b316df8e")
for f in ["md2docx.js", "render.ps1"]:
    if not os.path.exists(os.path.join(T, f)): shutil.copy2(os.path.join(R, "build", f), os.path.join(T, f))
    assert open(os.path.join(T, f), "rb").read() == open(os.path.join(R, "build", f), "rb").read(), f
src = open(MS, encoding="utf-8").read()
SRC_ADD = {
 "C-LAND-free_F": "; (분석)\\검증루프_20261012\\C11\\C11-T2\\verify_5\\v5_out.json(range_168), v5_frac.json",
 "C-AGREE-US": "; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s2_agree.csv; (분석)\\검증루프_20261012\\C14\\C14-T2\\pkr_ipc_altA.csv",
 "C-AGREE-JP": "; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s2_agree.csv; (분석)\\검증루프_20261012\\C14\\C14-T2\\pkr_ipc_altA.csv",
 "C-AGREE-EP": "; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s2_agree.csv; (분석)\\검증루프_20261012\\C14\\C14-T2\\pkr_ipc_altA.csv",
 "C-T3-AUCS": "; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s6_t3_summary.json; (분석)\\검증루프_20261012\\C15\\C15-T3\\res\\s5_t3_summary.json",
 "C-T3-RET": "; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s6_t3_summary.json; (분석)\\검증루프_20261012\\C15\\C15-T3\\res\\s5_t3_summary.json",
 "NTIS-T3-SUM": "; (분석)\\검증루프_20261012\\C15\\C15-T1\\cache\\s4_horizon.csv(regdate)",
 "DATA-KR-REG": "; 수집파이프라인\\registry_full.sqlite(rgst_no 10/20 집계, 읽기 전용)",
 "DATA-NTIS-PJ": "; (분석)\\검증루프_20261012\\C17\\C17-T3\\cache\\ntis_count_check.csv(원문 대조 0.99993–1.00000)",
 "LINK-KOSIS5": "; (분석)\\검증루프_20261012\\C07\\C07-T1\\official_compare.csv",
}
def appx_src(t, rid):
    """부록 표에서 id 행의 마지막 칸(출처) 끝에 덧붙임"""
    lines = t.split("\n"); hit = [k for k, s in enumerate(lines) if s.startswith("| " + rid + " |")]
    assert len(hit) == 1, (rid, len(hit))
    s = lines[hit[0]]; assert s.endswith(" |")
    lines[hit[0]] = s[:-2] + SRC_ADD[rid] + " |"
    return "\n".join(lines), len(SRC_ADD[rid])
E = {
 "O1": [("(빼거나 합쳐도 skill 0.38–0.42)", "(신설 분류 셀을 빼거나 합쳐도 skill 0.38–0.42)")],
 "O2": [("물가·산업연관표·생산지수는", "물가·산업연관표·광공업지수는")],
 "O3": [("법인 연결표를 빼면", "법인 연결표·DART 업종을 빼면"), ("국민연금 사업장은 상호 변경 키를", "국민연금은 상호 변경 키를")],
 "O4": [("기운다(연결률", "기운다(N6 미연결 기업은 결과 0; 연결률")],
 "O5": [("T1 위험쌍 조건도", "T1 법인·위험쌍 조건도")],
 "O6": [("| KR 출원의 9.34%, 등록의 10.95% |", "| KR 출원(실용신안 포함)의 9.34%, 등록의 10.95% |")],
 "O7": [("0에 닿는 T4(−0.003)", "0 밑인 T4(−0.003)"), ("CI 하한이 0에 닿고(−0.003)", "CI 하한이 0 밑이고(−0.003)")],
 "O8": [("약 46–48%", "약 47–48%")],
 "O9": [("권리범위확인심판 예고등록이다", "권리범위확인심판 예고등록(직권)이다")],
 "A1": [("| C-LAND-prov | 표지·요약, 4.4 |", "| C-LAND-prov | 표지·요약, 3.1.1, 4.1 |")],
 "A2": [("| CIRC-AGREE-CC | 표지·요약, 1.3, 3.6.1, 3.6.4 |", "| CIRC-AGREE-CC | 표지·요약, 1.3 |")],
 "A3": "SRC:C-LAND-free_F,C-AGREE-US,C-AGREE-JP,C-AGREE-EP,C-T3-AUCS,C-T3-RET,NTIS-T3-SUM,DATA-KR-REG",
 "A4": [("10년 내(등록 2007–15) 55.5% 대 52.2%(6년차 기한 집중, C15-T1)", "10년 내(등록 2007–15) 55.5% 대 52.2%(6년차 기한 집중, C15-T1); 출원인 유형·등록연도·KSIC 표준화 시 격차 1.4–3.5%p")],
 "A5": [("소멸등록 349,507, 회복등록", "소멸등록 349,507(실용신안 포함; 특허 불납 293,169·만료 20,112), 회복등록")],
 "A6": "SRC:DATA-NTIS-PJ",
 "A7": [("T3 공식 skill 0.423은 10묶음 중 최솟값(0.423–0.432) |", "T3 공식 skill 0.423은 10묶음 중 최솟값(0.423–0.432)(SE=CI폭/3.92) |")],
 "A8": [("| LINK-NPS | 1.4, 2.5 |", "| LINK-NPS | 1.4 |"), ("| NTIS-SRC | 4.4 |", "| NTIS-SRC | 데이터 출처 |")],
 "A9": [("KOSIS 광업제조업조사 5자리(2008): KSIC5 동일코드 49/50 KSIC(사업체·종사자·출하·부가가치)",
         "KOSIS 광업제조업조사 5자리(DT_1FS1001 10차 2008–2020, DT_1FS1101 11차 2020–2024): KSIC5 동일코드 50/50(2008년 값만은 49/50 — 31312 무인 항공기는 10차 신설; 사업체·종사자·출하·부가가치). 비교 가능 5,925칸 전부 일치(2026-10-13)"),
        "SRC:LINK-KOSIS5"],
 "A10": [("DART +4.04%p [+1.09, +6.96] |", "DART +4.04%p [+1.09, +6.96](상속 없는 IPC 표, 결함 연결 비의존; C14-T2) |")],
}
BODY = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"]
APPX = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
V = {"N0": [], "C": [], "M": [], "O": BODY + APPX}
for b in BODY: V["S_" + b] = [b]
def apply(t, keys):
    dch = {}
    for k in keys:
        spec = E[k]; specs = spec if isinstance(spec, list) else [spec]
        tot = 0
        for sp in specs:
            if isinstance(sp, str) and sp.startswith("SRC:"):
                for rid in sp[4:].split(","):
                    t, n = appx_src(t, rid); tot += n
            else:
                o, n = sp
                assert t.count(o) == 1, (k, o, t.count(o))
                t = t.replace(o, n); tot += len(n) - len(o)
        dch[k] = tot
    return t, dch
env = dict(os.environ, FIG_NATIVE="1", NODE_PATH=r"C:\Users\PC\AppData\Roaming\npm\node_modules")
outp = os.path.join(T, "test_builds.json")
res = json.load(open(outp, encoding="utf-8")) if os.path.exists(outp) else {}
def pageinfo(d):
    bots, firsts = [], []
    for pg in d:
        ph = pg.rect.height; ys = []; first = None
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                txt = "".join(s["text"] for s in l["spans"]).strip()
                if txt and l["bbox"][1] < ph * 0.93:
                    ys.append(l["bbox"][3])
                    if first is None or l["bbox"][1] < first[0]: first = (l["bbox"][1], txt[:40])
        bots.append(round(max(ys), 1) if ys else 0); firsts.append(first[1] if first else "")
    return bots, firsts
base_pdf = fitz.open(R + r"\보고서_초안_v19.pdf"); base_b, base_f = pageinfo(base_pdf)
z19 = zipfile.ZipFile(R + r"\보고서_초안_v19.docx"); xml19 = z19.read("word/document.xml")
med19 = sorted(hashlib.sha1(z19.read(n)).hexdigest()[:8] for n in z19.namelist() if n.startswith("word/media/") and not n.endswith("/"))
res["_v19_base"] = dict(pdf_pages=base_pdf.page_count, page_bottoms=base_b, last_bottom=base_b[-1], slack_pt=round(785.2 - base_b[-1], 1),
                        p5_bottom=base_b[4], p6_bottom=base_b[5], p6_first=base_f[5], n_para=xml19.count(b"<w:p>") + xml19.count(b"<w:p "),
                        media=med19, n_media=len(med19), docxml_sha1=hashlib.sha1(xml19).hexdigest()[:8])
for k in sys.argv[1:]:
    t, dch = apply(src, V[k])
    if k in ("C", "M"): t = src
    md = os.path.join(T, f"ms_{k}.md"); open(md, "w", encoding="utf-8", newline="\n").write(t)
    docx = os.path.join(T, f"{k}.docx")
    r1 = subprocess.run(["node", os.path.join(T, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8")
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(T, "render.ps1"), docx], capture_output=True, text=True, encoding="utf-8", errors="replace")
    mword = re.search(r"PAGES:\s*(\d+)", r2.stdout or "")
    d = fitz.open(docx.replace(".docx", ".pdf")); b, fl = pageinfo(d)
    z = zipfile.ZipFile(docx); xml = z.read("word/document.xml")
    med = sorted(hashlib.sha1(z.read(n)).hexdigest()[:8] for n in z.namelist() if n.startswith("word/media/") and not n.endswith("/"))
    npara = xml.count(b"<w:p>") + xml.count(b"<w:p ")
    body_d = sum(v for kk, v in dch.items() if kk.startswith("O")); appx_d = sum(v for kk, v in dch.items() if kk.startswith("A"))
    res[k] = dict(edits=V[k], d_chars_body=body_d, d_chars_appendix=appx_d, per_edit=dch, pdf_pages=d.page_count, word_pages=int(mword.group(1)) if mword else None,
                  last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1), page_bottoms=b, p5_bottom=b[4], p6_bottom=b[5], p6_first=fl[5],
                  p56_boundary_same=(b[4] == base_b[4] and b[5] == base_b[5] and fl[5] == base_f[5]),
                  pages_differing_from_v19=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v19=(xml == xml19), docxml_sha1=hashlib.sha1(xml).hexdigest()[:8], n_para=npara, media_same_as_v19=(med == med19), n_media=len(med),
                  md_sha1=hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], node_err=(r1.stderr or "")[-300:], render_out=(r2.stdout or "")[-120:])
    print(k, "pdf", d.page_count, "word", res[k]["word_pages"], "last", b[-1], "slack", res[k]["slack_pt"], "dbody", body_d, "dappx", appx_d,
          "diffpages", res[k]["pages_differing_from_v19"], "xml_same", res[k]["docxml_same_as_v19"], "para", npara, "media", res[k]["media_same_as_v19"], len(med),
          "p56", res[k]["p56_boundary_same"], (r1.stderr or "")[-200:], flush=True)
    d.close(); z.close()
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("v19 base", {k: v for k, v in res["_v19_base"].items() if k != "page_bottoms"})
