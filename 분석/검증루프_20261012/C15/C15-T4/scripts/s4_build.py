# C15-T4 s4: v16 원고 사본에 수정안을 적용 → node md2docx.js(FIG_NATIVE=1) → Word PDF → 쪽수·마지막 쪽 여유 실측 (C13-T3 s7 방식)
# 원고는 읽기만. 모든 산출은 build_test\ 안. 사용: python s4_build.py N0 F1a F2 ...
import os, sys, json, subprocess, hashlib, zipfile, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C15\C15-T4\build_test"
MS = R + r"\manuscript_v3.md"
src = open(MS, encoding="utf-8").read()
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("e7f27faf")
E = {  # id: (옛 문구, 새 문구) — 옛 문구는 원고에 정확히 1회
 # F1 2.1 잠긴 최종시험(U007, 과장)
 "F1a": ("가장 최근 구간은 처음부터 잠가 정답을 열어 보지 않았고,", "가장 최근 구간은 라운드마다 새로 잠갔고(T1·T3는 앞 라운드 최종시험과 결과 기간 일부 겹침),"),
 "F1b1": ("가장 최근 구간은 처음부터 잠가 정답을 열어 보지 않았고,", "가장 최근 구간은 라운드마다 새로 잠갔고,"),
 "F1b2": ("T3는 감사에서 작은 누수를 고친 뒤 다시 채점했으므로 수정 전 값(0.424)도 공개한다.",
          "T3는 누수를 고친 뒤 다시 채점해 수정 전 값(0.424)도 공개한다. T1·T3 공식 창은 앞 라운드의 다른 정의 최종시험(T1 0.418, T3 0.392)과 결과 기간이 겹친다."),
 # F2 2.5 공식 통계 대비 비율(U019)
 "F2": ("(KR 등록 0.25–0.26, CN 0.20–0.21; 예외 4.4)", "(KR 등록 2012–23년 0.25–0.26, CN 2018–22년 0.20–0.21; 예외 4.4)"),
 "F2s": ("제공 데이터 연도별 건수는 공식 통계 대비 비율이 안정적이다(KR 등록 0.25–0.26, CN 0.20–0.21; 예외 4.4).",
         "제공 데이터 건수의 공식 통계 대비 비율은 KR 등록 2012–23년 0.25–0.26, CN 2018–22년 0.20–0.21로 안정적이다."),
 "F2z": ("제공 데이터 연도별 건수는 공식 통계 대비 비율이 안정적이다(KR 등록 0.25–0.26, CN 0.20–0.21; 예외 4.4).",
         "공식 통계 대비 건수 비율은 KR 등록 2012–23년 0.25–0.26, CN 2018–22년 0.20–0.21로 안정적이다."),
 # F3 3.1.3 코드 안 분산(U031)
 "F3": ("세계 추세는 연계표로 읽히지만 한국 지표는 그렇지 않다.", "합계 추세는 연계표로 읽히지만 한국 지표는(US·JP·EP도) 그렇지 않다."),
 "F3s": ("세계 추세는 연계표로 읽히지만 한국 지표는 그렇지 않다.", "합계 추세는 연계표로 읽히지만 KR·US·JP·EP 지표는 그렇지 않다."),
 # F4 T1 권리목록 출처(U039·U020)
 "F4a": ("외부 데이터로 KIPRIS 디자인권·실용신안, ECOS 물가", "외부 데이터로 KIPRIS 권리목록(범위밖 특허·디자인·실용신안), ECOS 물가"),
 "F4b": ("KIPRIS 디자인·실용신안은 T1 피처로", "KIPRIS 권리목록(범위밖 특허·디자인·실용신안)은 T1 피처로"),
 "F4c": ("KIPRIS Plus 디자인·실용신안, ECOS 물가, 한국은행 산업연관표", "KIPRIS Plus 권리목록·디자인·실용신안, ECOS 물가, 한국은행 산업연관표"),
 # F5 3.4 N9 AUC 우위(U042)
 "F5a": ("모형의 우위는 순위 전반(AUC)에 있다.", "두 원판 대비 우위는 순위 전반(AUC)에 있다."),
 "F5b": ("모형의 우위는 순위 전반(AUC)에 있다.", "두 원판 대비 우위는 순위 전반(AUC)에 있다(원판 챔피언 AUC 0.899는 더 높다)."),
 # F6 3.6.5 종합(U086)
 "F6": ("예측 피처(T3·N9·N6), 수출 정합은 같고", "예측 피처(T3 경계·N9·N6), 수출 정합은 같고"),
 # F7 1.4 등록 656,360건 표기(U003-b)
 "F7": ("제공 범위 KR 등록특허 656,360건을", "제공 범위 KR 등록특허·실용신안 656,360건을"),
}
V = {"N0": [], "S1a": ["F1a"], "S1b": ["F1b1", "F1b2"], "S2": ["F2"], "S2s": ["F2s"], "S3": ["F3"], "S3s": ["F3s"],
     "S4": ["F4a", "F4b", "F4c"], "S5a": ["F5a"], "S5b": ["F5b"], "S6": ["F6"], "S7": ["F7"],
     "P1": ["F1a", "F2", "F4a", "F4b", "F4c", "F5a", "F6"],
     "P2": ["F1a", "F2", "F3", "F4a", "F4b", "F4c", "F5a", "F6", "F7"],
     "P3": ["F1b1", "F1b2", "F2", "F3", "F4a", "F4b", "F4c", "F5b", "F6", "F7"]}
for k in sys.argv[1:]:
    if k.startswith("+"):  # 임시 조합: +F1a,F2
        V[k] = k[1:].split(",")
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
base_b = pagebottoms(fitz.open(R + r"\보고서_초안_v16.pdf"))
xml16 = zipfile.ZipFile(R + r"\보고서_초안_v16.docx").read("word/document.xml")
for k in sys.argv[1:]:
    t = src; dch = {}
    for e in V[k]:
        o, n = E[e]; assert t.count(o) == 1, (e, t.count(o)); t = t.replace(o, n); dch[e] = len(n) - len(o)
    md = os.path.join(T, f"ms_{k.strip('+').replace(',', '_')}.md"); open(md, "w", encoding="utf-8").write(t)
    docx = md.replace("ms_", "").replace(".md", ".docx")
    r1 = subprocess.run(["node", os.path.join(T, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8")
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(T, "render.ps1"), docx], capture_output=True, text=True)
    d = fitz.open(docx.replace(".docx", ".pdf")); b = pagebottoms(d)
    xml = zipfile.ZipFile(docx).read("word/document.xml")
    res[k] = dict(edits=V[k], d_chars=sum(dch.values()), per_edit=dch, pages=d.page_count, last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1),
                  page_bottoms=b, pages_differing_from_v16=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v16=(xml == xml16), node_err=r1.stderr[-300:], render_out=r2.stdout[-120:])
    print(k, res[k]["pages"], res[k]["last_bottom"], res[k]["slack_pt"], res[k]["d_chars"], res[k]["per_edit"], res[k]["pages_differing_from_v16"], "xml_same", res[k]["docxml_same_as_v16"], r1.stderr[-200:])
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("v16 base page bottoms", base_b)
