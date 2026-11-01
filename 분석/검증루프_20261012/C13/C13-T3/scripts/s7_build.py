# C13-T3 s7: 원고 사본에 편집 묶음을 적용 → node md2docx.js(FIG_NATIVE=1) → Word PDF → 쪽수·마지막 쪽 여유
# 원고는 읽기만. 모든 산출은 build_test\ 안.
import os, sys, json, subprocess, hashlib, zipfile, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T3\build_test"
MS = R + r"\manuscript_v3.md"
src = open(MS, encoding="utf-8").read()
assert hashlib.sha1(open(MS, "rb").read()).hexdigest().startswith("13c3d505")

E = {  # 편집 id: (옛 문구, 새 문구)  — 옛 문구는 원고에 정확히 1회 있어야 함
 "P219": ("해외 신호와 외부 데이터가 그 위에 더한다.", "해외 신호가 그 위에 더한다(외부 데이터 몫은 작아 최종시험에서는 빼도 같거나 오른다)."),
 "P168": ("한국 RTA 순위상관은 0.19–0.37이며", "한국 RTA 순위상관은 0.19–0.37(최빈 동률을 반대로 처리하면 0.21–0.48)이며"),
 "P143": ("(물가·산업연관표는 원점 이후 공표된 현행 빈티지", "(물가·산업연관표·생산지수는 원점 이후 공표된 현행 빈티지"),
 "P079b": ("(예외: 물가·산업연관표의 현행 빈티지, 2.5)", "(예외: 물가·산업연관표·생산지수의 현행 빈티지, 2.5)"),
 "P079a": ("① 모든 트랙의 모집단·단위·결과는 제공 데이터에서 정의한다", "① 모든 트랙의 모집단·단위는 제공 데이터에서 정의한다(N4 폐업·N7 과제 진입 결과는 외부 사건)"),
 "P013": ("AUC 0.002~0.005로 작다.", "AUC 0.002~0.005로 작다(해설서 단독 대비는 이원 군집 CI가 0 포함)."),
 "P336": ("T3 메인그룹 성장과 N9·N6에서는", "T3 메인그룹 성장(경계)과 N9·N6에서는"),
 "P374": ("거친 코드로 보면 두 추세가 서로 지워진다(연계표 대비 기술통계).", "거친 코드로 보면 두 추세가 서로 지워진다(연계표 대비 기술통계; OLED 증가는 두 기업 출원에 기대어 한 기업 몫을 20%로 제한하면 사라진다)."),
 "P387": ("이력 없는 새 분쟁은 9/78건(과거 분쟁 기준선 4/78)이나 차이의 불확실성이 큼", "새 분쟁은 9/78건(과거 분쟁 기준선 4/78)이나 전향 채점에서는 4/65건(M0 7/65건)"),
 "P395": ("(포기 위험 열은 공식 외 시험에서 단순 지속 지수보다 낫지 않았다)", "(포기·분쟁 위험 열은 공식 외 시험에서 단순 지속 지수보다 낫지 않았다)"),
 "P168v2": ("주) 이 RTA는 제공 50개 제조업 KSIC 안의 상대 특화다(WIPO 35개 분야로 옮기면 한국 전체 기술 특화와의 순위상관 0.45, 5개국 중 최저). 국가별 분야 수록 범위를 재가중하거나(한국 1위 9개, RTA>1 27–29개) 산업별 출원 1위 출원인을 빼도(1위 8개)",
            "주) RTA는 제공 50개 제조업 KSIC 안의 상대 특화다(WIPO 35개 분야로 옮기면 한국 전체 기술 특화와의 순위상관 0.45, 5개국 중 최저). 분야 수록 범위를 재가중하거나(1위 9개, RTA>1 27–29개) 산업별 1위 출원인을 빼도(1위 8개)"),
 "P168v2b": ("한국 RTA 순위상관은 0.19–0.37이며 어느 배정이 맞는지는 판정하지 못했다.", "한국 RTA 순위상관은 0.19–0.37(동률 규칙에 따라 최대 0.48)이며 어느 쪽이 맞는지 판정하지 못했다."),
 "Q73": ("(공개번호 연결이면 46.9–53.5%)", "(공개 시점 무관, 상속분의 22%는 더 늦게 공개된 KR 문서; 공개번호 연결이면 46.9–53.5%)"),
 "Q410": ("(연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%).", "(연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%; 2013년 이후 발급 코드는 99% 이상이 사업자번호를 갖고 그 이전 코드는 17–42%뿐이어서 이전 코드의 번호는 사후 보완분으로 보인다; 2013년 이후 코드만 보면 N4 0.19, N7 0.25)."),
 "Y410": ("국민연금 상호 키 보정 후 N4는 0.140, N7은 0.251이다. ", ""),
 "Y409": ("공식 성적은 트랙마다 잠긴 최종시험 1회 값이고, T3 시험 구간(2022–24)은 코로나19 이후다.", "T3 시험 구간(2022–24)은 코로나19 이후다."),
 "Q410s": ("(연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%).", "(연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%; 2013년 이후 발급 코드만 보면 N4 0.19, N7 0.25)."),
 "P286": ("39.3%(과거 분쟁 기준선 33.3%)다.", "39.3%(과거 분쟁 기준선 33.3%, 차이 CI 하한 0.0)다."),
 "P286b": ("(권리자 청구 추정 약 9% 포함). 무효심판만으로 다시 채점해도 상위 1% 포착은 39.3%(과거 분쟁 기준선 33.3%)다.", "(권리자 청구 추정 약 9%). 무효심판만으로 재채점해도 상위 1% 포착은 39.3%(과거 분쟁 기준선 33.3%, 차이 하한 0)다."),
 # 축약 후보(같은 쪽 상쇄용)
 "X168": ("국가별 분야 수록 범위를 재가중하거나(한국 1위 9개, RTA>1 27–29개) 산업별 출원 1위 출원인을 빼도(1위 8개) 큰 그림은 같지만,", "분야 수록 범위를 재가중하거나(한국 1위 9개, RTA>1 27–29개) 산업별 1위 출원인을 빼도(1위 8개) 큰 그림은 같지만,"),
 "X219": ("예측 상위 10%의 56.9%는 직전 3년 KR 공개가 0건인 셀이었고, KR 5건 이상 셀만 봐도 AUC 0.807(기준선 0.499)이다.", "예측 상위 10%의 56.9%는 직전 3년 KR 0건 셀이었고, KR 5건 이상 셀만 봐도 AUC 0.807(기준선 0.499)이다."),
}
V = {
 "N0": [],
 "B1": ["P219", "P168"],
 "B2": ["P219", "P168", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395"],
 "B3": ["P079b", "P079a", "P013", "P374"],
 "B4": ["P219", "P168", "P143", "P079b", "P013", "P336", "P374", "P387", "P395", "X168", "X219"],
 "B5": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395"],
 "B6": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b"],
 "S168v2": ["P168v2", "P168v2b"],
 "B7": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "Q73"],
 "B8": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "Q410", "Y410", "Y409"],
 "S410": ["Q410"],
 "B13": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "P286b"],
 "B14": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "P286b", "Q73", "Q410s", "Y410", "Y409"],
 "B11": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "P286"],
 "B12": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "P286", "Q73", "Q410s", "Y410", "Y409"],
 "B9": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "Q410s", "Y410", "Y409"],
 "B10": ["P219", "P143", "P079b", "P079a", "P013", "P336", "P374", "P387", "P395", "P168v2", "P168v2b", "Q410s"],
 "S168": ["P168"], "S219": ["P219"], "S168x": ["P168", "X168"], "S219x": ["P219", "X219"],
 "S336": ["P336"], "S387": ["P387"], "S395": ["P395"], "S143": ["P143"], "S013": ["P013"], "S374": ["P374"], "S079": ["P079a", "P079b"],
}
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
base_b = pagebottoms(fitz.open(R + r"\보고서_초안_v14.pdf"))
xml14 = zipfile.ZipFile(R + r"\보고서_초안_v14.docx").read("word/document.xml")
for k in sys.argv[1:]:
    t = src; dch = {}
    for e in V[k]:
        o, n = E[e]; assert t.count(o) == 1, (e, t.count(o)); t = t.replace(o, n); dch[e] = len(n) - len(o)
    md = os.path.join(T, f"ms_{k}.md"); open(md, "w", encoding="utf-8").write(t)
    docx = os.path.join(T, f"{k}.docx")
    r1 = subprocess.run(["node", os.path.join(T, "md2docx.js"), md, R + r"\figs", docx], env=env, capture_output=True, text=True, encoding="utf-8")
    r2 = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(T, "render.ps1"), docx], capture_output=True, text=True)
    d = fitz.open(os.path.join(T, f"{k}.pdf")); b = pagebottoms(d)
    xml = zipfile.ZipFile(docx).read("word/document.xml")
    res[k] = dict(edits=V[k], d_chars=sum(dch.values()), per_edit=dch, pages=d.page_count, last_bottom=b[-1], slack_pt=round(785.2 - b[-1], 1),
                  page_bottoms=b, pages_differing_from_v14=[i + 1 for i, (x, y) in enumerate(zip(b, base_b)) if abs(x - y) > 0.5] + ([f"쪽수 {d.page_count}"] if d.page_count != len(base_b) else []),
                  docxml_same_as_v14=(xml == xml14), node_err=r1.stderr[-300:], render_out=r2.stdout[-120:])
    print(k, res[k]["pages"], res[k]["last_bottom"], res[k]["slack_pt"], res[k]["d_chars"], res[k]["pages_differing_from_v14"], "xml_same", res[k]["docxml_same_as_v14"], r1.stderr[-200:])
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
