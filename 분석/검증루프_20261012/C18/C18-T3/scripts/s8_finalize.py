# C18-T3 s8: verify 뒤 상태표 정정(근거 있는 것만, 정정 전 판은 cache\errata_status_v18_preverify.csv로 보존)과 build_results.json 통합.
import csv, io, json, os, sys, shutil, hashlib
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; W = AN + r"\검증루프_20261012\C18\C18-T3"
ms = open(R + r"\manuscript_v3.md", encoding="utf-8").read()
assert hashlib.sha1(ms.encode("utf-8")).hexdigest().startswith("b316df8e")
P = W + r"\errata_status_v18.csv"; BK = W + r"\cache\errata_status_v18_preverify.csv"
if not os.path.exists(BK): shutil.copyfile(P, BK)
rows = list(csv.DictReader(io.StringIO(open(BK, encoding="utf-8-sig").read())))
FIX = []
for r in rows:
    if r["ID"] == "E-C10T4-05":
        assert "통계청" not in ms.split("\n")[428]   # 429행
        d = (len("IPC-KSIC 연계표(특허청, 2013, 대조군)") - len("IPC-KSIC 연계표(2013, 대조군)")) + (len("WIPO IP 통계·IPC 기술분야 대응표") - len("WIPO IP 통계"))
        FIX.append((r["ID"], "글자증감", r["글자증감"], "+%d" % d))
        r["글자증감"] = "+%d" % d
        r["근거"] += " ; C18 verify: 429행 '통계청' 표기는 v19에 없음(첫 구절 대상 소멸), 남은 두 구절 +5·+13"
    if r["ID"] == "E-C13T1-01":
        FIX.append((r["ID"], "근거", "", "판정 경계 메모"))
        r["근거"] += " ; C18 verify: 병합판 중 E-C13T3-09 몫은 반영(독립 판정은 '부분'), 이 항목 고유 수정 '(직권)'은 없음 → '미반영' 유지"
with open(P, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# build_results.json
tb = json.load(open(W + r"\build_test\test_builds.json", encoding="utf-8"))
vb = json.load(open(W + r"\verify\v3_rebuild_out.json", encoding="utf-8"))
def sha(p): return hashlib.sha1(open(p, "rb").read()).hexdigest()
res = dict(
    기준=dict(manuscript="보고서_20261012\\manuscript_v3.md", sha1=sha(R + r"\manuscript_v3.md"), 판="v19(과제문 v18 39452163은 메인 반영 (17)로 이미 v19로 바뀜)",
              docx="보고서_초안_v19.docx", docx_sha1=sha(R + r"\보고서_초안_v19.docx"), 도구="build\\md2docx.js·render.ps1 사본(바이트 동일), FIG_NATIVE=1"),
    관문_N0=dict(docxml_same_as_v19=tb["N0"]["docxml_same_as_v19"], pdf_pages=tb["N0"]["pdf_pages"], word_pages=tb["N0"]["word_pages"], n_para=tb["N0"]["n_para"],
                n_png=tb["N0"]["n_media"], png_same=tb["N0"]["media_same_as_v19"], 합격=bool(tb["N0"]["docxml_same_as_v19"] and tb["N0"]["pdf_pages"] == 20)),
    후보_candidate=dict(파일="manuscript_v19_candidate.md", 내용="필수 0 + 권장 0 → v19와 바이트 동일", md_sha1=sha(W + r"\manuscript_v19_candidate.md"),
                     pdf_pages=tb["C"]["pdf_pages"], word_pages=tb["C"]["word_pages"], n_para=tb["C"]["n_para"], n_png=tb["C"]["n_media"], png_same=tb["C"]["media_same_as_v19"],
                     last_bottom_pt=tb["C"]["last_bottom"], 마지막쪽_여유_pt=tb["C"]["slack_pt"], p5_bottom=tb["C"]["p5_bottom"], p6_bottom=tb["C"]["p6_bottom"],
                     p6_first=tb["C"]["p6_first"], p56_경계_N0와_같음=tb["C"]["p56_boundary_same"], docxml_same_as_v19=tb["C"]["docxml_same_as_v19"], 합격=True),
    최소판_min=dict(파일="manuscript_v19_min.md", 내용="필수 0 → v19와 바이트 동일", md_sha1=sha(W + r"\manuscript_v19_min.md"), pdf_pages=tb["M"]["pdf_pages"],
                 word_pages=tb["M"]["word_pages"], n_para=tb["M"]["n_para"], n_png=tb["M"]["n_media"], 마지막쪽_여유_pt=tb["M"]["slack_pt"],
                 p56_경계_N0와_같음=tb["M"]["p56_boundary_same"], docxml_same_as_v19=tb["M"]["docxml_same_as_v19"], 합격=True),
    선택참고판_option=dict(파일="manuscript_v19_option.md", 구성="본문 O1–O9(+46자) + 부록 A1–A10(+1,050자, docx 밖)", md_sha1=sha(W + r"\manuscript_v19_option.md"),
                      pdf_pages=tb["O"]["pdf_pages"], word_pages=tb["O"]["word_pages"], n_para=tb["O"]["n_para"], png_same=tb["O"]["media_same_as_v19"],
                      마지막쪽_여유_pt=tb["O"]["slack_pt"], page4_bottom=tb["O"]["page_bottoms"][3], p5_bottom=tb["O"]["p5_bottom"], p6_bottom=tb["O"]["p6_bottom"],
                      p6_first=tb["O"]["p6_first"], 바뀐쪽=tb["O"]["pages_differing_from_v19"],
                      경고="4쪽 바닥 768.3→782.9pt(O6 표 3 +9자, 여유 약 2.3pt만 남음). 6쪽 바닥 782.6→761.7(O7 130행 −1자로 한 줄 당겨짐), 6쪽 첫 줄 동일. 21쪽 시도 없음(모든 판 20쪽)."),
    단일수정판=[dict(판=k, 항목=tb[k]["edits"], 글자=tb[k]["d_chars_body"], 쪽=tb[k]["pdf_pages"], 바뀐쪽=tb[k]["pages_differing_from_v19"], p56_same=tb[k]["p56_boundary_same"])
              for k in tb if k.startswith("S_O")],
    시도기록="N0 → C → M → S_O1–S_O9 → O 순서로 각 1회. 모두 20쪽이라 21쪽 처리(항목 빼기·0자판 교체)는 발생하지 않음.",
    독립재빌드_verify=dict(폴더="verify_build\\", 측정법="document.xml 전체 sha1, <w:p[ >] 정규식, PNG 전체 sha1, fitz words 바닥",
                       결과={k: {kk: vb[k][kk] for kk in ("docxml_same_as_v19", "docxml_sha1", "pdf_pages", "word_pages", "n_para", "n_png", "png_same_as_v19", "last_bottom", "p5", "p6")}
                           for k in ("N0", "C", "M", "O")},
                       재현=all(vb[k]["pdf_pages"] == 20 and vb[k]["n_para"] == 505 and vb[k]["png_same_as_v19"] for k in ("N0", "C", "M", "O"))
                           and vb["O"]["docxml_sha1"].startswith(tb["O"]["docxml_sha1"]) and all(vb[k]["docxml_same_as_v19"] for k in ("N0", "C", "M")),
                       비고="바닥 y는 측정법 차이로 0.4pt 다름(dict 줄 bbox 724.3 vs words 723.9)"),
)
json.dump(res, open(W + r"\build_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(FIX); print(res["독립재빌드_verify"]["재현"], res["관문_N0"])
