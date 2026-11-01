import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4"
s2 = json.load(open(os.path.join(W, r"cache\s2_v11.json"), encoding="utf-8"))
s3 = json.load(open(os.path.join(W, r"cache\s3_pages.json"), encoding="utf-8"))
body_limit = 841.9 - 1134 / 20  # A4 높이 − 하단 여백 20mm(1134 DXA = 56.7pt)
last = s3["v11"]["pages"][-1]
same_layout = all(a["bottom"] == b["bottom"] and a["last"] == b["last"] for a, b in zip(s3["v10"]["pages"], s3["v11"]["pages"]))
res = dict(
    target=dict(docx=r"분석\보고서_20261012\보고서_초안_v11.docx", pdf=r"분석\보고서_20261012\보고서_초안_v11.pdf",
                docx_sha1=s2["docx_sha1"], pdf_sha1=s2["pdf_sha1"], same_as=r"drafts\test_v11.docx/.pdf (sha1 동일)"),
    manuscript=dict(path=r"분석\보고서_20261012\manuscript_v3.md", sha1="29937d13c3ea704df7d463646e23f10dc0f29abd", lines=804,
                    equals=r"drafts\manuscript_for_v11.md", submission_range="1–444행('# [부록·제출본 제외]' 앞)"),
    method=r"C08-T2 scripts\s5_submission.py 복사본(scripts\s2_submission.py, 로직 동일) + scripts\s3_pages.py(쪽별 비교)",
    text_match=dict(normalized_chars_ms=s2["ms_chars_norm"], normalized_chars_docx=s2["docx_chars_norm"], char_diff_ops=s2["char_diff_ops"],
                    paragraphs_ms=s2["md_paras"], paragraphs_docx=s2["docx_paras"], para_diff_ops=len(s2["para_diff_ops"]), result="일치(차이 0)"),
    pdf=dict(pages=s2["pdf_pages"], pages_ok=s2["pdf_pages"] <= 20, verdict="적격(20쪽)",
             last_page_body_bottom_frac=s2["last_page_body_bottom_frac"], last_page_body_bottom_pt=last["bottom"],
             body_area_limit_pt=round(body_limit, 1), body_area_limit_frac=round(body_limit / 841.9, 3),
             spare_pt=round(body_limit - last["bottom"], 1), line_pitch_pt_last_page=s3["v11"]["last_line_pitch_median"],
             spare_lines=0,
             note="마지막 쪽 본문(참고문헌 끝)은 772.6pt, 본문 영역 하한 785.2pt. 여유 12.6pt로 한 줄(약 17–21pt)보다 작다 → 여유 0줄. 19–20쪽 범위에 한 줄이라도 늘면 21쪽(제출 불가). PREREG 경고 기준(≥0.95)에는 걸리지 않지만 실제 여유는 0줄(변경 기록 참고)",
             v10_vs_v11_page_layout_identical=same_layout,
             line168_effect="168행(+17자)은 7쪽에 있으며 v10과 v11의 20쪽 모두 쪽별 본문 하단·첫/끝 줄이 같다(줄 수 증가 흡수). 쪽 넘김 변화 없음",
             fig5_caption_page=s2["fig5_caption_page"]),
    images=dict(n_docx=s2["n_images_docx"], n_ms=s2["n_images_md"], all_match_figs=all(i["match"] for i in s2["images"]), detail=s2["images"],
                content_issue=r"1번 그림 fig01_framework.png(sha1 dcf60d48)는 figs와 같지만 그림 안 대조군 문구가 '해외 문서 FREE-IPC: 48–54%'로 v11 본문(134행 48–53%)·원천(48.225–53.478%)과 다르다 → E-C10T4-01. figs\make_diagrams.py 86행(문구)·104행(CSV 출처) 수정 후 fig01만 재렌더하면 된다. 이 폴더 build_test\에서 테스트 렌더: 크기 2007×1181·300dpi 동일, 픽셀 차이는 숫자 영역(1083,1094)-(1126,1116)뿐(새 sha1 2db9d37f) → 쪽수 영향 없음"),
    verdict="제출 가능(20쪽, 문단·그림 해시 일치). 단 그림 1 숫자 1곳 불일치(E-C10T4-01)와 마지막 쪽 여유 0줄(문구 추가 금지 수준)"
)
json.dump(res, open(os.path.join(W, "submission_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "images"}, ensure_ascii=False, indent=1))
