# C14-T2 s11: submission_check.json 조립(cache/s2_v15.json, s3_pages.json, s4_format.json, s9_fig_numbers.csv, build_test/test_builds.json)
import json, sys, hashlib
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T2"
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
s2 = json.load(open(W + r"\cache\s2_v15.json", encoding="utf-8")); s3 = json.load(open(W + r"\cache\s3_pages.json", encoding="utf-8"))
s4 = json.load(open(W + r"\cache\s4_format.json", encoding="utf-8")); tb = json.load(open(W + r"\build_test\test_builds.json", encoding="utf-8"))
fg = pd.read_csv(W + r"\cache\s9_fig_numbers.csv", encoding="utf-8-sig")
v15 = s3["v15"]
out = dict(
 기준=dict(manuscript=R + r"\manuscript_v3.md", manuscript_sha1=hashlib.sha1(open(R + r"\manuscript_v3.md", "rb").read()).hexdigest(), lines=805,
          same_as_drafts_manuscript_for_v15=True, docx_sha1=s2["docx_sha1"], pdf_sha1=s2["pdf_sha1"], docx_pdf_same_as_drafts_test_v15=True),
 제출불가기준=dict(
   a_pages=dict(value=v15["pages"], pass_=v15["pages"] <= 20),
   b_paragraphs=dict(md_paras=s2["md_paras"], docx_paras=s2["docx_paras"], ms_chars_norm=s2["ms_chars_norm"], docx_chars_norm=s2["docx_chars_norm"], char_diff_ops=s2["char_diff_ops"], para_diff_ops=len(s2["para_diff_ops"]), pass_=s2["char_diff_ops"] == 0 and not s2["para_diff_ops"]),
   c_png=dict(n=len(s2["images"]), matched=sum(i["match"] for i in s2["images"]), images=[(i["expected"], i["sha1"]) for i in s2["images"]], pass_=all(i["match"] for i in s2["images"])),
   d_format=dict(n_tables=s4["n_tables"], inconsistent_tables=s4["inconsistent_tables"], empty_cells=sum(t["n_empty"] for t in s4["tables"]), markdown_residue={k: v for k, v in s4["markdown_residue"].items() if v and k != "]("},
                 note_paren_bracket="'](' 1건은 CI 닫는 괄호 뒤 여는 괄호(275행 '0.179](국민연금…') — 마크다운 링크 아님", pdf_marks=s4["pdf_marks"], fonts=s4["pdf_fonts"], pass_=not s4["inconsistent_tables"] and not s4["pdf_marks"])),
 판정="제출 가능(경고 1: 마지막 쪽 여유 1줄 미만)",
 마지막쪽=dict(body_bottom_pt=v15["last_body_bottom"], body_limit_pt=785.2, slack_pt=v15["slack_pt"], line_gap_pt=round(v15["line_gap_pt"], 1), slack_lines=v15["slack_lines"],
             v14_vs_v15_pages_changed=s3["cmp_v14_v15"], note="B13 시험 빌드 예측(3쪽 733.4, 17쪽 758.4, 마지막 778.9)과 실측 동일; v15 docx document.xml = C13-T3 build_test\B13.docx와 바이트 동일"),
 그림숫자=dict(checked=int(len(fg)), in_body=int(fg.in_body.sum()), not_in_body=fg[~fg.in_body][["fig", "item", "token"]].values.tolist(),
            note="본문에 없는 4개는 그림 2의 N4 CI·N5 AUC/CI로 부록 OFF-*와 같고 본문과 모순 없음. 그림 1 숫자(87–90%, 40.0%, 7.1%, 29.4%, 40.4%, 48–53%, 50/50, 16/50, 48/50) 본문 일치. v15 변경 줄과 겹치는 그림 숫자 없음(그림 재렌더 불필요)."),
 시험빌드=tb,
)
json.dump(out, open(W + r"\submission_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in ["제출불가기준", "판정", "마지막쪽"]}, ensure_ascii=False, default=str)[:2500])
