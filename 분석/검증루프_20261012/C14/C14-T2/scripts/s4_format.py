# C14-T2 s4: v15 docx 서식 점검(표 행별 셀 수, 남은 마크다운 기호, 빈 표 칸) + B13 시험 빌드와 document.xml 비교 + PDF 쪽별 남은 기호
import sys, re, zipfile, json, hashlib, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
L = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012"
OUT = sys.argv[1]
z = zipfile.ZipFile(R + r"\보고서_초안_v15.docx")
xml = z.read("word/document.xml").decode("utf-8")
def txt(s):
    t = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", s))
    return t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
tables = []
for ti, tb in enumerate(re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S)):
    rows = re.findall(r"<w:tr[ >].*?</w:tr>", tb, re.S)
    ncell = []; empty = []; head = ""
    for ri, r in enumerate(rows):
        cells = re.findall(r"<w:tc>.*?</w:tc>", r, re.S)
        # gridSpan 반영
        n = 0
        for c in cells:
            m = re.search(r'<w:gridSpan w:val="(\d+)"', c); n += int(m.group(1)) if m else 1
            if not txt(c).strip(): empty.append((ri, cells.index(c)))
        ncell.append(n)
        if ri == 0: head = " | ".join(txt(c)[:12] for c in cells)
    tables.append(dict(table=ti + 1, rows=len(rows), cols=sorted(set(ncell)), consistent=len(set(ncell)) == 1, empty_cells=empty[:10], n_empty=len(empty), head=head[:120]))
paras = [txt(p) for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)]
alltext = "\n".join(paras)
marks = {k: [p[:80] for p in paras if k in p][:5] for k in ["**", "`", "](", "[그림", "<br", "\\|", "~~"]}
# '|'는 본문 문장 안 표 구분자가 남았는지
pipe = [p[:100] for p in paras if "|" in p][:10]
# PDF 쪽별 기호
d = fitz.open(R + r"\보고서_초안_v15.pdf")
pdf_marks = []
for i, pg in enumerate(d):
    t = pg.get_text()
    for k in ["**", "`", "[그림", "�"]:
        if k in t: pdf_marks.append((i + 1, k, t[max(0, t.find(k) - 30): t.find(k) + 30]))
# 글꼴 대체 여부
fonts = sorted({f[3] for pg in d for f in pg.get_fonts()})
b13 = L + r"\C13\C13-T3\build_test\B13.docx"
xml13 = zipfile.ZipFile(b13).read("word/document.xml").decode("utf-8")
# B13 대비 차이(부록은 docx 범위 밖이므로 같아야 함)
same_b13 = xml13 == xml
res = dict(n_tables=len(tables), tables=tables, inconsistent_tables=[t["table"] for t in tables if not t["consistent"]],
           markdown_residue=marks, pipe_paras=pipe, pdf_marks=pdf_marks, pdf_fonts=fonts,
           docx_xml_same_as_C13_B13=same_b13, n_paras_xml=len(paras))
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "tables"}, ensure_ascii=False, indent=1)[:3000])
for t in tables: print(t["table"], t["rows"], t["cols"], t["consistent"], t["n_empty"], t["head"][:60])
