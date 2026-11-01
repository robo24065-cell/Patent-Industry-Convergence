import sys, zipfile, hashlib, fitz
sys.stdout.reconfigure(encoding="utf-8")
B=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T3"
R=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
x14=zipfile.ZipFile(R+r"\보고서_초안_v14.docx").read("word/document.xml")
def media(z): return {n:hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}
m14=media(zipfile.ZipFile(R+r"\보고서_초안_v14.docx"))
def bot(p):
    d=fitz.open(p); o=[]
    for pg in d:
        H=pg.rect.height; ws=[w for w in pg.get_text("words") if w[4].strip() and w[1]<H*0.93]
        o.append(round(785.2-max(w[3] for w in ws),1))
    return d.page_count,o
for k in ["B13","B14","N0"]:
    zr=zipfile.ZipFile(B+rf"\verify_3\r_{k}.docx"); zb=zipfile.ZipFile(B+rf"\build_test\{k}.docx")
    xr=zr.read("word/document.xml")
    n,o=bot(B+rf"\verify_3\r_{k}.pdf"); n2,o2=bot(B+rf"\build_test\{k}.pdf")
    print(k,"xml==build_test",xr==zb.read("word/document.xml"),"xml==v14",xr==x14,"png==v14",media(zr)==m14,"pages",n,"slack",o,"same_as_buildtest_pdf",o==o2)
# 1쪽·20쪽 구조
d=fitz.open(R+r"\보고서_초안_v14.pdf")
for pi in [0,2,16,17,18,19]:
    pg=d[pi]; H=pg.rect.height
    lines=[]
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines",[]):
            t="".join(s["text"] for s in l["spans"]).strip()
            if t: lines.append((round(l["bbox"][1],1),round(l["bbox"][3],1),t[:40]))
    lines.sort()
    print("page",pi+1,"H",H,"first",lines[:1],"last3",lines[-3:])
