# verify_3: 지면 실측 주장 독립 재계산 (읽기 전용; 원 PDF/docx에서 직접)
import sys, json, hashlib, zipfile, re, fitz, os
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T3\build_test"
print("manuscript_v3 sha1", hashlib.sha1(open(R+r"\manuscript_v3.md","rb").read()).hexdigest())
print("md2docx same", hashlib.sha1(open(T+r"\md2docx.js","rb").read()).hexdigest()==hashlib.sha1(open(R+r"\build\md2docx.js","rb").read()).hexdigest())
def bottoms(p):
    d=fitz.open(p); out=[]
    for pg in d:
        H=pg.rect.height
        ws=[w for w in pg.get_text("words") if w[4].strip() and w[1]<H*0.93]
        out.append(round(max(w[3] for w in ws),1) if ws else 0)
    return d.page_count,out
v14n,v14b=bottoms(R+r"\보고서_초안_v14.pdf")
print("v14", v14n, [round(785.2-x,1) for x in v14b])
z14=zipfile.ZipFile(R+r"\보고서_초안_v14.docx")
x14=z14.read("word/document.xml")
med14={n:hashlib.sha1(z14.read(n)).hexdigest() for n in z14.namelist() if n.startswith("word/media/")}
print("v14 media", len(med14))
res={}
for f in sorted(os.listdir(T)):
    if not f.endswith(".pdf"): continue
    k=f[:-4]; n,b=bottoms(os.path.join(T,f))
    z=zipfile.ZipFile(os.path.join(T,k+".docx")); x=z.read("word/document.xml")
    med={nn:hashlib.sha1(z.read(nn)).hexdigest() for nn in z.namelist() if nn.startswith("word/media/")}
    npara=len(re.findall(rb"<w:p[ >]",x))
    diff=[i+1 for i,(a,c) in enumerate(zip(b,v14b)) if abs(a-c)>0.5]
    res[k]=dict(pages=n,slack_last=round(785.2-b[-1],1),diff=diff,xml_same=(x==x14),media_same=(med==med14),npara=npara)
    print(k,n,res[k]["slack_last"],diff,"xml",x==x14,"png",med==med14,"para",npara, "p3",round(785.2-b[2],1),"p17",round(785.2-b[16],1) if n>16 else None)
print("v14 para", len(re.findall(rb"<w:p[ >]",x14)))
json.dump(res,open(os.path.join(os.path.dirname(__file__),"v3_recheck.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
