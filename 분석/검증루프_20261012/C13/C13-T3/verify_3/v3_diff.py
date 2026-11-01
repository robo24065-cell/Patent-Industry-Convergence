import sys, difflib
sys.stdout.reconfigure(encoding="utf-8")
R=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\manuscript_v3.md"
T=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T3\build_test"
a=open(R,encoding="utf-8").read()
for k in sys.argv[1:]:
    b=open(T+rf"\ms_{k}.md",encoding="utf-8").read()
    al=a.splitlines(); bl=b.splitlines()
    print(k,"len diff",len(b)-len(a),"lines",len(al),len(bl))
    for i,(x,y) in enumerate(zip(al,bl)):
        if x!=y:
            sm=difflib.SequenceMatcher(None,x,y,autojunk=False)
            ch=[(t,x[i1:i2],y[j1:j2]) for t,i1,i2,j1,j2 in sm.get_opcodes() if t!="equal"]
            print(" L%d d=%d"%(i+1,len(y)-len(x)), ch[:4])
