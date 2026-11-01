import sys, re, ntis_B_common as c, json
sys.stdout.reconfigure(encoding="utf-8")
out={}
for y in range(1998,2027):
    x=c.call("public_project",{"collection":"project","SRWR":"","addQuery":f"PY={y}/SAME","startPosition":1,"displayCnt":1})
    tot=re.search(r"<TOTALHITS>(\d+)",x); tot=int(tot.group(1)) if tot else None
    nav={}
    for name,body in re.findall(r'<NAVIGATIONENTRY NAME="([^"]+)">(.*?)</NAVIGATIONENTRY>',x,re.S):
        nav[name]=re.findall(r'<NAVIGATIONELEMENT NAME="([^"]*)" COUNT="(\d+)"',body)
    out[y]={"total":tot,"nav":nav}
    print(y,tot,{k:len(v) for k,v in nav.items()})
json.dump(out,open(c.WORK+r"\ntis_B_year_nav.json","w",encoding="utf-8"),ensure_ascii=False)
