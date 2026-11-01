import sys, re, ntis_B_common as c, urllib.request, urllib.parse, time, gzip
sys.stdout.reconfigure(encoding="utf-8")
k=c.load_key()
def raw(params, gz):
    q={"apprvKey":k,"collection":"project"}; q.update(params)
    url="https://www.ntis.go.kr/rndopen/openApi/public_project?"+urllib.parse.urlencode(q)
    h={"User-Agent":"Mozilla/5.0"}
    if gz: h["Accept-Encoding"]="gzip"
    t=time.time()
    with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=120) as r:
        b=r.read(); enc=r.headers.get("Content-Encoding")
    time.sleep(0.6)
    return b, enc, time.time()-t
b,enc,dt=raw({"SRWR":"","addQuery":"PY=2015/SAME","startPosition":1,"displayCnt":1000},True)
print("gzip", len(b), enc, round(dt,1))
for sp in [10001, 50001, 54001]:
    b,enc,dt=raw({"SRWR":"","addQuery":"PY=2015/SAME","startPosition":sp,"displayCnt":10},True)
    if enc=="gzip": b=gzip.decompress(b)
    x=b.decode("utf-8","replace")
    print(sp, len(b), re.search(r"<HITS>(\d+)",x) and re.search(r"<HITS>(\d+)",x).group(1), re.findall(r"<ProjectNumber>(\d+)",x)[:3], x[:300] if len(b)<2000 else "")
