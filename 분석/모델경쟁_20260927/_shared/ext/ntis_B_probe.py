import sys, re, ntis_B_common as c, urllib.request, urllib.parse, time
sys.stdout.reconfigure(encoding="utf-8")
def get(params):
    q={"collection":"project","startPosition":1,"displayCnt":1}; q.update(params)
    try:
        x=c.call("public_project",q)
    except Exception as e:
        return f"ERR {e}"
    tot=re.search(r"<TOTALHITS>(\d+)",x); hits=re.search(r"<HITS>(\d+)",x)
    err=re.search(r"<(ERROR|error)[^>]*>(.{0,200})",x)
    yrs=re.findall(r'projectyear.*?</NAVIGATIONENTRY>',x,re.S)
    return f"len={len(x)} tot={tot and tot.group(1)} hits={hits and hits.group(1)} err={err and err.group(0)[:150]} head={x[:0]}"
tests=[
 {"SRWR":""},
 {"SRWR":"*"},
 {"SRWR":"반도체","addQuery":"PY=2015/SAME"},
 {"SRWR":"반도체","addQuery":"PY=2015"},
 {"SRWR":"반도체","searchFd":"KW"},
 {"SRWR":"반도체","searchFd":"TI"},
 {"SRWR":"반도체","displayCnt":100},
 {"SRWR":"반도체","displayCnt":1000},
]
for t in tests:
    print(t, get(t))
