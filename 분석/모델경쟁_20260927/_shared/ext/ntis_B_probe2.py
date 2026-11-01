import sys, re, ntis_B_common as c
sys.stdout.reconfigure(encoding="utf-8")
def get(params):
    q={"collection":"project","startPosition":1,"displayCnt":1}; q.update(params)
    x=c.call("public_project",q)
    tot=re.search(r"<TOTALHITS>(\d+)",x)
    return tot and tot.group(1), x
for t in [{"SRWR":"","addQuery":"PY=2015/SAME"},
          {"SRWR":"2015","searchFd":"PY"},
          {"SRWR":"연구","addQuery":"PY=2015/SAME"},
          {"SRWR":"기술","addQuery":"PY=2015/SAME"},
          {"SRWR":"a","addQuery":"PY=2015/SAME"},
          {"SRWR":"2015","addQuery":"PY=2015/SAME"},
          {"SRWR":"2015","searchFd":"BI","addQuery":"PY=2015/SAME"},
          {"SRWR":"2015","searchFd":"PY","addQuery":"PY=2015/SAME"},
          ]:
    n,x=get(t); print(t,n)
