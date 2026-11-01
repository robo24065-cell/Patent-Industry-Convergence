import sys, json, glob, os, re
sys.stdout.reconfigure(encoding="utf-8")
D = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\download\crossref"
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    try:
        m = json.load(open(f, encoding="utf-8"))["message"]
    except Exception as e:
        print(f, "ERR", open(f, encoding="utf-8", errors="replace").read()[:200]); continue
    au = "; ".join(f"{a.get('family')}, {a.get('given')}" for a in m.get("author", []))
    yr = (m.get("published-print") or m.get("published") or m.get("issued"))["date-parts"][0]
    print(json.dumps(dict(doi=m.get("DOI"), title=m.get("title"), container=m.get("container-title"), vol=m.get("volume"), issue=m.get("issue"),
                          page=m.get("page"), year_print=(m.get("published-print") or {}).get("date-parts"), issued=m.get("issued", {}).get("date-parts"),
                          online=(m.get("published-online") or {}).get("date-parts"), authors=au), ensure_ascii=False))
x = os.path.join(D, "arxiv_2205.01833.xml")
if os.path.exists(x):
    t = open(x, encoding="utf-8").read()
    print("arxiv title:", re.findall(r"<title>(.*?)</title>", t, re.S)[1:2], "published:", re.findall(r"<published>(.*?)</published>", t), "authors:", re.findall(r"<name>(.*?)</name>", t))
