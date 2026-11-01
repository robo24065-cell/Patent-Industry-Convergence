# ECOS 생산자물가(기본분류 404Y014, 품목 404Y016) 4개 사례 품목 2007–2018 재수신 → 기존 _shared 보유본과 대조. 키는 API키_기록.txt에서 읽기만, 출력 금지.
import re, sys, time, json, hashlib, datetime, urllib.request, pathlib
sys.stdout.reconfigure(encoding="utf-8")
H = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1")
t = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\API키_기록.txt").read_text(encoding="utf-8")
i = t.index("[한국은행 ECOS"); j = t.find("\n[", i + 1); K = re.search(r"키:\s*(\S+)", t[i:j]).group(1)
out = H / "download" / "ecos"; out.mkdir(parents=True, exist_ok=True)
jobs = [("404Y014", "309211AA"), ("404Y014", "309212AA"), ("404Y014", "309112AA"), ("404Y014", "311253AA"), ("404Y016", "30911201AA")]
for tb, it in jobs:
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{K}/json/kr/1/100/{tb}/A/2007/2018/{it}"
    raw = urllib.request.urlopen(url, timeout=120).read()
    fp = out / f"ecos_{tb}_{it}_2007_2018.json"; fp.write_bytes(raw)
    with open(H / "download" / "_calllog.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\tecos/{fp.name}\t{len(raw)}\t{hashlib.sha256(raw).hexdigest()}\t" + url.replace(K, "<KEY>") + "\n")
    jj = json.loads(raw)
    rows = jj.get("StatisticSearch", {}).get("row", [])
    print(tb, it, len(rows), [(r["TIME"], r["DATA_VALUE"]) for r in rows][:3], str(jj)[:150] if not rows else "")
    time.sleep(2.2)
