# -*- coding: utf-8 -*-
r"""C12-T3 KOSIS(C07-T1 klib 복사) OpenAPI 도우미. 키는 수집파이프라인\collect_kosis_new.py에서 읽기만(PROTOCOL_R2 5.1b).
키 값은 출력·저장하지 않는다. 요청 간 ≥2초. 오류 40·41·42면 즉시 중단.
응답 원문은 download\<sub>\ 에 저장, 호출 기록은 download\_calllog.txt(키 가림)."""
import os, re, json, time, hashlib, urllib.request, urllib.parse, datetime
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.join(BASE, r"분석\검증루프_20261012\C12\C12-T3")
DL = os.path.join(HERE, "download")


def _key():
    s = open(os.path.join(BASE, "수집파이프라인", "collect_kosis_new.py"), encoding="utf-8").read()
    m = re.search(r'KEY\s*=\s*urllib\.parse\.quote\(\s*"([^"]+)"\s*\)', s)
    return urllib.parse.quote(m.group(1))


KEY = _key()
_last = [0.0]


class Stop(Exception):
    pass


def _red(s):
    return str(s).replace(KEY, "<KEY>").replace(urllib.parse.unquote(KEY), "<KEY>")


def call(endpoint, params, sub, name, gap=2.2):
    q = dict(params); q["apiKey"] = "__K__"
    qs = urllib.parse.urlencode(q, safe="+,").replace("__K__", KEY)
    url = f"https://kosis.kr/openapi/{endpoint}?{qs}"
    w = time.time() - _last[0]
    if w < gap:
        time.sleep(gap - w)
    _last[0] = time.time()
    try:
        raw = urllib.request.urlopen(url, timeout=120).read()
    except Exception as e:
        raise RuntimeError(_red(e))
    d = os.path.join(DL, sub); os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, name + ".json")
    open(fp, "wb").write(raw)
    h = hashlib.sha256(raw).hexdigest()
    with open(os.path.join(DL, "_calllog.txt"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\t{sub}/{name}.json\t{len(raw)}\t{h}\t{_red(url)}\n")
    txt = raw.decode("utf-8", errors="replace")
    try:
        j = json.loads(txt)
    except Exception:
        j = json.loads(re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', txt))
    if isinstance(j, dict) and str(j.get("err")) in ("40", "41", "42"):
        raise Stop(_red(j))
    return j


def meta(tbl, typ, sub="meta", org="101"):
    return call("statisticsData.do", {"method": "getMeta", "type": typ, "orgId": org, "tblId": tbl, "format": "json"}, sub, f"meta_{tbl}_{typ}")


def lst(parent, sub="list", vw="MT_OTITLE"):
    return call("statisticsList.do", {"method": "getList", "vwCd": vw, "parentListId": parent, "format": "json", "jsonVD": "Y"}, sub, f"list_{vw}_{parent or 'top'}")


def data(tbl, sub, name, org="101", **kw):
    p = {"method": "getList", "orgId": org, "tblId": tbl, "format": "json", "jsonVD": "Y"}
    p.update(kw)
    return call("Param/statisticsParameterData.do", p, sub, name)
