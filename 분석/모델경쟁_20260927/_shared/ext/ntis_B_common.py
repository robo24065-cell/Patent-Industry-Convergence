# NTIS OpenAPI 공통 (임무 B). 키 값은 파일에서 읽고 출력하지 않는다.
import re, time, urllib.request, urllib.parse, os, sys
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = BASE + r"\분석\모델경쟁_20260927\_shared\ext"
WORK = EXT + r"\ntis_B_work"
LOG = WORK + r"\ntis_B_calllog.txt"

def load_key():
    txt = open(BASE + r"\API키_기록.txt", encoding="utf-8").read()
    sec = txt.split("[NTIS", 1)[1]
    m = re.search(r"키\s*:\s*(\S+)", sec)
    return m.group(1)

_KEY = None
_last = [0.0]
def call(endpoint, params, timeout=60):
    global _KEY
    if _KEY is None: _KEY = load_key()
    wait = 0.6 - (time.time() - _last[0])
    if wait > 0: time.sleep(wait)
    q = {"apprvKey": _KEY}; q.update(params)
    url = "https://www.ntis.go.kr/rndopen/openApi/" + endpoint + "?" + urllib.parse.urlencode(q)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    _last[0] = time.time()
    safe = {k: v for k, v in params.items()}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{endpoint}\t{safe}\t{len(b)}B\t{time.time()-t0:.1f}s\n")
    return b.decode("utf-8", errors="replace")
