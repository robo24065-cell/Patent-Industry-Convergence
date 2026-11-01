# NTIS OpenAPI 공용 헬퍼 — 키는 API키_기록.txt에서 읽고 출력하지 않는다.
import re, time, urllib.request, urllib.parse, pathlib
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
EXT = ROOT / r"분석\모델경쟁_20260927\_shared\ext"
WORK = EXT / "ntis_work"
def _key():
    txt = (ROOT / "API키_기록.txt").read_text(encoding="utf-8")
    sec = txt.split("[NTIS", 1)[1]
    m = re.search(r"^키:\s*(\S+)", sec, re.M)
    return m.group(1)
KEY = _key()
_last = [0.0]
def call(endpoint, params, timeout=60, gap=0.6):
    dt = time.time() - _last[0]
    if dt < gap: time.sleep(gap - dt)
    q = dict(params); q["apprvKey"] = KEY
    url = f"https://www.ntis.go.kr/rndopen/openApi/{endpoint}?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read()
    finally:
        _last[0] = time.time()
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("cp949", errors="replace")
def redact(s):
    return s.replace(KEY, "<KEY>")
