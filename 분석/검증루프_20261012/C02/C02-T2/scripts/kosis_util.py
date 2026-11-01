# KOSIS OpenAPI 호출 도우미. 키는 승인된 수집파이프라인 파일에서 읽기만 하고 출력·저장하지 않는다(PROTOCOL_R2 5.1b).
import re, os, time, json, hashlib, urllib.request, urllib.parse
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\download\kosis")
os.makedirs(OUT, exist_ok=True)
_src = open(os.path.join(ROOT, r"수집파이프라인\collect_kosis_new.py"), encoding="utf-8").read()
_KEY = re.search(r'KEY\s*=\s*urllib\.parse\.quote\("([^"]+)"\)', _src).group(1)
_LAST = [0.0]
def call(base, params, tag):
    """base: 예 'statisticsList.do'. 응답 원문을 download/kosis/<tag>.json 으로 저장, URL은 키를 가린 채 로그."""
    wait = 2.0 - (time.time() - _LAST[0])
    if wait > 0: time.sleep(wait)
    q = dict(params); q['apiKey'] = _KEY
    u = f"https://kosis.kr/openapi/{base}?" + urllib.parse.urlencode(q, safe='+=/')
    for att in range(3):
        try:
            r = urllib.request.urlopen(u, timeout=120).read()
            break
        except Exception as e:
            if att == 2: raise
            time.sleep(5 * (att + 1))
    _LAST[0] = time.time()
    fn = os.path.join(OUT, f"{tag}.json"); open(fn, "wb").write(r)
    shown = f"https://kosis.kr/openapi/{base}?" + urllib.parse.urlencode({k: v for k, v in params.items()})
    with open(os.path.join(OUT, "_calllog.txt"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{tag}\t{shown}&apiKey=<KEY>\t{len(r)}\tsha256={hashlib.sha256(r).hexdigest()}\n")
    try: txt = r.decode('utf-8')
    except UnicodeDecodeError: txt = r.decode('cp949', errors='replace')
    try: return json.loads(txt)
    except Exception: return txt
