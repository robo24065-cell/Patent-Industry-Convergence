# -*- coding: utf-8 -*-
# ECOS OpenAPI 호출(C10-T1). 키는 API키_기록.txt [한국은행 ECOS Open API] 절에서 읽기만, 출력·저장 안 함. 호출 간격 2초.
import re, os, time, hashlib, urllib.request, sys
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C10\C10-T1\download\ecos")
_t = open(os.path.join(ROOT, 'API키_기록.txt'), encoding='utf-8').read()
_sec = _t[_t.index('[한국은행 ECOS Open API]'):]; _KEY = re.search(r'키:\s*([A-Za-z0-9]+)', _sec).group(1)
_LAST = [0.0]
def call(path, tag):
    w = 2.0 - (time.time() - _LAST[0])
    if w > 0: time.sleep(w)
    url = 'https://ecos.bok.or.kr/api/' + path.replace('{KEY}', _KEY)
    r = urllib.request.urlopen(url, timeout=120).read(); _LAST[0] = time.time()
    open(os.path.join(OUT, tag + '.json'), 'wb').write(r)
    with open(os.path.join(OUT, '_calllog.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{tag}\thttps://ecos.bok.or.kr/api/{path.replace('{KEY}', '<KEY>')}\t{len(r)}\tsha256={hashlib.sha256(r).hexdigest()}\n")
    return r
if __name__ == '__main__':
    call(sys.argv[1], sys.argv[2])
