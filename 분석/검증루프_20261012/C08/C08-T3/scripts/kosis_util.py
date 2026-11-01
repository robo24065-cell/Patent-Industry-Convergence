# -*- coding: utf-8 -*-
# KOSIS OpenAPI 호출(C08-T3). 키는 수집파이프라인 파일에서 읽기만 하고 출력·저장하지 않는다(PROTOCOL_R2 5.1b). 2초 간격.
import re, os, time, json, hashlib, urllib.request, urllib.parse, sys
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T3\download\kosis")
_src = open(os.path.join(ROOT, r"수집파이프라인\collect_kosis_new.py"), encoding="utf-8").read()
_KEY = re.search(r'KEY\s*=\s*urllib\.parse\.quote\("([^"]+)"\)', _src).group(1)
_LAST = [0.0]
def call(base, params, tag):
    wait = 2.0 - (time.time() - _LAST[0])
    if wait > 0: time.sleep(wait)
    q = dict(params); q['apiKey'] = _KEY
    url = 'https://kosis.kr/openapi/' + base + '?' + urllib.parse.urlencode(q, safe=':+')
    shown = 'https://kosis.kr/openapi/' + base + '?' + urllib.parse.urlencode(params, safe=':+')
    r = urllib.request.urlopen(url, timeout=60).read(); _LAST[0] = time.time()
    fn = os.path.join(OUT, tag + '.json'); open(fn, 'wb').write(r)
    with open(os.path.join(OUT, '_calllog.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{tag}\t{shown}&apiKey=<KEY>\t{len(r)}\tsha256={hashlib.sha256(r).hexdigest()}\n")
    return r
if __name__ == '__main__':
    for vw in sys.argv[1:]:
        call('statisticsList.do', dict(method='getList', vwCd='MT_OTITLE', parentListId=vw, format='json', jsonVD='Y'), 'list_' + vw)
