# -*- coding: utf-8 -*-
# KOSIS OpenAPI 호출(C10-T1). 키는 수집파이프라인\collect_kosis_new.py 에서 읽기만 하고 출력·저장하지 않는다(PROTOCOL_R2 5.1b). 2초 간격.
import re, os, time, json, hashlib, urllib.request, urllib.parse, sys
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C10\C10-T1\download\kosis")
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
def lst(vw): return call('statisticsList.do', dict(method='getList', vwCd='MT_OTITLE', parentListId=vw, format='json', jsonVD='Y'), 'list_' + vw)
def meta(tbl): return call('statisticsData.do', dict(method='getMeta', type='ITM', orgId='138', tblId=tbl, format='json'), 'meta_' + tbl)
def data(tbl, s, e, objs=('ALL', 'ALL', 'ALL'), tag=None):
    p = dict(method='getList', orgId='138', tblId=tbl, itmId='ALL', format='json', jsonVD='Y', prdSe='Y', startPrdDe=str(s), endPrdDe=str(e))
    for i, o in enumerate(objs): p[f'objL{i+1}'] = o
    return call('Param/statisticsParameterData.do', p, tag or f'data_{tbl}_{s}_{e}')
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'list':
        for vw in sys.argv[2:]: lst(vw)
    elif cmd == 'meta':
        for t in sys.argv[2:]: meta(t)
    elif cmd == 'data':
        data(sys.argv[2], sys.argv[3], sys.argv[4], tuple(sys.argv[5].split(',')) if len(sys.argv) > 5 else ('ALL', 'ALL', 'ALL'))
