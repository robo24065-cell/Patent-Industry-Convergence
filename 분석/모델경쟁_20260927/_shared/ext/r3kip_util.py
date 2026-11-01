# r3kip_ KIPRIS Plus 공용 유틸 (키 값은 출력·저장하지 않음)
import re, time, urllib.request, urllib.parse
KF=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\API키_기록.txt'
BASE='http://plus.kipris.or.kr'
def kipris_key():
    t=open(KF,encoding='utf-8').read()
    sec=t.split('[KIPRIS Plus]')[1]
    m=re.search(r'키\s*[:=]\s*(\S+)',sec); return m.group(1)
KEY=kipris_key()
def get(path, params, keyname='accessKey', timeout=30):
    p=dict(params); p[keyname]=KEY
    url=BASE+path+'?'+urllib.parse.urlencode(p)
    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8','replace')
def mask(s):
    return s.replace(KEY,'<KEY>')
