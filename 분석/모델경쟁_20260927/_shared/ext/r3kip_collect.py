# r3kip_ KIPRIS Plus 인용(citationInfoV3)·피인용(citingInfo) 수집기. 재개 가능, 호출 상한·오류 즉시 중단.
# 보내는 값: 공개 출원번호뿐. 키는 파일에서 읽고 출력하지 않음.
import sys, os, json, time, re, urllib.request, urllib.parse, urllib.error, datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import r3kip_util as u
D=os.path.dirname(os.path.abspath(__file__))
RAW=os.path.join(D,'r3kip_raw'); os.makedirs(RAW,exist_ok=True)
CNT=os.path.join(D,'r3kip_calls.json'); LOG=os.path.join(D,'r3kip_calllog.csv')
CAP=int(os.environ.get('R3KIP_CAP','760'))       # 이번 달 총 호출 상한(시험 호출 포함)
MAXRUN=int(sys.argv[1]) if len(sys.argv)>1 else 300
GAP=1.0
st=json.load(open(CNT)) if os.path.exists(CNT) else {'total':5,'note':'2026-09-29 시험 호출 5회 포함'}
S=pd.read_parquet(os.path.join(D,'r3kip_sample.parquet'))
jobs=[('citing',a) for a in S[S.set=='T4'].applNo]+[('citation',a) for a in S[S.bw].applNo]+[('citing',a) for a in S[S.set=='T3'].applNo]
EP={'citing':('/openapi/rest/CitingService/citingInfo','standardCitationApplicationNumber'),
    'citation':('/openapi/rest/CitationService/citationInfoV3','applicationNumber')}
def done(kind,a): return os.path.exists(os.path.join(RAW,f'{kind}_{a}.xml'))
def call(kind,a):
    path,prm=EP[kind]
    url=u.BASE+path+'?'+urllib.parse.urlencode({prm:a,'accessKey':u.KEY})
    raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40).read()
    return raw.decode('utf-8','replace')
n=0; stop=None
logf=open(LOG,'a',encoding='utf-8')
if os.path.getsize(LOG)==0: logf.write('ts,kind,applNo,status,resultCode,n_items,bytes\n')
for kind,a in jobs:
    if done(kind,a): continue
    if n>=MAXRUN: break
    if st['total']>=CAP: stop=f'CAP {CAP} 도달'; break
    body=None
    for attempt in range(2):
        try:
            st['total']+=1; n+=1
            body=call(kind,a); break
        except urllib.error.HTTPError as e:
            logf.write(f'{datetime.datetime.now().isoformat()},{kind},{a},HTTP{e.code},,0,0\n'); logf.flush()
            if e.code in (401,403,429) or attempt==1: stop=f'HTTP {e.code}'; break
            time.sleep(10)
        except Exception as e:
            logf.write(f'{datetime.datetime.now().isoformat()},{kind},{a},ERR {type(e).__name__},,0,0\n'); logf.flush()
            if attempt==1: stop=f'네트워크 {type(e).__name__}'; break
            time.sleep(10)
    json.dump(st,open(CNT,'w'))
    if stop: break
    rc=(re.search(r'<resultCode>(.*?)</resultCode>',body) or [None,''])[1].strip()
    msg=(re.search(r'<resultMsg>(.*?)</resultMsg>',body) or [None,''])[1].strip()
    tag='citingInfo' if kind=='citing' else 'citationInfoV3'
    ni=body.count(f'<{tag}>')
    logf.write(f'{datetime.datetime.now().isoformat()},{kind},{a},200,{rc},{ni},{len(body)}\n'); logf.flush()
    if rc not in ('','00'):
        stop=f'resultCode {rc} {msg[:80]} (applNo {a})'; break
    open(os.path.join(RAW,f'{kind}_{a}.xml'),'w',encoding='utf-8').write(body)
    time.sleep(GAP)
json.dump(st,open(CNT,'w'))
left=sum(1 for k,a in jobs if not done(k,a))
print(f'이번 실행 호출 {n}, 누적 {st["total"]}, 남은 작업 {left}, 중단사유 {stop}')
