# -*- coding: utf-8 -*-
"""r3oa 04: OpenAlex 메인그룹 검색어 → 연도별 논문 수 (group_by=publication_year).
Polite Pool: mailto=사용자 이메일(OpenAlex에만, PROTOCOL_R2 §5.1b 승인). 요청 간 1.2초, 429/5xx 지수 백오프, 403·한도 신호 시 즉시 중단.
variant: ALL(전 세계), company(기업 소속 저자 포함), KR/US/JP/CN/DE(저자국).
실행: python r3oa_04_fetch.py <variant> <max_rank> [guard=40]  — 8분 경과 시 자동 종료(재실행하면 이어서)."""
import json, time, sys, os, urllib.request, urllib.parse, datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3oa_terms import TERMS
W=os.path.dirname(os.path.abspath(__file__)); EXT=os.path.dirname(W)
MAIL='robo24065@gmail.com'
BASE='https://api.openalex.org/works'
UA={'User-Agent':'patent-contest-research/1.0 (mailto:%s)'%MAIL}
VAR=sys.argv[1]; MAXR=int(sys.argv[2]); GUARD=int(sys.argv[3]) if len(sys.argv)>3 else 40
OUTF=os.path.join(EXT,'r3oa_mg_year_%s.parquet'%VAR)
LOG=os.path.join(W,'r3oa_fetch.log')
T0=time.time()
def log(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True)
    open(LOG,'a',encoding='utf-8').write(datetime.datetime.now().isoformat(timespec='seconds')+' '+s+'\n')
def call(params):
    params=dict(params); params['mailto']=MAIL
    url=BASE+'?'+urllib.parse.urlencode(params)
    wait=5
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60) as r:
                return json.load(r), r.headers.get('X-RateLimit-Remaining')
        except urllib.error.HTTPError as e:
            if e.code in (401,402,403):
                log('STOP http',e.code,e.headers.get('X-RateLimit-Remaining')); return None,'STOP'
            if e.code==429:
                rem=e.headers.get('X-RateLimit-Remaining')
                if rem is not None and int(rem)<=0:
                    log('STOP 429 quota exhausted'); return None,'STOP'
            if e.code in (429,500,502,503,504):
                log('retry',e.code,wait); time.sleep(wait); wait*=2; continue
            raise
        except Exception as e:
            log('err',repr(e)[:120],wait); time.sleep(wait); wait*=2
    return None,'FAIL'
def q_of(mg): return '('+' OR '.join('"%s"'%w for w in TERMS[mg])+')'
M=pd.read_csv(os.path.join(W,'r3oa_mg_rank_titles.csv'))
order=[m for m in M.mg.tolist()[:MAXR] if m in TERMS]
done=pd.read_parquet(OUTF) if os.path.exists(OUTF) else pd.DataFrame(columns=['mg','variant','year','n_works','query','fetched'])
have=set(done.mg); rows=done.to_dict('records')
qcache={r['query']:None for r in rows}
byq={}
for r in rows: byq.setdefault(r['query'],{})[int(r['year'])]=int(r['n_works'])
n_calls=0; rem=None
for mg in order:
    if mg in have: continue
    if time.time()-T0>480: log('time cap, stop (resume later)'); break
    q=q_of(mg)
    if q in byq:   # 같은 검색어 재사용(호출 절약)
        got=byq[q]; src='reuse'
    else:
        f='title_and_abstract.search:%s,publication_year:1995-2025'%q
        if VAR=='company': f+=',authorships.institutions.type:company'
        elif VAR!='ALL': f+=',authorships.countries:%s'%VAR
        d,rem=call({'filter':f,'group_by':'publication_year'})
        if d is None: log('stopped at',mg,rem); break
        got={int(g['key']):int(g['count']) for g in d['group_by']}; byq[q]=got; src='api'; n_calls+=1
        time.sleep(1.2)
    now=datetime.date.today().isoformat()
    for y in range(1995,2026): rows.append({'mg':mg,'variant':VAR,'year':y,'n_works':got.get(y,0),'query':q,'fetched':now})
    have.add(mg)
    log(VAR,mg,src,'total',sum(got.values()),'remaining',rem)
    if n_calls%20==0: pd.DataFrame(rows).to_parquet(OUTF,index=False)
    if rem is not None and rem!='STOP' and int(rem)<=GUARD: log('budget guard stop at remaining',rem); break
pd.DataFrame(rows).to_parquet(OUTF,index=False)
log('saved',OUTF,'mg',len(have),'calls this run',n_calls,'last remaining',rem)
