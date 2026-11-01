import csv,re,collections,sys
ms=open(r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\manuscript_v3.md',encoding='utf-8').read().split('\n')
print('lines',len(ms))
r=list(csv.DictReader(open('errata_status_v18.csv',encoding='utf-8-sig')))
res=collections.Counter(); bad=[]
for x in r:
    ev=x['근거'].split(' ; ')[0]
    parts=[p.strip() for p in ev.split(' / ')]
    for p in parts:
        m=re.match(r'^(\d+)행: (.+)$',p)
        if not m: res['noline']+=1; continue
        ln=int(m.group(1)); t=m.group(2)
        line=ms[ln-1] if ln<=len(ms) else ''
        if t in line: res[('ok',x['상태'])]+=1
        else:
            # maybe elsewhere
            hits=[i+1 for i,l in enumerate(ms) if t in l]
            res[('miss',x['상태'])]+=1; bad.append((x['ID'],x['상태'],ln,hits,t[:80]))
print(res)
for b in bad: print(b)
