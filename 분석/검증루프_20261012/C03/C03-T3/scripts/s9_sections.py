import sys, re, json; sys.stdout.reconfigure(encoding='utf-8')
W=sys.argv[1]
L=open(W+'/snap/manuscript_v3_snap.md',encoding='utf8').read().split('\n')
sec=[];cur='표지·요약'
for i,l in enumerate(L):
    m=re.match(r'^#{1,3}\s+(\d+(?:\.\d+)*)\s',l)
    if m: cur=m.group(1)
    if l.startswith('## 데이터 출처'): cur='데이터 출처'
    if l.startswith('# [부록'): cur='APPX'
    sec.append(cur)
body=[(i+1,sec[i],L[i]) for i in range(len(L)) if sec[i]!='APPX']
rows={}
for i,l in enumerate(L):
    if sec[i]=='APPX' and l.startswith('| ') and not l.startswith('| id') and not l.startswith('|---'):
        c=[x.strip() for x in l.strip('|').split('|')]
        if len(c)>=4: rows[c[0]]=(i+1,c[1],c[2])
keys={'CIRC-P1-NPS':['26.98'],'CIRC-P1-DART':['61.37'],'C-LAND-cweq_F':['8–16'],'C-LAND-free_F':['8–16','0.20–0.36'],'C-LAND-prov':['25개에서','7개에서'],
 'OFF-DEV2FINAL':['−0.005'],'CIRC-AGREE-CC':['87–90'],'C-T3-AUCS':['0.7867','0.0339'],'B-AUCS':['0.9027','0.9003'],'NTIS-SRC':['NTIS Open API'],
 'LINK-NPS':['44.4%'],'CIRC-BOUND':['약 1%p'],'OFF-T3-TOP100':['50%가 맞았다','50개'],'LBL-CW-GROUP':['41.9'],'OFF-N9-SUB':['0.257'],
 'A-COVER':['9,054'],'NTIS-T3-SUM':['14.2%'],'C-AGREE-KR':['39.2','39.3'],'MAP-KR-TOP':['1.72'],'MAP-LEADERS':['CN 18'],'CIRC-INVONLY':['발명자만'],'MAP-LEADLAG':['400개']}
out=[]
for k,ks in keys.items():
    if k not in rows: out.append(dict(id=k,missing=True)); continue
    ln,used,val=rows[k]
    found=sorted({s for (n,s,t) in body for q in ks if q in t})
    lines=sorted({n for (n,s,t) in body for q in ks if q in t})
    out.append(dict(id=k,appx_line=ln,listed=used,found_sections=found,found_lines=lines[:12],value=val[:120]))
json.dump(out,open(W+'/_s9_sections.json','w',encoding='utf8'),ensure_ascii=False,indent=1)
for o in out: print(o)
