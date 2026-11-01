import sys; sys.stdout.reconfigure(encoding="utf-8")
import difflib, json, sys, csv
W=sys.argv[1]
a=open(W+'/snap/v2.md',encoding='utf8').read().split('\n')
b=open(W+'/snap/manuscript_v3_snap.md',encoding='utf8').read().split('\n')
sm=difflib.SequenceMatcher(None,a,b,autojunk=False)
rows=[]
for tag,i1,i2,j1,j2 in sm.get_opcodes():
    if tag=='equal': continue
    rows.append(dict(tag=tag,v2_lines=f"{i1+1}-{i2}" if i2>i1 else '',v3_lines=f"{j1+1}-{j2}" if j2>j1 else '',before='\n'.join(a[i1:i2]),after='\n'.join(b[j1:j2])))
json.dump(rows,open(W+'/scripts/_diff_raw.json','w',encoding='utf8'),ensure_ascii=False,indent=1)
for r in rows:
    print('=====',r['tag'],'v2',r['v2_lines'],'v3',r['v3_lines'])
    print('<',r['before'][:1500]); print('>',r['after'][:1500])
