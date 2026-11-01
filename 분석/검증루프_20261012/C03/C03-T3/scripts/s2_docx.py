import sys, json, re, docx
sys.stdout.reconfigure(encoding="utf-8")
W=sys.argv[1]
d=docx.Document(W+'/snap/report_v4_snap.docx')
paras=[p.text for p in d.paragraphs]
for t in d.tables:
    for r in t.rows:
        paras.append(' | '.join(c.text for c in r.cells))
full='\n'.join(paras)
norm=lambda s: re.sub(r'\s+','',re.sub(r'[*_`]|\[\d+\]','',s))
F=norm(full)
md=open(W+'/snap/manuscript_v3_snap.md',encoding='utf8').read().split('\n')
rows=json.load(open(W+'/scripts/_diff_raw.json',encoding='utf8'))
out=[]
for r in rows:
    for ln in r['after'].split('\n'):
        if not ln.strip(): continue
        s=ln.strip()
        if s.startswith('|'):
            cells=[c.strip() for c in s.strip('|').split('|')]
            ok=all(norm(c) in F for c in cells if c)
            miss=[c[:60] for c in cells if c and norm(c) not in F]
        else:
            s2=re.sub(r'^[-#>\s]+','',s)
            s2=re.sub(r'^\d+\.\s*','',s2)
            ok=norm(s2) in F
            miss=[] if ok else [s2[:80]]
        out.append(dict(v3=r['v3_lines'],ok=ok,miss=miss))
for o in out: print(o['v3'],o['ok'],o['miss'])
# also old phrases absent
for old in ['87–91','기준선 26.0%','122,974건(18.7%)','40% 남짓','단순 기준선 3.7%','WIPO IP 통계(공개건수 보정)','모멘텀 기준선 33개']:
    print('OLD',old, norm(old) in F)
print('n_paras',len(d.paragraphs),'appendix_in_docx', norm('숫자 출처표') in F)
