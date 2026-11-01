import zipfile, re, sys, os, json
d=sys.argv[1]; out={}
for v in ['20210101','20220101','20230101','20240101','20250101','20260101']:
    z=zipfile.ZipFile(os.path.join(d,f'EN_ipc_title_list_{v}.zip'))
    subs=set(); names=z.namelist()
    for n in names:
        txt=z.read(n).decode('utf-8','replace')
        for line in txt.splitlines():
            m=re.match(r'^([A-H]\d\d[A-Z])\t',line)
            if m: subs.add(m.group(1))
    out[v]={'files':names[:5],'n_sub':len(subs),'H10':sorted(s for s in subs if s.startswith('H10')),'H01L' in subs and 'H01L':1}
    print(v,len(subs),sorted(s for s in subs if s.startswith('H10')),'H01L' in subs, names[:3])
