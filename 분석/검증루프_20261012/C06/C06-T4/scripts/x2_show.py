import sys,json; sys.stdout.reconfigure(encoding='utf-8')
L=json.load(open(sys.argv[1],encoding='utf-8'))
for k,v in L.items():
    if k not in ('P1','P3'): print(k, v)
