import json,sys
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): yield from walk(v,f'{p}.{k}' if p else str(k))
    elif isinstance(o,list):
        if len(o)<=12 and all(not isinstance(x,(dict,list)) for x in o): yield p,o
        else:
            for i,v in enumerate(o): yield from walk(v,f'{p}[{i}]')
    else: yield p,o
d=json.load(open(sys.argv[1],encoding='utf-8'))
pat=sys.argv[2] if len(sys.argv)>2 else ''
for k,v in walk(d):
    s=str(v)
    if pat.lower() in k.lower(): print(k,'=',s[:200])
