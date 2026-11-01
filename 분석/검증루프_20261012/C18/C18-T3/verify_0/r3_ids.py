import csv,re,collections
E=open(r'..\..\ERRATA.txt',encoding='utf-8').read().split('\n')
r=list(csv.DictReader(open('errata_status_v18.csv',encoding='utf-8-sig')))
S=set(x['ID'] for x in r)
# items that start a line (item headers)
hdr=collections.OrderedDict()
for i,l in enumerate(E,1):
    m=re.match(r'^\s{0,2}(E-C\d{2}[-A-Za-z0-9가-힣·_]*)',l)
    if m:
        if '반영 기록' in l: continue
        hdr.setdefault(m.group(1).rstrip('·'),i)
H=set(hdr)
print('csv',len(S),'errata line-start ids',len(H))
print('in errata not csv:',[ (k,hdr[k]) for k in hdr if k not in S])
print('in csv not errata-start:',sorted(S-H))
# all mentioned ids
allm=set(re.findall(r'E-C\d{2}[A-Za-z0-9\-]*',open(r'..\..\ERRATA.txt',encoding='utf-8').read()))
print('mentioned not in csv:',sorted(x for x in allm-S))
