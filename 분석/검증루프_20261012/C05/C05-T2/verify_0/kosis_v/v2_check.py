import json,sys,io,csv
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
d=sys.argv[1]
raw=json.load(open(d+r'\download\kosis_DT_133N_9815A_raw_capture.json',encoding='utf-8'))
t,end=json.JSONDecoder().raw_decode(raw[0]['text'])
rows=[l.split('\t') for l in t.split('\n')]
years=list(range(2015,2026))
def num(s):
    s=s.strip().replace(',','')
    if s in ('-',''): return None
    return int(s)
data={}
biz=None
for c in rows[3:]:
    if len(c)!=200:
        print('skip',len(c),c[:3]); continue
    if c[0]: biz=c[0]
    ind=c[1]
    vals=[num(x) for x in c[2:]]
    data[(biz,ind)]={y:vals[i*18:(i+1)*18] for i,y in enumerate(years)}
print('keys',len(data))
for k in data: print(k)
# compare with CSV
lines=[l for l in open(d+r'\download\kosis_DT_133N_9815A_closures_by_type_industry.csv',encoding='utf-8') if not l.startswith('#')]
r=list(csv.reader(lines,delimiter=';'))
hdr=r[0]; mism=0; n=0
for row in r[1:]:
    k=(row[0],row[1])
    if k not in data: print('missing in raw',k); continue
    for i,y in enumerate(years):
        v=int(row[2+i]); n+=1
        if data[k][y][0]!=v: mism+=1; print('MISMATCH',k,y,v,data[k][y][0])
    # regsum
    for col,y in (('regsum_2025',2025),('regsum_2016',2016)):
        rv=int(row[hdr.index(col)]); s=sum(x for x in data[k][y][1:] if x is not None)
        if rv!=s: print('regsum mismatch',k,y,rv,s)
print('csv cells compared',n,'mismatch',mism)
# all-year region sums & industry sums for 법인
corp=[k for k in data if k[0]=='법인사업자']
print(corp)
for y in years:
    tot=data[('법인사업자','합계')][y]
    regs=sum(x for x in tot[1:] if x is not None)
    inds=sum(data[k][y][0] for k in corp if k[1]!='합계' and data[k][y][0] is not None)
    # each industry region sum
    bad=[k[1] for k in corp if data[k][y][0] is not None and sum(x for x in data[k][y][1:] if x is not None)!=data[k][y][0]]
    print(y,'total',tot[0],'regsum',regs,'indsum',inds,'ind-region mismatches',bad)
# overall: sum over biz types equals 합계?
bizs=sorted(set(k[0] for k in data)); print(bizs)
for y in years:
    s=sum(data[(b,'합계')][y][0] for b in bizs if b!='합계' and (b,'합계') in data)
    print(y,'grand',data[('합계','합계')][y][0],'sum of biz types',s)
# 2825 comparison
off={2016:69,2017:70,2018:70,2019:70,2020:67,2021:65,2022:67,2023:76,2024:83,2025:85}
for y,v in off.items():
    x=data[('법인사업자','합계')][y][0]; print(y,x,v,round(x/1000/v-1,4))
