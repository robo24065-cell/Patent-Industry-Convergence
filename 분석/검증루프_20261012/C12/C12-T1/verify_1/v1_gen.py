# -*- coding: utf-8 -*-
"""verify_1: KR 행 5G 비중(반영연도별) 재현 + N5 양성 단위 5G 비중. csv 모듈로 zip 직접 읽기."""
import csv, io, zipfile, json, time, sys, os, re
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
t0=time.time()
Z=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext\trade_etsi_ISLD-export.zip"
OUT=os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(10**8)
z=zipfile.ZipFile(Z)
f=io.TextIOWrapper(z.open("ISLD-export.csv"), encoding="utf-8", errors="replace", newline="")
r=csv.reader(f, delimiter=";")
h=next(r); ix={c:i for i,c in enumerate(h)}
iC,iR,iA,i5,i4,i3,i2=ix["Country_Of_Registration"],ix["Reflected_Date"],ix["PATT_APPLICATION_NUMBER"],ix["5G"],ix["4G"],ix["3G"],ix["2G"]
yr=defaultdict(lambda:[0,0,0,0])  # KR rows: total, 5G==1, blank(all 4 empty), 5G==1 among nonblank denominators
vals5=defaultdict(int)
app=defaultdict(dict)  # appl_no -> {refl_date: [any5, anynonblank, any4]}
n=0
for row in r:
    n+=1
    if len(row)<len(h): continue
    if not row[iC][:2]=="KR": continue
    y=row[iR][:4]; d=row[iR][:10].replace("-","")
    g5=row[i5]; vals5[g5]+=1
    blank=(row[i5]=="" and row[i4]=="" and row[i3]=="" and row[i2]=="")
    a=yr[y]; a[0]+=1; a[1]+= (g5=="1"); a[2]+=blank
    pa=row[iA].strip().replace(" ","").replace("-","")
    if len(pa)==13 and pa[:2]=="KR" and pa[2:].isdigit():
        k="10"+pa[2:]
        e=app[k].get(d)
        if e is None: e=app[k][d]=[0,0,0]
        e[0]|=(g5=="1"); e[1]|=(not blank); e[2]|=(row[i4]=="1")
print("rows",n,"secs",round(time.time()-t0))
R1={y:dict(rows=v[0],share5=round(v[1]/v[0],4),blank=round(v[2]/v[0],4),share5_nonblank=round(v[1]/max(1,v[0]-v[2]),4)) for y,v in sorted(yr.items())}
first={}
for k,dd in app.items():
    d=min(dd); e=dd[d]
    any5ever=int(any(v[0] for v in dd.values()))
    first[k]=dict(refl=int(d) if d.isdigit() else None,g5=e[0],nonblank=e[1],g4=e[2],any5ever=any5ever)
json.dump(dict(rows=n,KR_values_5G=dict(vals5),R1=R1),open(os.path.join(OUT,"v1_R1.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
json.dump(first,open(os.path.join(OUT,"v1_first.json"),"w"))
print(json.dumps(R1,ensure_ascii=False)); print("apps",len(first),"secs",round(time.time()-t0))
